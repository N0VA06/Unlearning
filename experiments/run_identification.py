"""End-to-end driver for the data-identification experiment.

For each of the 4 same-domain dataset pairs (Q1..Q4):

    1. Load the combined JSON.
    2. Compute base-GPT-2 embeddings + uncertainty signals for every label=1 row.
    3. Pick an informative (uncertain + diverse) training subset via k-means
       clusters with max-uncertainty medoid per cluster.
    4. Strict leakage audit on train vs. eval (zero overlap on row idx,
       (source,id), exact text, normalized text).
    5. Train LoRA on the chosen training rows (target only).
    6. Score the held-out eval set (held-out target + all forget) with:
         - PPL ratio   (base vs LoRA)
         - Embedding   (mean-pooled LoRA last-layer hidden state, cosine vs centroid)
         - Combined    (0.4 · ppl + 0.6 · emb, both min-max'd)
    7. Compute AUROC / AP / F1 / KS / overlap / P@K metrics.
    8. Run layerwise activation-drift analysis (cosine distance between base
       and LoRA hidden states, per layer, per class).
    9. Save metrics + audit + drift JSONs and the per-quadrant plot set.
   10. Print a summary table and observations across quadrants.
"""

# ---- Suppress third-party load noise BEFORE any heavy imports ----------------
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL",              "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS",             "0")
os.environ.setdefault("BITSANDBYTES_NOWELCOME",            "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM",            "false")

import sys
sys.modules["bitsandbytes"] = None   # negative cache — keeps the bnb traceback away

import json
import warnings
import logging
from pathlib import Path

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("peft").setLevel(logging.ERROR)

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import transformers
transformers.logging.set_verbosity_error()

from config import QUADRANTS, RESULTS, PLOTS, SPLIT_RATIO, SPLIT_SEED
from src.data_loader    import load_quadrant, make_informative_split, audit_split
from src.lora_trainer   import train_lora
from src.scorer         import (
    compute_ppl_ratio_scores, compute_embedding_scores, compute_combined_scores,
)
from src.metrics        import compute_all_metrics
from src.plotting       import plot_all, plot_summary, plot_layerwise_drift
from src.probe          import run_layerwise_drift
from src.sample_select  import gather_selection_inputs


METHODS = ["ppl", "emb", "combined"]
METHOD_PRETTY = {"ppl": "PPL ratio", "emb": "Embedding", "combined": "Combined"}


def print_metrics_table(all_results):
    print("\nEval set composition per quadrant (held-out target + all forget):")
    for qid in ["q1", "q2", "q3", "q4"]:
        sp = all_results[qid]["split"]
        print(f"  {QUADRANTS[qid]['label']:<14}  "
              f"train_target={sp['train_target_n']}  "
              f"heldout_target={sp['heldout_target_n']}  "
              f"forget={sp['forget_n']}")

    header = (
        f"{'Quadrant':<12} | {'Method':<10} | {'AUROC':>5} | {'AP':>5} | "
        f"{'F1':>5} | {'KS':>5} | {'Overlap':>7} | {'Cohen d':>7} | "
        f"{'P@10':>5} | {'P@50':>5}"
    )
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for qid in ["q1", "q2", "q3", "q4"]:
        qlabel = QUADRANTS[qid]["label"]
        for method in METHODS:
            m = all_results[qid]["metrics"][method]
            p10 = m["p_at_k"].get("10", float("nan"))
            p50 = m["p_at_k"].get("50", float("nan"))
            print(
                f"{qlabel:<12} | {METHOD_PRETTY[method]:<10} | "
                f"{m['auroc']:>5.3f} | {m['ap']:>5.3f} | "
                f"{m['f1']:>5.3f} | {m['ks_stat']:>5.3f} | "
                f"{m['overlap']:>7.3f} | {m['cohens_d']:>7.2f} | "
                f"{p10:>5.3f} | {p50:>5.3f}"
            )
    print("=" * len(header))


def print_observations(all_results):
    print("\n" + "#" * 70)
    print("# OBSERVATIONS  (computed from results — never hardcoded)")
    print("#" * 70)
    qids = ["q1", "q2", "q3", "q4"]

    # Best/worst quadrant
    mean_auroc_per_q = {
        q: float(np.nanmean([all_results[q]["metrics"][m]["auroc"] for m in METHODS]))
        for q in qids
    }
    valid = {q: v for q, v in mean_auroc_per_q.items() if not np.isnan(v)}
    if valid:
        best_q  = max(valid, key=valid.get)
        worst_q = min(valid, key=valid.get)
        print(f"\nHighest mean AUROC: {QUADRANTS[best_q]['label']} ({valid[best_q]:.3f})")
        print(f"Lowest  mean AUROC: {QUADRANTS[worst_q]['label']} ({valid[worst_q]:.3f})")

    # Per-quadrant PPL vs Embedding winner
    print("\nPer-quadrant: PPL ratio vs Embedding similarity (higher AUROC wins)")
    for qid in qids:
        m = all_results[qid]["metrics"]
        p, e = m["ppl"]["auroc"], m["emb"]["auroc"]
        winner = "PPL" if p > e else ("Embedding" if e > p else "tie")
        if np.isnan(p) or np.isnan(e):
            winner = "n/a"
        print(f"  {QUADRANTS[qid]['label']:<14}  PPL={p:.3f}  EMB={e:.3f}  →  {winner}")

    # Combined improvement
    print("\nCombined score improvement over both PPL and Embedding:")
    for qid in qids:
        m = all_results[qid]["metrics"]
        c, p, e = m["combined"]["auroc"], m["ppl"]["auroc"], m["emb"]["auroc"]
        verdict = "YES" if (not np.isnan(c) and c > p and c > e) else "no"
        print(f"  {QUADRANTS[qid]['label']:<14}  "
              f"Combined={c:.3f}  (PPL={p:.3f}, EMB={e:.3f})  →  {verdict}")

    # Near-random warnings
    print("\nWarnings — near-random AUROC (< 0.55):")
    any_warn = False
    for qid in qids:
        for method in METHODS:
            a = all_results[qid]["metrics"][method]["auroc"]
            if not np.isnan(a) and a < 0.55:
                any_warn = True
                print(f"  [WARNING] {METHOD_PRETTY[method]} on {qid} ≈ random "
                      f"(AUROC={a:.3f})")
    if not any_warn:
        print("  (none)")

    # Drift summary
    print("\nLayerwise activation drift (cosine distance, base vs LoRA per class):")
    header_p = (f"{'':<14} | {'best Δ layer':>12} | "
                f"{'Δ(t-f) cos':>11} | {'target_drift':>13} | {'forget_drift':>13}")
    print(header_p)
    print("-" * len(header_p))
    for qid in qids:
        d = all_results[qid].get("drift", {})
        if not d:
            continue
        layer = d["best_diff_layer"]
        print(f"{QUADRANTS[qid]['label']:<14} | "
              f"{layer:>12d} | "
              f"{d['delta_per_layer'][layer]:>+11.4f} | "
              f"{d['target_drift_per_layer'][layer]:>13.4f} | "
              f"{d['forget_drift_per_layer'][layer]:>13.4f}")

    # Overlap matrix
    print("\nOverlap coefficient (4 quadrants × 3 methods):")
    header_o = f"{'':<14} | " + " | ".join(f"{METHOD_PRETTY[m]:>10}" for m in METHODS)
    print(header_o)
    print("-" * len(header_o))
    for qid in qids:
        row = f"{QUADRANTS[qid]['label']:<14} | " + " | ".join(
            f"{all_results[qid]['metrics'][m]['overlap']:>10.3f}" for m in METHODS
        )
        print(row)


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}  (split_ratio={SPLIT_RATIO}, seed={SPLIT_SEED})")

    all_results = {}
    for qid in ["q1", "q2", "q3", "q4"]:
        print(f"\n{'=' * 60}")
        print(f"  Processing {qid} — {QUADRANTS[qid]['label']}")
        print(f"{'=' * 60}")

        df = load_quadrant(qid)

        # Active-learning selection inputs: base-model embeddings + uncertainty
        target_rows = df[df["quadrant_label"] == 1].reset_index(drop=True)
        target_signals = gather_selection_inputs(
            target_rows["text"].tolist(), device,
            include_grad_norm=True,
        )

        train_target_idx, eval_idx = make_informative_split(df, target_signals)
        audit = audit_split(df, train_target_idx, eval_idx)
        with (RESULTS / f"{qid}_{QUADRANTS[qid]['name']}_audit.json").open("w") as f:
            json.dump(audit, f, indent=2)

        train_target_df = df.iloc[train_target_idx].reset_index(drop=True)
        eval_df         = df.iloc[eval_idx].reset_index(drop=True)

        ckpt        = train_lora(qid, train_target_df)
        ppl_results = compute_ppl_ratio_scores(eval_df, ckpt, device)
        emb_results = compute_embedding_scores(eval_df, train_target_df, ckpt, device)

        score_combined = compute_combined_scores(
            ppl_results["score_ppl"], emb_results["score_emb"]
        )
        scores_dict = {
            "ppl":      ppl_results["score_ppl"],
            "emb":      emb_results["score_emb"],
            "combined": score_combined,
        }

        metrics = compute_all_metrics(eval_df["quadrant_label"].values, scores_dict, qid)
        plot_all(qid, eval_df, scores_dict, emb_results["embeddings"], metrics)

        print(f"\n[run] {qid}: layerwise activation-drift analysis")
        drift = run_layerwise_drift(eval_df, ckpt, device)
        with (RESULTS / f"{qid}_{QUADRANTS[qid]['name']}_drift.json").open("w") as f:
            json.dump(drift, f, indent=2)
        plot_layerwise_drift(qid, drift)

        all_results[qid] = {
            "df":      eval_df,
            "scores":  scores_dict,
            "metrics": metrics,
            "drift":   drift,
            "split": {
                "train_target_n":   int(len(train_target_df)),
                "heldout_target_n": int((eval_df["quadrant_label"] == 1).sum()),
                "forget_n":         int((eval_df["quadrant_label"] == 0).sum()),
            },
        }

    plot_summary(all_results)
    print_metrics_table(all_results)
    print_observations(all_results)


if __name__ == "__main__":
    main()
