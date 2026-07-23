"""Stage 2 + Stage 8 driver: compute each overlap estimator independently on the
unlabeled pool U and report target–forget separability.

For each quadrant it:
  1. Loads the combined data and the EXISTING LoRA checkpoint (no retrain).
  2. Builds a leakage-free 3-way split from that checkpoint's train keys
     (lora_train / P / held-out) via data_loader.make_pu_split.
  3. Computes PPL ratio, Min-K++, and embedding scores on U (held-out target +
     all forget). LiRA is added when shadow adapters are available
     (--with-lira; use --train-shadows to build them first).
  4. Reports AUROC / AUPRC / FPR@95TPR per estimator and writes a JSON.

Usage:
    python experiments/run_overlap_scorers.py                 # PPL, MinK++, embedding
    python experiments/run_overlap_scorers.py --quadrants q2  # single quadrant
    python experiments/run_overlap_scorers.py --with-lira --train-shadows
"""

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import sys
sys.modules["bitsandbytes"] = None

import json
import argparse
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

from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
)

from config import (
    QUADRANTS, RESULTS, CHECKPOINTS, SPLIT_RATIO, SPLIT_SEED, TRAIN_EPOCHS,
    MINK_FRAC,
)
from src.data_loader import load_quadrant, load_train_keys, make_pu_split
from src.overlap_scorers import (
    token_logp_features, ppl_ratio_from_features, minkpp_from_features,
    compute_embedding_scores, compute_lira_scores,
)


def _checkpoint_dir(qid: str) -> Path:
    tag = f"r{int(round(SPLIT_RATIO * 100)):02d}_s{SPLIT_SEED}_e{TRAIN_EPOCHS}"
    return CHECKPOINTS / f"{qid}_{QUADRANTS[qid]['name']}_{tag}"


def _fpr_at_tpr(labels, scores, target_tpr=0.95):
    if len(np.unique(labels)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(labels, scores)
    ok = np.where(tpr >= target_tpr)[0]
    return float(fpr[ok[0]]) if len(ok) else 1.0


def _best_f1(labels, scores):
    """Precision/recall/F1 at the threshold that maximises F1, plus that
    threshold. Threshold-free metrics (AUROC/AUPRC) don't need this; F1/P/R do."""
    prec, rec, thr = precision_recall_curve(labels, scores)
    # prec/rec have len == len(thr)+1; drop the trailing (recall=0) point
    p, r = prec[:-1], rec[:-1]
    f1 = np.where((p + r) > 0, 2 * p * r / (p + r + 1e-12), 0.0)
    if len(f1) == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    j = int(np.argmax(f1))
    return float(f1[j]), float(p[j]), float(r[j]), float(thr[j])


def evaluate(labels, scores):
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=np.float64)
    valid = ~np.isnan(scores)
    n_drop = int((~valid).sum())
    y, s = labels[valid], scores[valid]
    if len(np.unique(y)) < 2:
        nan = float("nan")
        return {"auroc": nan, "auprc": nan, "fpr_at_95tpr": nan,
                "f1": nan, "precision": nan, "recall": nan, "threshold": nan,
                "n_used": int(valid.sum()), "n_dropped": n_drop}
    f1, prec, rec, thr = _best_f1(y, s)
    return {
        "auroc":        float(roc_auc_score(y, s)),
        "auprc":        float(average_precision_score(y, s)),
        "fpr_at_95tpr": _fpr_at_tpr(y, s),
        "f1":           f1,
        "precision":    prec,
        "recall":       rec,
        "threshold":    thr,
        "n_used":       int(valid.sum()),
        "n_dropped":    n_drop,
    }


def run_quadrant(qid, device, with_lira=False, train_shadows=False):
    print(f"\n{'=' * 60}\n  Overlap estimators — {qid} ({QUADRANTS[qid]['label']})\n{'=' * 60}")
    df = load_quadrant(qid)

    ckpt = _checkpoint_dir(qid)
    if not (ckpt / "adapter_config.json").exists():
        print(f"[skip] no LoRA checkpoint at {ckpt} — run run_identification.py first")
        return None

    keys = load_train_keys(ckpt)
    split = make_pu_split(df, lora_train_keys=keys)
    u_df = df.iloc[split["u_idx"]].reset_index(drop=True)
    p_df = df.iloc[split["p_idx"]].reset_index(drop=True)
    labels = u_df["quadrant_label"].values.astype(int)

    scores = {}

    # PPL ratio + Min-K++ share one token-level pass
    feats = token_logp_features(u_df, ckpt, device, need_calib=True)
    scores["ppl_ratio"] = ppl_ratio_from_features(feats)
    scores["minkpp"]    = minkpp_from_features(feats, frac=MINK_FRAC)

    # Embedding similarity to the P centroid
    scores["embedding"] = compute_embedding_scores(u_df, p_df, ckpt, device)

    # LiRA (optional)
    if with_lira:
        from src.shadow_lora import train_shadow_loras, load_shadow_loras
        # Shadows may only train on labeled positives (lora_train ∪ P), which are
        # disjoint from U — otherwise the IN branch leaks U's ground-truth labels.
        pos_pool_idx = np.concatenate([split["lora_train_idx"], split["p_idx"]])
        pool_df = df.iloc[pos_pool_idx].reset_index(drop=True)
        shadows = (train_shadow_loras(qid, pool_df) if train_shadows
                   else load_shadow_loras(qid))
        if len(shadows) == 0:
            print("[lira] no shadow adapters found — skipping (use --train-shadows)")
        else:
            ckpts = [c for c, _ in shadows]
            key_sets = [k for _, k in shadows]
            u_keys = set(zip(u_df["source"], u_df["id"]))
            leaked = u_keys & set().union(*key_sets)
            if leaked:
                raise RuntimeError(
                    f"LEAKAGE: {len(leaked)} U rows appear in shadow training sets — "
                    f"delete checkpoints/shadows/{qid}_* and retrain with the fixed pool")
            print(f"[lira] shadow pool={len(pool_df)} labeled positives; "
                  f"U∩shadow-train = 0 (leak-free)")
            scores["lira_lite"] = compute_lira_scores(
                u_df, ckpt, ckpts, key_sets, device, mode="lite")
            scores["lira_full"] = compute_lira_scores(
                u_df, ckpt, ckpts, key_sets, device, mode="full")

    metrics = {name: evaluate(labels, s) for name, s in scores.items()}

    print(f"\n{'Estimator':<12} | {'AUROC':>6} | {'AUPRC':>6} | {'F1':>5} | "
          f"{'Prec':>5} | {'Rec':>5} | {'FPR@95':>7} | used")
    print("-" * 72)
    for name, m in metrics.items():
        print(f"{name:<12} | {m['auroc']:>6.3f} | {m['auprc']:>6.3f} | "
              f"{m['f1']:>5.3f} | {m['precision']:>5.3f} | {m['recall']:>5.3f} | "
              f"{m['fpr_at_95tpr']:>7.3f} | {m['n_used']}")

    out = {
        "qid": qid,
        "split": {k: int(len(v)) for k, v in split.items()},
        "n_pos_in_u": int((labels == 1).sum()),
        "n_neg_in_u": int((labels == 0).sum()),
        "metrics": metrics,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS / f"{qid}_{QUADRANTS[qid]['name']}_overlap_scores.json"
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"[overlap] wrote {out_path}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quadrants", nargs="+", default=["q1", "q2", "q3", "q4"])
    ap.add_argument("--with-lira", action="store_true")
    ap.add_argument("--train-shadows", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device={device}  MINK_FRAC={MINK_FRAC}")

    for qid in args.quadrants:
        run_quadrant(qid, device, with_lira=args.with_lira,
                     train_shadows=args.train_shadows)


if __name__ == "__main__":
    main()
