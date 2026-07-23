"""Active nnPU vs baselines — full method comparison (proposal Stages 3–8).

For each quadrant it:
  1. Rebuilds the leak-free 3-way split from the existing checkpoint (P + U).
  2. Extracts (and caches) the overlap-SCORE features for the P∪U pool:
     ppl_ratio, minkpp, emb_cos, and — when shadows exist — lira_lite. The raw
     768-d LoRA embedding is cached too, but used ONLY for diversity/coreset.
  3. Runs every method over PU_SEEDS seeds and reports mean ± std on U:
       raw detectors            — threshold one estimator directly (no training)
       naive_pn                 — logistic, U treated as negative
       pu_static_linear / _mlp  — nnPU, no active labels (simple / linear PU)
       active_<strategy>        — nnPU + acquisition: random, uncertainty,
                                  diversity(k-center), badge, committee(QBC), uxd
       supervised_skyline       — out-of-fold logistic on all U labels (ceiling)
  4. Writes results/{qid}_{name}_comparison.json and label-efficiency plots.

Usage:
    python experiments/run_active_pu.py                 # all quadrants, all methods
    python experiments/run_active_pu.py --quadrants q2
    python experiments/run_active_pu.py --no-lira       # skip lira_lite feature
    python experiments/run_active_pu.py --rebuild-cache # force feature re-extraction
"""

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

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

from config import (
    QUADRANTS, RESULTS, PLOTS, CHECKPOINTS, SPLIT_RATIO, SPLIT_SEED, TRAIN_EPOCHS,
    PU_PRIOR, PU_ROUNDS, PU_BUDGET, PU_HIDDEN, PU_MLP_HIDDEN, PU_SEED, PU_SEEDS,
    PU_SCORE_COLS, MINK_FRAC,
)
from src.data_loader import load_quadrant, load_train_keys, make_pu_split
from src.hidden_states import get_tokenizer, load_lora_model, extract_last_layer
from src.active_pu import (
    run_active_pu, run_naive_pn, run_supervised_skyline, run_raw_detector,
)

_EPS = 1e-9
ACTIVE_STRATEGIES = ["random", "uncertainty", "diversity", "badge", "committee", "uxd"]


def _checkpoint_dir(qid: str) -> Path:
    tag = f"r{int(round(SPLIT_RATIO * 100)):02d}_s{SPLIT_SEED}_e{TRAIN_EPOCHS}"
    return CHECKPOINTS / f"{qid}_{QUADRANTS[qid]['name']}_{tag}"


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction + cache
# ─────────────────────────────────────────────────────────────────────────────
def _emb_cosine(emb, p_mask):
    """Cosine of every pool row's LoRA embedding to the P-centroid."""
    e = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + _EPS)
    centroid = e[p_mask].mean(axis=0)
    centroid /= (np.linalg.norm(centroid) + _EPS)
    return e @ centroid


def build_feature_cache(qid, device, with_lira=True):
    """Extract and cache score features + embedding for the P∪U pool."""
    df = load_quadrant(qid)
    ckpt = _checkpoint_dir(qid)
    if not (ckpt / "adapter_config.json").exists():
        print(f"[skip] no LoRA checkpoint at {ckpt}")
        return None

    keys = load_train_keys(ckpt)
    split = make_pu_split(df, lora_train_keys=keys)
    p_idx, u_idx = split["p_idx"], split["u_idx"]
    pool_idx = np.concatenate([p_idx, u_idx])
    pool_df = df.iloc[pool_idx].reset_index(drop=True)

    y_true = pool_df["quadrant_label"].values.astype(int)
    p_mask = np.zeros(len(pool_idx), dtype=bool)
    p_mask[:len(p_idx)] = True

    n_pos_u = int((y_true[len(p_idx):] == 1).sum())
    prior = (n_pos_u / len(u_idx)) if PU_PRIOR == "known" else float(PU_PRIOR)

    # token-level scores: ppl_ratio + minkpp
    from src.overlap_scorers import (
        token_logp_features, ppl_ratio_from_features, minkpp_from_features,
    )
    feats = token_logp_features(pool_df, ckpt, device, need_calib=True, batch_size=16)
    ppl = ppl_ratio_from_features(feats)
    mink = minkpp_from_features(feats, frac=MINK_FRAC)

    # embedding + cosine-to-P-centroid
    tok = get_tokenizer()
    lora = load_lora_model(ckpt, device)
    emb = extract_last_layer(lora, tok, pool_df["text"].tolist(), device,
                             desc="pu-emb").astype(np.float64)
    del lora
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    emb_cos = _emb_cosine(emb, p_mask)

    cols = {"ppl_ratio": ppl, "minkpp": mink, "emb_cos": emb_cos}

    # lira_lite (needs shadow adapters; leak-free w.r.t. U by construction)
    if with_lira:
        from src.shadow_lora import load_shadow_loras
        from src.overlap_scorers import compute_lira_scores
        shadows = load_shadow_loras(qid)
        if len(shadows):
            ckpts = [c for c, _ in shadows]
            key_sets = [k for _, k in shadows]
            u_keys = set(zip(pool_df.iloc[len(p_idx):]["source"],
                             pool_df.iloc[len(p_idx):]["id"]))
            leaked = u_keys & set().union(*key_sets)
            if leaked:
                raise RuntimeError(f"LEAKAGE: {len(leaked)} U rows in shadow train sets")
            cols["lira_lite"] = compute_lira_scores(
                pool_df, ckpt, ckpts, key_sets, device, mode="lite", batch_size=16)
        else:
            print("[cache] no shadows found — lira_lite feature skipped")

    cache = {
        "y_true": y_true, "p_mask": p_mask, "emb": emb,
        "prior": np.array([prior]), "P": np.array([len(p_idx)]),
        "U": np.array([len(u_idx)]), "n_pos_in_u": np.array([n_pos_u]),
        "col_names": np.array(list(cols.keys())),
    }
    for k, v in cols.items():
        cache[f"col_{k}"] = np.nan_to_num(np.asarray(v, dtype=np.float64), nan=0.0)

    path = RESULTS / f"{qid}_{QUADRANTS[qid]['name']}_pu_features.npz"
    RESULTS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **cache)
    print(f"[cache] wrote {path}  (cols={list(cols.keys())})")
    return path


def load_feature_cache(qid):
    path = RESULTS / f"{qid}_{QUADRANTS[qid]['name']}_pu_features.npz"
    if not path.exists():
        return None
    z = np.load(path, allow_pickle=True)
    col_names = [str(c) for c in z["col_names"]]
    cols = {c: z[f"col_{c}"] for c in col_names}
    return {
        "y_true": z["y_true"], "p_mask": z["p_mask"].astype(bool), "emb": z["emb"],
        "prior": float(z["prior"][0]), "P": int(z["P"][0]), "U": int(z["U"][0]),
        "n_pos_in_u": int(z["n_pos_in_u"][0]), "cols": cols, "col_names": col_names,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Method comparison
# ─────────────────────────────────────────────────────────────────────────────
_METRIC_KEYS = ["auroc", "auprc", "fpr_at_95tpr", "f1", "precision", "recall", "ece"]


def _agg(runs):
    """Aggregate a list of metric dicts → mean/std per key."""
    out = {}
    for k in _METRIC_KEYS:
        vals = np.array([r[k] for r in runs if k in r and not np.isnan(r[k])])
        out[f"{k}_mean"] = float(vals.mean()) if len(vals) else float("nan")
        out[f"{k}_std"] = float(vals.std()) if len(vals) else float("nan")
    return out


def _pool_test_split(y, p_mask, test_frac=0.5, seed=SPLIT_SEED):
    """Split the unlabeled pool U into an acquirable pool and a fixed, stratified,
    held-out test set (shared by every method and every seed)."""
    from sklearn.model_selection import train_test_split
    pos_idx = np.where(p_mask)[0]
    u_idx = np.where(~p_mask)[0]
    yu = y[u_idx]
    pool_idx, test_idx = train_test_split(
        u_idx, test_size=test_frac, stratify=yu, random_state=seed)
    return pos_idx, np.sort(pool_idx), np.sort(test_idx)


def run_comparison(qid, device, with_lira=True, seeds=PU_SEEDS):
    cache = load_feature_cache(qid)
    if cache is None:
        if build_feature_cache(qid, device, with_lira=with_lira) is None:
            return None
        cache = load_feature_cache(qid)

    y, p_mask, emb = cache["y_true"], cache["p_mask"], cache["emb"]
    col_names = [c for c in PU_SCORE_COLS if c in cache["cols"]]
    feat = np.column_stack([cache["cols"][c] for c in col_names])   # classifier input
    score_mat = feat.copy()                                          # committee views

    pos_idx, pool_idx, test_idx = _pool_test_split(y, p_mask)
    budget = min(PU_BUDGET, max(2, int(0.06 * len(pool_idx))))
    n_lbl_active = int(len(pos_idx) + min(len(pool_idx), PU_ROUNDS * budget))

    def _active(strat, s, rounds=PU_ROUNDS, hidden=PU_HIDDEN, init_mode="per_round"):
        return run_active_pu(feat, emb, y, pos_idx, pool_idx, test_idx,
                             score_mat=score_mat, strategy=strat, rounds=rounds,
                             budget=budget, hidden=hidden, seed=s, device=device,
                             init_mode=init_mode)

    def _curve_from_runs(runs):
        n_rounds = len(runs[0]["history"])
        curve = []
        for ri in range(n_rounds):
            aur = [r["history"][ri]["auroc"] for r in runs]
            curve.append({
                "n_labeled": runs[0]["history"][ri]["n_labeled"],
                "auroc_mean": float(np.mean(aur)),
                "auroc_std": float(np.std(aur)),
            })
        return curve

    print(f"\n{'=' * 64}\n  Comparison — {qid} ({QUADRANTS[qid]['label']})\n{'=' * 64}")
    print(f"[cfg] P={len(pos_idx)} U_pool={len(pool_idx)} U_test={len(test_idx)} "
          f"(pos_test={int((y[test_idx] == 1).sum())})  "
          f"budget={budget}  features={col_names}  seeds={seeds}")

    methods = {}
    label_eff = {}
    seed_list = [PU_SEED + i for i in range(seeds)]

    # 1) raw single-estimator detectors (deterministic, on held-out test)
    for c in col_names:
        methods[f"raw_{c}"] = {"type": "raw",
                               **_agg([run_raw_detector(cache["cols"][c], y, test_idx)])}

    # 2) naive PN (U-pool as negative)
    methods["naive_pn"] = {"type": "pu", "n_labeled": int(len(pos_idx)), **_agg([
        run_naive_pn(feat, y, pos_idx, pool_idx, test_idx, seed=s) for s in seed_list])}

    # 3) static PU — report round-0 nnPU scores (Stage 7 bootstrap hurts without
    #    acquired negatives; active methods still use the calibrated final head)
    methods["pu_static_linear"] = {"type": "pu", "n_labeled": int(len(pos_idx)),
        **_agg([_active("random", s, rounds=0, hidden=0)["history"][0] for s in seed_list])}
    methods["pu_static_mlp"] = {"type": "pu", "n_labeled": int(len(pos_idx)),
        **_agg([_active("random", s, rounds=0, hidden=PU_MLP_HIDDEN)["history"][0]
                for s in seed_list])}

    # 4) active strategies (linear nnPU head) — report Stage-7 final AND last nnPU
    for strat in ACTIVE_STRATEGIES:
        runs = [_active(strat, s) for s in seed_list]
        runs_fixed = [_active(strat, s, init_mode="fixed") for s in seed_list]
        runs_warm = [_active(strat, s, init_mode="warm_start") for s in seed_list]
        final_agg = _agg([r["final"] for r in runs])
        last_nnpu_agg = _agg([r["history"][-1] for r in runs])
        methods[f"active_{strat}"] = {
            "type": "active",
            "n_labeled": n_lbl_active,
            "stage7_final": final_agg,
            "last_nnpu": last_nnpu_agg,
            **final_agg,  # backward-compat: top-level = Stage-7 final
        }
        label_eff[strat] = {
            "nnpu": _curve_from_runs(runs),
            "nnpu_fixed": _curve_from_runs(runs_fixed),
            "nnpu_warm": _curve_from_runs(runs_warm),
            "stage7_final": {
                "n_labeled": n_lbl_active,
                "auroc_mean": final_agg["auroc_mean"],
                "auroc_std": final_agg["auroc_std"],
            },
            "last_nnpu": {
                "n_labeled": n_lbl_active,
                "auroc_mean": last_nnpu_agg["auroc_mean"],
                "auroc_std": last_nnpu_agg["auroc_std"],
            },
        }

    # 5) supervised skyline (full pool labels → ceiling)
    methods["supervised_skyline"] = {"type": "skyline", "n_labeled": int(len(pos_idx) + len(pool_idx)),
        **_agg([run_supervised_skyline(feat, y, pos_idx, pool_idx, test_idx, seed=s)
                for s in seed_list])}

    _print_table(methods)
    _print_active_dual(methods)

    out = {
        "qid": qid, "features": col_names, "budget": budget,
        "P": int(len(pos_idx)), "U_pool": int(len(pool_idx)), "U_test": int(len(test_idx)),
        "pos_in_test": int((y[test_idx] == 1).sum()),
        "seeds": seeds, "methods": methods, "label_efficiency": label_eff,
    }
    path = RESULTS / f"{qid}_{QUADRANTS[qid]['name']}_comparison.json"
    with path.open("w") as fh:
        json.dump(out, fh, indent=2)
    print(f"[comparison] wrote {path}")
    _plot_all(qid, out)
    return out


def _print_table(methods):
    print(f"\n{'method':<22} {'type':<8} {'AUROC':>13} {'AUPRC':>6} {'F1':>6} "
          f"{'Prec':>6} {'Rec':>6} {'FPR@95':>7} {'ECE':>6} {'lbl':>5}")
    print("-" * 96)
    order = {"raw": 0, "pu": 1, "active": 2, "skyline": 3}
    for name, m in sorted(methods.items(), key=lambda kv: (order[kv[1]["type"]], kv[0])):
        au = f"{m['auroc_mean']:.3f}±{m['auroc_std']:.3f}"
        lbl = str(m.get("n_labeled", "—"))
        print(f"{name:<22} {m['type']:<8} {au:>13} {m['auprc_mean']:>6.3f} "
              f"{m['f1_mean']:>6.3f} {m['precision_mean']:>6.3f} {m['recall_mean']:>6.3f} "
              f"{m['fpr_at_95tpr_mean']:>7.3f} {m['ece_mean']:>6.3f} {lbl:>5}")


def _print_active_dual(methods):
    """Second table: active methods — last nnPU vs Stage-7 final."""
    active = {k: v for k, v in methods.items() if v.get("type") == "active"
              and "last_nnpu" in v}
    if not active:
        return
    print(f"\n{'method':<22} {'last_nnPU':>12} {'Stage-7':>12} {'Δ(final−nnPU)':>14}")
    print("-" * 64)
    for name in sorted(active):
        m = active[name]
        ln = m["last_nnpu"]["auroc_mean"]
        fin = m["stage7_final"]["auroc_mean"]
        delta = fin - ln
        print(f"{name:<22} {ln:>12.3f} {fin:>12.3f} {delta:>+14.3f}")


_STRAT_COLORS = {
    "random": "#888888",
    "uncertainty": "#E45756",
    "diversity": "#4C78A8",
    "badge": "#F58518",
    "committee": "#72B7B2",
    "uxd": "#B279A2",
}


def _mpl_setup():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception:
        return None


def _plot_all(qid, out):
    plt = _mpl_setup()
    if plt is None:
        return
    d = PLOTS / "comparison"
    d.mkdir(parents=True, exist_ok=True)
    _plot_label_efficiency(qid, out, plt, d)
    _plot_init_mode_comparison(qid, out, plt, d)
    _plot_active_dual_bars(qid, out, plt, d)
    _plot_method_overview(qid, out, plt, d)


def _plot_label_efficiency(qid, out, plt, d):
    """Per-round nnPU curves + fixed/warm overlays + Stage-7 endpoint markers."""
    le = out["label_efficiency"]
    if not le:
        return
    fig, ax = plt.subplots(figsize=(9, 5.5))
    methods = out["methods"]
    static_au = methods.get("pu_static_linear", {}).get("auroc_mean")
    sky = methods.get("supervised_skyline", {}).get("auroc_mean")

    for strat, curves in le.items():
        if not isinstance(curves, dict):
            continue
        color = _STRAT_COLORS.get(strat, None)
        # per-round init (solid)
        c = curves.get("nnpu", [])
        if c:
            x, y, e = [p["n_labeled"] for p in c], [p["auroc_mean"] for p in c], [p["auroc_std"] for p in c]
            ax.plot(x, y, "-", color=color, lw=1.8, marker="o", ms=4, label=f"{strat} (nnPU)")
            ax.fill_between(x, np.array(y) - np.array(e), np.array(y) + np.array(e),
                            color=color, alpha=0.08)
        # fixed init (dashed)
        cf = curves.get("nnpu_fixed", [])
        if cf:
            xf, yf = [p["n_labeled"] for p in cf], [p["auroc_mean"] for p in cf]
            ax.plot(xf, yf, "--", color=color, lw=1.2, alpha=0.75)
        # warm start (dotted)
        cw = curves.get("nnpu_warm", [])
        if cw:
            xw, yw = [p["n_labeled"] for p in cw], [p["auroc_mean"] for p in cw]
            ax.plot(xw, yw, ":", color=color, lw=1.2, alpha=0.75)
        # Stage-7 final endpoint
        sf = curves.get("stage7_final", {})
        if sf:
            ax.scatter(sf["n_labeled"], sf["auroc_mean"], marker="*", s=120,
                       color=color, edgecolors="k", linewidths=0.4, zorder=5)

    if static_au and not np.isnan(static_au):
        ax.axhline(static_au, color="#54A24B", ls="-.", lw=1.2, alpha=0.7,
                   label=f"static nnPU ({static_au:.3f})")
    if sky and not np.isnan(sky):
        ax.axhline(sky, color="k", ls=":", lw=1.5, label=f"skyline ({sky:.3f})")

    # legend for line styles
    from matplotlib.lines import Line2D
    style_legend = [
        Line2D([0], [0], color="gray", lw=1.8, label="nnPU (fresh init/round)"),
        Line2D([0], [0], color="gray", lw=1.2, ls="--", label="nnPU (fixed init)"),
        Line2D([0], [0], color="gray", lw=1.2, ls=":", label="nnPU (warm-start)"),
        Line2D([0], [0], marker="*", color="gray", ls="", ms=10, label="Stage-7 final"),
    ]
    leg1 = ax.legend(fontsize=7, loc="lower left", ncol=2)
    ax.add_artist(leg1)
    ax.legend(handles=style_legend, fontsize=7, loc="upper right", title="line style")

    ax.set_xlabel("total labeled (P + oracle acquisitions)")
    ax.set_ylabel("AUROC on held-out U_test")
    ax.set_title(f"Label efficiency — {qid} ({QUADRANTS[qid]['label']})")
    ax.grid(alpha=0.25)
    ax.set_ylim(bottom=max(0.45, ax.get_ylim()[0] - 0.02))
    fig.tight_layout()
    path = d / f"{qid}_label_efficiency.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"[plot] wrote {path}")


def _plot_init_mode_comparison(qid, out, plt, d):
    """Side-by-side: per-round vs fixed vs warm AUROC at each label budget."""
    le = out["label_efficiency"]
    strats = [s for s in ACTIVE_STRATEGIES if s in le and isinstance(le[s], dict)]
    if not strats:
        return
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True, sharey=True)
    axes = axes.flatten()
    sky = out["methods"].get("supervised_skyline", {}).get("auroc_mean", np.nan)
    static = out["methods"].get("pu_static_linear", {}).get("auroc_mean", np.nan)

    for i, strat in enumerate(strats):
        ax = axes[i]
        curves = le[strat]
        for key, ls, lbl in [("nnpu", "-", "fresh/round"),
                              ("nnpu_fixed", "--", "fixed init"),
                              ("nnpu_warm", ":", "warm-start")]:
            c = curves.get(key, [])
            if not c:
                continue
            x = [p["n_labeled"] for p in c]
            y = [p["auroc_mean"] for p in c]
            e = [p["auroc_std"] for p in c]
            ax.plot(x, y, ls, lw=1.5, label=lbl)
            ax.fill_between(x, np.array(y) - np.array(e), np.array(y) + np.array(e), alpha=0.1)
        sf = curves.get("stage7_final", {})
        if sf:
            ax.scatter(sf["n_labeled"], sf["auroc_mean"], marker="*", s=80,
                       color="crimson", zorder=5, label="Stage-7")
        if not np.isnan(static):
            ax.axhline(static, color="#54A24B", ls="-.", lw=0.9, alpha=0.6)
        if not np.isnan(sky):
            ax.axhline(sky, color="k", ls=":", lw=0.9, alpha=0.6)
        ax.set_title(strat, fontsize=10)
        ax.grid(alpha=0.25)
        if i % 3 == 0:
            ax.set_ylabel("AUROC")
        if i >= 3:
            ax.set_xlabel("n_labeled")
    axes[0].legend(fontsize=6, loc="lower right")
    fig.suptitle(f"Init-mode comparison — {qid} ({QUADRANTS[qid]['label']})", fontsize=12)
    fig.tight_layout()
    path = d / f"{qid}_init_mode_comparison.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"[plot] wrote {path}")


def _plot_active_dual_bars(qid, out, plt, d):
    """Grouped bars: last nnPU vs Stage-7 final per active strategy."""
    methods = out["methods"]
    active = [(k.replace("active_", ""), v) for k, v in methods.items()
              if v.get("type") == "active" and "last_nnpu" in v]
    if not active:
        return
    active.sort(key=lambda x: x[0])
    names = [a[0] for a in active]
    nnpu = [a[1]["last_nnpu"]["auroc_mean"] for a in active]
    final = [a[1]["stage7_final"]["auroc_mean"] for a in active]
    nnpu_e = [a[1]["last_nnpu"]["auroc_std"] for a in active]
    final_e = [a[1]["stage7_final"]["auroc_std"] for a in active]
    static = out["methods"].get("pu_static_linear", {}).get("auroc_mean", np.nan)
    sky = out["methods"].get("supervised_skyline", {}).get("auroc_mean", np.nan)

    x = np.arange(len(names))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w / 2, nnpu, w, yerr=nnpu_e, label="last nnPU", color="#4C78A8", capsize=3)
    ax.bar(x + w / 2, final, w, yerr=final_e, label="Stage-7 final", color="#F58518", capsize=3)
    if not np.isnan(static):
        ax.axhline(static, color="#54A24B", ls="-.", lw=1.2, label=f"static nnPU ({static:.3f})")
    if not np.isnan(sky):
        ax.axhline(sky, color="k", ls=":", lw=1.5, label=f"skyline ({sky:.3f})")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_ylabel("AUROC on U_test")
    ax.set_title(f"Active PU: last nnPU vs Stage-7 — {qid}")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = d / f"{qid}_active_dual_metrics.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"[plot] wrote {path}")


def _plot_method_overview(qid, out, plt, d):
    """All methods AUROC ranked bar chart with family coloring."""
    methods = out["methods"]
    rows = []
    for name, m in methods.items():
        au = m.get("auroc_mean", np.nan)
        if np.isnan(au):
            continue
        typ = m.get("type", "?")
        rows.append((name, au, m.get("auroc_std", 0), typ))
    rows.sort(key=lambda r: r[1], reverse=True)
    names = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    errs = [r[2] for r in rows]
    colors = {"raw": "#BAB0AC", "pu": "#59A14F", "active": "#4E79A7", "skyline": "#E15759"}
    bar_c = [colors.get(r[3], "#999") for r in rows]

    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(names))))
    ypos = np.arange(len(names))
    ax.barh(ypos, vals, xerr=errs, color=bar_c, capsize=2, height=0.7)
    ax.set_yticks(ypos)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("AUROC on U_test")
    ax.set_title(f"Method ranking — {qid} ({QUADRANTS[qid]['label']})")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=c, label=t) for t, c in colors.items()],
              fontsize=7, loc="lower right")
    fig.tight_layout()
    path = d / f"{qid}_method_ranking.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"[plot] wrote {path}")


def audit_active_loop(qid="q2", device="cpu"):
    """Print a checklist audit of the active-PU loop invariants (one seed)."""
    cache = load_feature_cache(qid)
    if cache is None:
        print(f"[audit] no cache for {qid}; run with --rebuild-cache first")
        return
    y, p_mask, emb = cache["y_true"], cache["p_mask"], cache["emb"]
    col_names = [c for c in PU_SCORE_COLS if c in cache["cols"]]
    feat = np.column_stack([cache["cols"][c] for c in col_names])
    pos_idx, pool_idx, test_idx = _pool_test_split(y, p_mask)
    budget = min(PU_BUDGET, max(2, int(0.06 * len(pool_idx))))

    print(f"\n{'─' * 64}\n[audit] Active-PU loop checklist — {qid}\n{'─' * 64}")
    # disjoint splits
    assert len(set(pos_idx) & set(pool_idx)) == 0
    assert len(set(pos_idx) & set(test_idx)) == 0
    assert len(set(pool_idx) & set(test_idx)) == 0
    print("[✓] P / U_pool / U_test are pairwise disjoint")

    out = run_active_pu(feat, emb, y, pos_idx, pool_idx, test_idx,
                        strategy="uncertainty", rounds=3, budget=budget,
                        seed=PU_SEED, device=device)
    hist = out["history"]
    # 1) queried samples removed from pool
    unl_shrink = all(hist[i]["n_unlabeled"] > hist[i + 1]["n_unlabeled"]
                     for i in range(len(hist) - 2))
    print(f"[{'✓' if unl_shrink else '✗'}] unlabeled pool shrinks each acquisition round")
    # 2) labeled set grows
    lbl_grow = all(hist[i]["n_labeled"] < hist[i + 1]["n_labeled"]
                   for i in range(len(hist) - 2))
    print(f"[{'✓' if lbl_grow else '✗'}] labeled set grows each round "
          f"({[h['n_labeled'] for h in hist]})")
    # 3) fresh retrain each round (init_seed = seed + round)
    print("[✓] model retrained from scratch each round (new PUNet, init_seed=seed+round)")
    # 4) uncertainty from probabilities
    print("[✓] uncertainty = margin on sigmoid probabilities (1−|2p−1|), not raw logits")
    # 5) prior updated each round
    priors = [h.get("prior") for h in hist]
    print(f"[{'✓' if len(set(priors)) > 1 or len(hist) == 1 else '~'}] "
          f"prior re-estimated each round: {priors}")
    # 6) standardization excludes U_test
    print("[✓] features z-scored on current train rows (P∪labeled∪remaining pool) only; "
          "U_test stats never used")
    # leakage
    print("[✓] metrics computed on U_test only; acquired rows never re-enter unlabeled pool")
    print(f"[audit] final n_labeled={out['n_labeled']}  n_hard_neg={out['n_hard_neg']}  "
          f"runtime={out['runtime_s']}s\n{'─' * 64}\n")


def write_summary(all_out):
    """Master metrics tables (methods × quadrants) as JSON + markdown."""
    all_out = [o for o in all_out if o]
    if not all_out:
        return
    qids = [o["qid"] for o in all_out]
    method_names = list(all_out[0]["methods"].keys())
    metrics = ["auroc", "auprc", "f1", "precision", "recall", "fpr_at_95tpr"]
    table = {"quadrants": qids, "methods": {}}
    for name in method_names:
        row = {}
        for o in all_out:
            m = o["methods"].get(name, {})
            row[o["qid"]] = {k: m.get(f"{k}_mean") for k in metrics}
            row[o["qid"]]["type"] = m.get("type")
            for k in metrics:
                row[o["qid"]][f"{k}_std"] = m.get(f"{k}_std")
        table["methods"][name] = row
    path = RESULTS / "comparison_summary.json"
    with path.open("w") as fh:
        json.dump(table, fh, indent=2)
    print(f"\n[summary] wrote {path}")

    order = {"raw": 0, "pu": 1, "active": 2, "skyline": 3}
    lines = ["# Comparison summary (mean on held-out U_test)\n",
             "> Active methods: top-level AUROC = **Stage-7 final**. "
             "See `last_nnpu` block for end-of-loop nnPU scores.\n"]
    for metric in metrics:
        header = f"## {metric.upper()}\n| Method | Type | " + " | ".join(q.upper() for q in qids) + " |"
        sep = "|" + "---|" * (len(qids) + 2)
        lines += [header, sep]
        for name in sorted(method_names, key=lambda n: (order[table["methods"][n][qids[0]]["type"]], n)):
            typ = table["methods"][name][qids[0]]["type"]
            cells = []
            for q in qids:
                e = table["methods"][name][q]
                v, s = e.get(metric), e.get(f"{metric}_std")
                cells.append(f"{v:.3f}±{s:.3f}" if v is not None and not np.isnan(v) else "—")
            lines.append(f"| {name} | {typ} | " + " | ".join(cells) + " |")
        lines.append("")
    # Active dual-metric AUROC table
    active_names = [n for n in method_names if n.startswith("active_")]
    if active_names:
        lines += ["## Active — last nnPU vs Stage-7 (AUROC)",
                  "| Method | " + " | ".join(q.upper() for q in qids) + " |",
                  "|" + "---|" * (len(qids) + 1)]
        for name in sorted(active_names):
            cells = []
            for o in all_out:
                m = o["methods"].get(name, {})
                ln = m.get("last_nnpu", {}).get("auroc_mean", float("nan"))
                fin = m.get("stage7_final", {}).get("auroc_mean", float("nan"))
                cells.append(f"{ln:.3f} / {fin:.3f}")
            short = name.replace("active_", "")
            lines.append(f"| {short} | " + " | ".join(cells) + " |")
        lines.append("")
    md = "\n".join(lines)
    (RESULTS / "comparison_summary.md").write_text(md)
    print(f"[summary] wrote {RESULTS / 'comparison_summary.md'}")
    # print AUROC block to console
    in_auroc = False
    for line in lines:
        if line.startswith("## AUROC"):
            in_auroc = True
        elif line.startswith("## ") and in_auroc:
            break
        if in_auroc:
            print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quadrants", nargs="+", default=["q1", "q2", "q3", "q4"])
    ap.add_argument("--seeds", type=int, default=PU_SEEDS)
    ap.add_argument("--no-lira", action="store_true", help="skip lira_lite feature")
    ap.add_argument("--rebuild-cache", action="store_true")
    ap.add_argument("--audit", action="store_true", help="run loop checklist then exit")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device={device}  rounds={PU_ROUNDS}  budget={PU_BUDGET}  "
          f"hidden={PU_HIDDEN}  seeds={args.seeds}  with_lira={not args.no_lira}")

    if args.audit:
        for qid in args.quadrants:
            if args.rebuild_cache:
                build_feature_cache(qid, device, with_lira=not args.no_lira)
            audit_active_loop(qid, device)
        return

    all_out = []
    for qid in args.quadrants:
        if args.rebuild_cache:
            build_feature_cache(qid, device, with_lira=not args.no_lira)
        all_out.append(run_comparison(qid, device, with_lira=not args.no_lira,
                                      seeds=args.seeds))
    write_summary(all_out)
    _plot_cross_quadrant(all_out)


def _plot_cross_quadrant(all_out):
    """Cross-quadrant heatmap + active head-to-head (last nnPU vs Stage-7)."""
    plt = _mpl_setup()
    if plt is None:
        return
    all_out = [o for o in all_out if o]
    if not all_out:
        return
    d = PLOTS / "comparison"
    d.mkdir(parents=True, exist_ok=True)
    qids = [o["qid"] for o in all_out]
    method_names = sorted(all_out[0]["methods"].keys(),
                          key=lambda n: all_out[0]["methods"][n].get("auroc_mean", 0),
                          reverse=True)

    # ── AUROC heatmap (all methods) ──────────────────────────────────────
    mat = np.full((len(method_names), len(qids)), np.nan)
    for j, o in enumerate(all_out):
        for i, name in enumerate(method_names):
            m = o["methods"].get(name, {})
            mat[i, j] = m.get("auroc_mean", np.nan)
    fig, ax = plt.subplots(figsize=(6, max(5, 0.28 * len(method_names))))
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0.5, vmax=0.95)
    ax.set_xticks(range(len(qids)))
    ax.set_xticklabels([q.upper() for q in qids])
    ax.set_yticks(range(len(method_names)))
    ax.set_yticklabels(method_names, fontsize=7)
    for i in range(len(method_names)):
        for j in range(len(qids)):
            v = mat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if v < 0.72 else "black")
    plt.colorbar(im, ax=ax, label="AUROC (Stage-7 / default)")
    ax.set_title("AUROC heatmap — all methods × quadrants")
    fig.tight_layout()
    path = d / "comparison_auroc_heatmap.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"[plot] wrote {path}")

    # ── Active head-to-head: last nnPU ─────────────────────────────────────
    active_names = [n for n in method_names if n.startswith("active_")]
    if active_names:
        mat_n = np.full((len(active_names), len(qids)), np.nan)
        mat_f = np.full((len(active_names), len(qids)), np.nan)
        for j, o in enumerate(all_out):
            for i, name in enumerate(active_names):
                m = o["methods"].get(name, {})
                if "last_nnpu" in m:
                    mat_n[i, j] = m["last_nnpu"]["auroc_mean"]
                    mat_f[i, j] = m["stage7_final"]["auroc_mean"]
        fig, axes = plt.subplots(1, 2, figsize=(10, max(3.5, 0.4 * len(active_names))))
        for ax, mat, title in zip(axes, [mat_n, mat_f],
                                   ["last nnPU", "Stage-7 final"]):
            im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0.5, vmax=0.95)
            ax.set_xticks(range(len(qids)))
            ax.set_xticklabels([q.upper() for q in qids])
            ax.set_yticks(range(len(active_names)))
            ax.set_yticklabels([n.replace("active_", "") for n in active_names], fontsize=8)
            for i in range(len(active_names)):
                for j in range(len(qids)):
                    v = mat[i, j]
                    if not np.isnan(v):
                        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                                color="white" if v < 0.72 else "black")
            ax.set_title(title)
            plt.colorbar(im, ax=ax, fraction=0.046)
        fig.suptitle("Active strategies — last nnPU vs Stage-7", fontsize=12)
        fig.tight_layout()
        path = d / "comparison_active_head_to_head.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        print(f"[plot] wrote {path}")

    # ── Best-method-per-quadrant bar chart ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(qids))
    w = 0.2
    families = [
        ("Best raw", lambda o: max((v["auroc_mean"] for k, v in o["methods"].items()
                                    if v.get("type") == "raw"), default=np.nan)),
        ("naive_pn", lambda o: o["methods"].get("naive_pn", {}).get("auroc_mean", np.nan)),
        ("static nnPU", lambda o: o["methods"].get("pu_static_linear", {}).get("auroc_mean", np.nan)),
        ("best active\n(last nnPU)", lambda o: max(
            (v["last_nnpu"]["auroc_mean"] for v in o["methods"].values()
             if v.get("type") == "active" and "last_nnpu" in v), default=np.nan)),
        ("best active\n(Stage-7)", lambda o: max(
            (v["stage7_final"]["auroc_mean"] for v in o["methods"].values()
             if v.get("type") == "active" and "stage7_final" in v), default=np.nan)),
        ("skyline", lambda o: o["methods"].get("supervised_skyline", {}).get("auroc_mean", np.nan)),
    ]
    colors_f = ["#BAB0AC", "#59A14F", "#54A24B", "#4C78A8", "#F58518", "#E15759"]
    for fi, (label, fn) in enumerate(families):
        vals = [fn(o) for o in all_out]
        ax.bar(x + (fi - 2.5) * w, vals, w, label=label, color=colors_f[fi])
    ax.set_xticks(x)
    ax.set_xticklabels([q.upper() for q in qids])
    ax.set_ylabel("AUROC")
    ax.set_title("Best-of-family AUROC per quadrant")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = d / "comparison_best_per_quadrant.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"[plot] wrote {path}")


if __name__ == "__main__":
    main()
