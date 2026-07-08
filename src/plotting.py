from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve
from scipy.stats import gaussian_kde
from scipy.spatial import ConvexHull, QhullError
from umap import UMAP

from config import PLOTS, QUADRANTS, COLORS, UMAP_SEED


plt.style.use("seaborn-v0_8-whitegrid")


METHOD_PRETTY = {
    "ppl":      "PPL Ratio",
    "emb":      "Embedding Sim",
    "combined": "Combined",
}


def _drop_nan(labels, scores):
    s = np.asarray(scores, dtype=np.float64)
    valid = ~np.isnan(s)
    return np.asarray(labels)[valid].astype(int), s[valid]


def _quadrant_dir(qid: str) -> Path:
    d = PLOTS / f"{qid}_{QUADRANTS[qid]['name']}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def plot_score_distributions(qid, df, scores_dict, metrics):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    labels_arr = df["quadrant_label"].values.astype(int)

    for ax, method in zip(axes, ["ppl", "emb", "combined"]):
        labels_v, scores_v = _drop_nan(labels_arr, scores_dict[method])
        m = metrics[method]
        st = scores_v[labels_v == 1]
        sf = scores_v[labels_v == 0]

        if len(st) > 1 and len(sf) > 1 and np.std(st) > 1e-12 and np.std(sf) > 1e-12:
            xs = np.linspace(min(st.min(), sf.min()) - 0.1,
                             max(st.max(), sf.max()) + 0.1, 400)
            kde_t = gaussian_kde(st)
            kde_f = gaussian_kde(sf)
            yt = kde_t(xs)
            yf = kde_f(xs)
            ax.plot(xs, yt, color=COLORS["target"], lw=1.5, label="target (1)")
            ax.fill_between(xs, yt, alpha=0.3, color=COLORS["target"])
            ax.plot(xs, yf, color=COLORS["forget"], lw=1.5, label="forget (0)")
            ax.fill_between(xs, yf, alpha=0.3, color=COLORS["forget"])

        rug_y_t = np.full_like(st, -0.02 * (ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1.0))
        rug_y_f = np.full_like(sf, -0.05 * (ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1.0))
        ax.plot(st, rug_y_t, "|", color=COLORS["target"], alpha=0.4, markersize=8)
        ax.plot(sf, rug_y_f, "|", color=COLORS["forget"], alpha=0.4, markersize=8)

        y_t = m.get("youden_threshold", float("nan"))
        if not (y_t is None or np.isnan(y_t)):
            ax.axvline(y_t, color="black", linestyle="--", lw=1.0, alpha=0.7,
                       label=f"Youden t={y_t:.3f}")

        auroc   = m.get("auroc", float("nan"))
        overlap = m.get("overlap", float("nan"))
        ax.set_title(f"{METHOD_PRETTY[method]} | AUROC={auroc:.3f} | Overlap={overlap:.3f}",
                     fontsize=11)
        ax.set_xlabel("score")
        ax.set_ylabel("density")
        ax.legend(loc="best", fontsize=8)

        if not np.isnan(overlap) and overlap > 0.85:
            ax.text(0.5, 0.92, "Low signal — distributions nearly identical",
                    transform=ax.transAxes, ha="center",
                    color="red", fontsize=9,
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="red"))

    fig.suptitle(f"{QUADRANTS[qid]['label']} — Score distributions", fontsize=13)
    fig.tight_layout()
    out = _quadrant_dir(qid) / "plot1_score_distributions.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curves(qid, df, scores_dict, metrics):
    fig, ax = plt.subplots(figsize=(7, 6))
    labels_arr = df["quadrant_label"].values.astype(int)

    aurocs = {m: metrics[m].get("auroc", float("nan")) for m in ["ppl", "emb", "combined"]}
    valid_aurocs = {k: v for k, v in aurocs.items() if not np.isnan(v)}
    best_method = max(valid_aurocs, key=valid_aurocs.get) if valid_aurocs else None

    method_colors = {"ppl": COLORS["ppl"], "emb": COLORS["emb"], "combined": COLORS["combined"]}

    for method in ["ppl", "emb", "combined"]:
        labels_v, scores_v = _drop_nan(labels_arr, scores_dict[method])
        if len(labels_v) == 0 or len(np.unique(labels_v)) < 2:
            continue
        fpr, tpr, _ = roc_curve(labels_v, scores_v)
        auc = aurocs[method]
        label = f"{METHOD_PRETTY[method]} (AUC={auc:.3f})"
        if auc < 0.55:
            label += "  ≈ random"
        ax.plot(fpr, tpr, color=method_colors[method], lw=1.8, label=label)
        if method == best_method:
            ax.fill_between(fpr, tpr, alpha=0.08, color=method_colors[method])

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", lw=1.0, label="Random (AUC=0.50)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{QUADRANTS[qid]['label']} — ROC")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    out = _quadrant_dir(qid) / "plot2_roc_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pr_curves(qid, df, scores_dict, metrics):
    fig, ax = plt.subplots(figsize=(7, 6))
    labels_arr = df["quadrant_label"].values.astype(int)
    baseline = float(labels_arr.mean())

    method_colors = {"ppl": COLORS["ppl"], "emb": COLORS["emb"], "combined": COLORS["combined"]}

    for method in ["ppl", "emb", "combined"]:
        labels_v, scores_v = _drop_nan(labels_arr, scores_dict[method])
        if len(labels_v) == 0 or len(np.unique(labels_v)) < 2:
            continue
        precision, recall, _ = precision_recall_curve(labels_v, scores_v)
        ap = metrics[method].get("ap", float("nan"))
        ax.plot(recall, precision, color=method_colors[method], lw=1.8,
                label=f"{METHOD_PRETTY[method]} (AP={ap:.3f})")

    ax.axhline(baseline, linestyle="--", color="gray", lw=1.0,
               label=f"Baseline (pos rate={baseline:.3f})")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"{QUADRANTS[qid]['label']} — Precision-Recall")
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    out = _quadrant_dir(qid) / "plot3_pr_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_2d_scatter(qid, df, scores_dict, metrics):
    score_ppl = np.asarray(scores_dict["ppl"], dtype=np.float64)
    score_emb = np.asarray(scores_dict["emb"], dtype=np.float64)
    labels    = df["quadrant_label"].values.astype(int)
    valid = (~np.isnan(score_ppl)) & (~np.isnan(score_emb))

    fig = plt.figure(figsize=(7, 6))
    gs = fig.add_gridspec(2, 2, width_ratios=(5, 1), height_ratios=(1, 5),
                          left=0.1, right=0.95, bottom=0.1, top=0.92,
                          wspace=0.05, hspace=0.05)
    ax_main = fig.add_subplot(gs[1, 0])
    ax_top  = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

    for cls, color, name in [(1, COLORS["target"], "target"), (0, COLORS["forget"], "forget")]:
        mask = valid & (labels == cls)
        ax_main.scatter(score_ppl[mask], score_emb[mask],
                        c=color, alpha=0.5, s=15, label=name, edgecolors="none")

    for cls, color in [(1, COLORS["target"]), (0, COLORS["forget"])]:
        mask = valid & (labels == cls)
        if mask.sum() >= 2 and np.std(score_ppl[mask]) > 1e-12:
            kde = gaussian_kde(score_ppl[mask])
            xs = np.linspace(score_ppl[valid].min(), score_ppl[valid].max(), 200)
            ax_top.plot(xs, kde(xs), color=color, lw=1.2)
            ax_top.fill_between(xs, kde(xs), alpha=0.25, color=color)
        if mask.sum() >= 2 and np.std(score_emb[mask]) > 1e-12:
            kde = gaussian_kde(score_emb[mask])
            ys = np.linspace(score_emb[valid].min(), score_emb[valid].max(), 200)
            ax_right.plot(kde(ys), ys, color=color, lw=1.2)
            ax_right.fill_betweenx(ys, kde(ys), alpha=0.25, color=color)

    ax_top.set_xticks([]); ax_top.set_yticks([])
    ax_right.set_xticks([]); ax_right.set_yticks([])
    ax_top.set_ylabel("")
    ax_right.set_xlabel("")

    ax_main.set_xlabel("score_ppl (raw)")
    ax_main.set_ylabel("score_emb (raw)")
    ax_main.legend(loc="best", fontsize=9)
    fig.suptitle(f"{QUADRANTS[qid]['label']} — PPL vs Embedding score", fontsize=12)
    out = _quadrant_dir(qid) / "plot4_2d_scatter.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_umap(qid, df, embeddings, metrics):
    labels = df["quadrant_label"].values.astype(int)
    assert embeddings.shape[0] == len(df), (
        f"UMAP input rows {embeddings.shape[0]} != df len {len(df)}"
    )

    reducer = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=UMAP_SEED)
    proj = reducer.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(7, 6))
    for cls, color, name in [(1, COLORS["target"], "target"), (0, COLORS["forget"], "forget")]:
        mask = labels == cls
        ax.scatter(proj[mask, 0], proj[mask, 1],
                   c=color, alpha=0.55, s=18, label=name, edgecolors="none")
        if mask.sum() >= 3:
            pts = proj[mask]
            try:
                hull = ConvexHull(pts)
                for simplex in hull.simplices:
                    ax.plot(pts[simplex, 0], pts[simplex, 1], color=color, lw=1.0, alpha=0.7)
            except QhullError:
                pass

    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_title(f"{QUADRANTS[qid]['label']} — UMAP of embeddings")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    out = _quadrant_dir(qid) / "plot5_umap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_threshold_sweep(qid, df, scores_dict, metrics):
    labels_arr = df["quadrant_label"].values.astype(int)
    labels_v, scores_v = _drop_nan(labels_arr, scores_dict["combined"])
    if len(scores_v) == 0:
        return

    s_norm = (scores_v - scores_v.min()) / (scores_v.max() - scores_v.min() + 1e-9)
    thresholds = np.arange(0.0, 1.01, 0.01)
    precs, recs, f1s = [], [], []
    for t in thresholds:
        preds = (s_norm >= t).astype(int)
        tp = int(((preds == 1) & (labels_v == 1)).sum())
        fp = int(((preds == 1) & (labels_v == 0)).sum())
        fn = int(((preds == 0) & (labels_v == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        precs.append(precision); recs.append(recall); f1s.append(f1)
    precs = np.array(precs); recs = np.array(recs); f1s = np.array(f1s)

    peak_t  = float(metrics["combined"].get("f1_threshold", float("nan")))
    peak_f1 = float(metrics["combined"].get("f1", float("nan")))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, precs, color="#3A8DD2", lw=1.6, label="precision")
    ax.plot(thresholds, recs,  color="#D85A30", lw=1.6, label="recall")
    ax.plot(thresholds, f1s,   color="#1D9E75", lw=1.8, label="F1")
    if not np.isnan(peak_t):
        ax.axvline(peak_t, color="black", linestyle="--", lw=1.0,
                   label=f"peak F1={peak_f1:.3f} @ t={peak_t:.2f}")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xlabel("threshold (min-max normalized combined score)")
    ax.set_ylabel("score")
    ax.set_title(f"{QUADRANTS[qid]['label']} — Threshold sweep (combined)")
    ax.legend(loc="best", fontsize=9)

    if (f1s.max() - f1s.min()) < 0.05:
        ax.text(0.5, 0.5, "Flat F1 — threshold insensitive",
                transform=ax.transAxes, ha="center", va="center",
                color="red", fontsize=11,
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="red"))

    fig.tight_layout()
    out = _quadrant_dir(qid) / "plot6_threshold_sweep.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_all(qid, df, scores_dict, embeddings, metrics):
    plot_score_distributions(qid, df, scores_dict, metrics)
    plot_roc_curves(qid, df, scores_dict, metrics)
    plot_pr_curves(qid, df, scores_dict, metrics)
    plot_2d_scatter(qid, df, scores_dict, metrics)
    plot_umap(qid, df, embeddings, metrics)
    plot_threshold_sweep(qid, df, scores_dict, metrics)
    print(f"[plotting] {qid}: wrote plots to {_quadrant_dir(qid)}")


def plot_layerwise_drift(qid, drift):
    """plot7 — per-layer cosine distance between base GPT-2 and LoRA-adapted
    GPT-2 hidden states, aggregated per class. Two lines: target and forget.
    The gap (target - forget) at each layer says where LoRA differentiated
    target from forget at the representation level."""
    if not drift:
        return
    tgt = np.asarray(drift["target_drift_per_layer"], dtype=np.float64)
    frg = np.asarray(drift["forget_drift_per_layer"], dtype=np.float64)
    tgt_std = np.asarray(drift.get("target_std_per_layer", [0.0] * len(tgt)))
    frg_std = np.asarray(drift.get("forget_std_per_layer", [0.0] * len(frg)))
    delta = np.asarray(drift.get("delta_per_layer", tgt - frg))
    x = np.arange(len(tgt))

    fig, (ax, ax_d) = plt.subplots(
        2, 1, figsize=(8, 6), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    ax.plot(x, tgt, "-o", color=COLORS["target"], lw=1.8, label="target rows")
    ax.fill_between(x, tgt - tgt_std, tgt + tgt_std,
                    color=COLORS["target"], alpha=0.15)
    ax.plot(x, frg, "-o", color=COLORS["forget"], lw=1.8, label="forget rows")
    ax.fill_between(x, frg - frg_std, frg + frg_std,
                    color=COLORS["forget"], alpha=0.15)

    best_layer = drift.get("best_diff_layer", int(np.nanargmax(delta)))
    ax.scatter([best_layer], [tgt[best_layer]], s=120, facecolor="none",
               edgecolor=COLORS["target"], lw=2.0, zorder=5)
    ax.scatter([best_layer], [frg[best_layer]], s=120, facecolor="none",
               edgecolor=COLORS["forget"], lw=2.0, zorder=5)

    ax.set_ylabel("cosine drift  (1 − cos(h_base, h_lora))")
    ax.set_title(
        f"{QUADRANTS[qid]['label']} — Activation drift by layer  "
        f"(metric = cosine distance, pool = {drift.get('pool','mean')})",
        fontsize=11,
    )
    ax.legend(loc="best", fontsize=9)

    # Bottom panel: Δ(target − forget) — where LoRA distinguished the classes
    bar_colors = ["#1D9E75" if v >= 0 else "#D85A30" for v in delta]
    ax_d.bar(x, delta, color=bar_colors, alpha=0.7)
    ax_d.axhline(0.0, color="black", lw=0.8)
    ax_d.set_xticks(x)
    ax_d.set_xlim(-0.5, len(tgt) - 0.5)
    ax_d.set_xlabel("Transformer layer index (0 = embedding output)")
    ax_d.set_ylabel("Δ(t − f)")

    fig.tight_layout()
    out = _quadrant_dir(qid) / "plot7_layerwise_drift.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_summary(all_results):
    qids = ["q1", "q2", "q3", "q4"]
    methods = ["ppl", "emb", "combined"]
    method_colors = {"ppl": COLORS["ppl"], "emb": COLORS["emb"], "combined": COLORS["combined"]}

    fig, axes = plt.subplots(len(qids), len(methods), figsize=(14, 16))
    for i, qid in enumerate(qids):
        df = all_results[qid]["df"]
        scores_dict = all_results[qid]["scores"]
        metrics     = all_results[qid]["metrics"]
        labels_arr  = df["quadrant_label"].values.astype(int)

        for j, method in enumerate(methods):
            ax = axes[i, j]
            labels_v, scores_v = _drop_nan(labels_arr, scores_dict[method])
            if len(labels_v) > 0 and len(np.unique(labels_v)) == 2:
                fpr, tpr, _ = roc_curve(labels_v, scores_v)
                ax.plot(fpr, tpr, color=method_colors[method], lw=1.6)
            ax.plot([0, 1], [0, 1], linestyle="--", color="gray", lw=0.8)
            auroc = metrics[method].get("auroc", float("nan"))
            ax.set_title(f"AUROC = {auroc:.3f}", fontsize=10)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            if i == 0:
                ax.set_xlabel(METHOD_PRETTY[method], fontsize=11)
                ax.xaxis.set_label_position("top")
            if j == 0:
                ax.set_ylabel(QUADRANTS[qid]["label"], fontsize=11)
            ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])

    fig.suptitle("Summary: ROC across all quadrants and methods", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = PLOTS / "summary_all_quadrants.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plotting] wrote summary plot to {out}")
