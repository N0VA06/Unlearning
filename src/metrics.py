import json

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from scipy.stats import ks_2samp, gaussian_kde

from config import RESULTS, QUADRANTS


METHOD_NAMES = ["ppl", "emb", "combined"]


def _minmax(x):
    rng = x.max() - x.min()
    return (x - x.min()) / (rng + 1e-9)


def _best_f1(labels: np.ndarray, scores: np.ndarray) -> dict:
    s_norm = _minmax(scores)
    thresholds = np.arange(0.0, 1.01, 0.01)
    best = {"f1": -1.0, "threshold": 0.0, "precision": 0.0, "recall": 0.0}
    for t in thresholds:
        preds = (s_norm >= t).astype(int)
        tp = int(((preds == 1) & (labels == 1)).sum())
        fp = int(((preds == 1) & (labels == 0)).sum())
        fn = int(((preds == 0) & (labels == 1)).sum())
        if tp + fp == 0:
            precision = 0.0
        else:
            precision = tp / (tp + fp)
        if tp + fn == 0:
            recall = 0.0
        else:
            recall = tp / (tp + fn)
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        if f1 > best["f1"]:
            best = {
                "f1":        float(f1),
                "threshold": float(t),
                "precision": float(precision),
                "recall":    float(recall),
            }
    return best


def _youden_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    fpr, tpr, thr = roc_curve(labels, scores)
    j = tpr - fpr
    idx = int(np.argmax(j))
    return float(thr[idx])


def _overlap_coefficient(scores_t: np.ndarray, scores_f: np.ndarray) -> float:
    if len(scores_t) < 2 or len(scores_f) < 2:
        return float("nan")
    if np.std(scores_t) < 1e-12 or np.std(scores_f) < 1e-12:
        return float("nan")
    try:
        kde_t = gaussian_kde(scores_t)
        kde_f = gaussian_kde(scores_f)
    except Exception:
        return float("nan")
    all_scores = np.concatenate([scores_t, scores_f])
    lo = float(all_scores.min()) - 0.1
    hi = float(all_scores.max()) + 0.1
    x  = np.linspace(lo, hi, 500)
    return float(np.trapezoid(np.minimum(kde_t(x), kde_f(x)), x))


def compute_all_metrics(labels: np.ndarray, scores_dict: dict, qid: str) -> dict:
    labels = np.asarray(labels).astype(int)
    out = {"qid": qid, "n_total": int(len(labels))}

    for method in METHOD_NAMES:
        scores = np.asarray(scores_dict[method], dtype=np.float64)
        if scores.shape[0] != labels.shape[0]:
            raise ValueError(
                f"Score array for {method} has shape {scores.shape}, "
                f"expected length {labels.shape[0]}"
            )

        valid_mask = ~np.isnan(scores)
        n_dropped  = int(len(labels) - int(valid_mask.sum()))
        scores_v   = scores[valid_mask]
        labels_v   = labels[valid_mask]
        assert len(scores_v) == len(labels_v)

        print(f"[metrics][{qid}][{method}] {n_dropped} NaN records dropped, "
              f"{len(scores_v)} used for metrics")

        if len(scores_v) == 0 or len(np.unique(labels_v)) < 2:
            print(f"[metrics][{qid}][{method}] insufficient data after NaN drop — skipping")
            out[method] = {
                "n_used":          int(len(scores_v)),
                "n_dropped":       n_dropped,
                "auroc":           float("nan"),
                "ap":              float("nan"),
                "baseline_ap":     float("nan"),
                "f1":              float("nan"),
                "f1_threshold":    float("nan"),
                "precision":       float("nan"),
                "recall":          float("nan"),
                "youden_threshold": float("nan"),
                "ks_stat":         float("nan"),
                "ks_pvalue":       float("nan"),
                "cohens_d":        float("nan"),
                "overlap":         float("nan"),
                "p_at_k":          {},
                "mean_target":     float("nan"),
                "mean_forget":     float("nan"),
            }
            continue

        auroc = float(roc_auc_score(labels_v, scores_v))
        ap    = float(average_precision_score(labels_v, scores_v))
        base_ap = float(labels_v.mean())

        f1_info  = _best_f1(labels_v, scores_v)
        youden_t = _youden_threshold(labels_v, scores_v)

        scores_t = scores_v[labels_v == 1]
        scores_f = scores_v[labels_v == 0]
        ks_stat, p_value = ks_2samp(scores_t, scores_f)

        mt = float(scores_t.mean())
        mf = float(scores_f.mean())
        st = float(scores_t.std())
        sf = float(scores_f.std())
        cohens_d = (mt - mf) / (np.sqrt((st ** 2 + sf ** 2) / 2.0) + 1e-9)

        overlap = _overlap_coefficient(scores_t, scores_f)

        sorted_labels = labels_v[np.argsort(scores_v)[::-1]]
        p_at_k = {
            str(k): float(sorted_labels[:k].mean())
            for k in [10, 25, 50, 100] if k <= len(sorted_labels)
        }

        print(f"[metrics][{qid}][{method}] AUROC={auroc:.4f}, AP={ap:.4f} (baseline={base_ap:.3f}), "
              f"F1={f1_info['f1']:.4f}@t={f1_info['threshold']:.2f}, KS={float(ks_stat):.4f}, "
              f"overlap={overlap:.4f}, cohen_d={cohens_d:.3f}")

        out[method] = {
            "n_used":          int(len(scores_v)),
            "n_dropped":       n_dropped,
            "auroc":           auroc,
            "ap":              ap,
            "baseline_ap":     base_ap,
            "f1":              f1_info["f1"],
            "f1_threshold":    f1_info["threshold"],
            "precision":       f1_info["precision"],
            "recall":          f1_info["recall"],
            "youden_threshold": youden_t,
            "ks_stat":         float(ks_stat),
            "ks_pvalue":       float(p_value),
            "cohens_d":        float(cohens_d),
            "overlap":         overlap,
            "p_at_k":          p_at_k,
            "mean_target":     mt,
            "mean_forget":     mf,
        }

    out_path = RESULTS / f"{qid}_{QUADRANTS[qid]['name']}_metrics.json"
    RESULTS.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[metrics][{qid}] wrote {out_path}")

    return out
