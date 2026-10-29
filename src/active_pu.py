"""Active non-negative Positive-Unlabeled (nnPU) learning for target–forget
separation (proposal Stages 3–7), plus the baselines it is compared against.

Given a small labeled-positive set P (reference target) and a large unlabeled
pool U (held-out target + forget) the Active-PU loop:

  * trains a non-negative PU classifier (Kiryo et al., 2017) — Stage 3;
  * actively acquires a few oracle labels per round — Stage 4;
  * mines reliable (easy) negatives from the low-probability tail — Stage 5;
  * surfaces hard negatives (high-prob rows that turn out to be forget) — Stage 6;
  * fits a final calibrated supervised classifier on the enriched labels — Stage 7.

The classifier input is a small vector of overlap SCORES (ppl_ratio, minkpp,
emb_cos, lira_lite) — never the raw 768-d embedding, which trivialises the task
via dataset identity. Embeddings are used only for the diversity/coreset term.

This module also provides the comparison baselines:
  * acquisition strategies: random / uncertainty / diversity(k-center) / badge /
    committee(QBC) / uxd (Uncertainty×Diversity);
  * classifier models: nnPU (linear or MLP head), naive PN (U treated as
    negative), a fully-supervised skyline, and the raw single-score detector.
"""

import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
)

from config import (
    PU_EPOCHS, PU_LR, PU_HIDDEN, PU_WEIGHT_DECAY, PU_ROUNDS, PU_BUDGET,
    PU_RELIABLE_FRAC, PU_NEG_LAMBDA, PU_SEED,
)

_EPS = 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# nnPU model + loss
# ─────────────────────────────────────────────────────────────────────────────
class PUNet(nn.Module):
    """Scalar-logit scorer. Linear when hidden==0, else a 1-hidden-layer MLP."""

    def __init__(self, in_dim, hidden=PU_HIDDEN):
        super().__init__()
        if hidden and hidden > 0:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, 1),
            )
        else:
            self.net = nn.Linear(in_dim, 1)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _nnpu_risk(g_p, g_u, prior, beta=0.0, gamma=1.0):
    """Kiryo non-negative PU risk with the sigmoid surrogate.

    ℓ(as +)=σ(−g), ℓ(as −)=σ(+g). Positive risk π·E_P[σ(−g)]; negative risk
    E_U[σ(g)] − π·E_P[σ(g)]. When the empirical negative risk drops below −β we
    descend on +γ·(negative risk) instead (the non-negative correction)."""
    r_p_plus = torch.sigmoid(-g_p).mean()
    r_p_minus = torch.sigmoid(g_p).mean()
    r_u_minus = torch.sigmoid(g_u).mean()
    pos_risk = prior * r_p_plus
    neg_risk = r_u_minus - prior * r_p_minus
    if neg_risk.item() >= -beta:
        return pos_risk + neg_risk
    return gamma * (-neg_risk)  # non-negative correction


def train_nnpu(X_p, X_u, prior, X_n=None, in_dim=None, hidden=PU_HIDDEN,
               epochs=PU_EPOCHS, lr=PU_LR, weight_decay=PU_WEIGHT_DECAY,
               neg_lambda=PU_NEG_LAMBDA, seed=PU_SEED, init_seed=None,
               init_model=None, device="cpu"):
    """Fit a PUNet on positives X_p and unlabeled X_u (optionally confirmed
    negatives X_n add a supervised σ(g) term). Returns the trained model.

    Retrained from scratch every call (new ``PUNet``). ``init_seed`` controls
    weight init; ``init_model`` warm-starts from a prior round's weights."""
    torch.manual_seed(init_seed if init_seed is not None else seed)
    in_dim = in_dim or X_p.shape[1]
    model = PUNet(in_dim, hidden).to(device)
    if init_model is not None:
        model.load_state_dict(init_model.state_dict())
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    tp = torch.as_tensor(X_p, dtype=torch.float32, device=device)
    tu = torch.as_tensor(X_u, dtype=torch.float32, device=device)
    tn = (torch.as_tensor(X_n, dtype=torch.float32, device=device)
          if X_n is not None and len(X_n) > 0 else None)

    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = _nnpu_risk(model(tp), model(tu), prior)
        if tn is not None:
            loss = loss + neg_lambda * torch.sigmoid(model(tn)).mean()
        loss.backward()
        opt.step()
    model.eval()
    return model


def pu_proba(model, X, device="cpu"):
    with torch.no_grad():
        t = torch.as_tensor(X, dtype=torch.float32, device=device)
        return torch.sigmoid(model(t)).cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Acquisition strategies
# ─────────────────────────────────────────────────────────────────────────────
def _uncertainty(prob):
    """Closeness to the nnPU boundary, in [0,1] (max at p=0.5)."""
    return 1.0 - np.abs(2.0 * prob - 1.0)


def _norm_rows(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + _EPS)


def _diversity(emb_pool, cand_idx, labeled_idx):
    """Min cosine distance from each candidate to the labeled set, in [0,1].
    Higher ⇒ more novel relative to what we've already labeled."""
    e = _norm_rows(emb_pool)
    lab, cand = e[labeled_idx], e[cand_idx]
    sims = cand @ lab.T if lab.shape[0] else np.zeros((len(cand), 1))
    max_sim = sims.max(axis=1) if lab.shape[0] else np.zeros(len(cand))
    return np.clip(1.0 - max_sim, 0.0, 1.0)


def _kcenter_greedy(emb_pool, cand_idx, labeled_idx, budget):
    """k-center / coreset: greedily pick the candidate farthest (cosine) from the
    growing selected+labeled set. Pure diversity, ignores the model."""
    e = _norm_rows(emb_pool)
    cand = e[cand_idx]
    if len(labeled_idx):
        dist = (1.0 - cand @ e[labeled_idx].T).min(axis=1)
    else:
        dist = np.full(len(cand_idx), 1.0)
    picks = []
    budget = min(budget, len(cand_idx))
    for _ in range(budget):
        j = int(np.argmax(dist))
        picks.append(cand_idx[j])
        dist = np.minimum(dist, 1.0 - cand @ cand[j])
    return np.array(picks, dtype=int)


def _committee_uncertainty(score_mat):
    """QBC disagreement: rank-normalise each estimator column to [0,1], then take
    the std across estimators. High ⇒ the views disagree on this row."""
    if score_mat.shape[0] < 2:
        return np.zeros(score_mat.shape[0])
    ranks = np.argsort(np.argsort(score_mat, axis=0), axis=0).astype(np.float64)
    ranks /= max(1, score_mat.shape[0] - 1)
    return ranks.std(axis=1)


def _kmeanspp_seed(G, k, rng):
    """k-means++ D² seeding: return k row indices of G spread by squared distance."""
    m = len(G)
    if k >= m:
        return np.arange(m)
    first = int(rng.integers(m))
    centers = [first]
    d2 = ((G - G[first]) ** 2).sum(axis=1)
    for _ in range(1, k):
        s = d2.sum()
        probs = (d2 / s) if s > 0 else np.full(m, 1.0 / m)
        nxt = int(rng.choice(m, p=probs))
        centers.append(nxt)
        d2 = np.minimum(d2, ((G - G[nxt]) ** 2).sum(axis=1))
    return np.array(centers, dtype=int)


def _badge_select(feat, model, cand_idx, budget, rng, device="cpu"):
    """BADGE: gradient embedding of the last layer under the pseudo-label, then
    k-means++ seeding. For a scorer g(x) with sigmoid loss the last-layer grad is
    (p − ŷ)·x; k-means++ on these picks points that are both uncertain (large ‖g‖)
    and diverse in gradient direction."""
    p = pu_proba(model, feat[cand_idx], device=device)
    yhat = (p >= 0.5).astype(np.float64)
    G = (p - yhat)[:, None] * feat[cand_idx]
    seeds = _kmeanspp_seed(G, min(budget, len(cand_idx)), rng)
    return cand_idx[seeds]


def acquire(strategy, *, prob_u, unl_idx, labeled_idx, emb, feat, score_mat,
            model, budget, rng, device="cpu"):
    """Return absolute pool indices to oracle-label this round.

    strategy ∈ {random, uncertainty, diversity, badge, committee, uxd}.
    prob_u are nnPU probabilities aligned with unl_idx."""
    if strategy == "random":
        pick = rng.choice(len(unl_idx), size=min(budget, len(unl_idx)), replace=False)
        return unl_idx[pick]
    if strategy == "diversity":
        return _kcenter_greedy(emb, unl_idx, labeled_idx, budget)
    if strategy == "badge":
        return _badge_select(feat, model, unl_idx, budget, rng, device=device)
    if strategy == "uncertainty":
        score = _uncertainty(prob_u)
    elif strategy == "committee":
        score = _committee_uncertainty(score_mat[unl_idx])
    elif strategy == "uxd":
        score = _uncertainty(prob_u) * _diversity(emb, unl_idx, labeled_idx)
    else:
        raise ValueError(f"unknown acquisition strategy: {strategy}")
    return unl_idx[np.argsort(-score)[:budget]]


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────
def _fpr_at_tpr(y, s, target_tpr=0.95):
    if len(np.unique(y)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y, s)
    ok = np.where(tpr >= target_tpr)[0]
    return float(fpr[ok[0]]) if len(ok) else 1.0


def _best_f1(y, s):
    prec, rec, thr = precision_recall_curve(y, s)
    p, r = prec[:-1], rec[:-1]
    f1 = np.where((p + r) > 0, 2 * p * r / (p + r + 1e-12), 0.0)
    if len(f1) == 0:
        return float("nan"), float("nan"), float("nan")
    j = int(np.argmax(f1))
    return float(f1[j]), float(p[j]), float(r[j])


def _ece(y, p, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        hi = p <= bins[i + 1] if i == n_bins - 1 else p < bins[i + 1]
        m = (p >= bins[i]) & hi
        if m.sum() == 0:
            continue
        ece += (m.sum() / len(p)) * abs(p[m].mean() - y[m].mean())
    return float(ece)


def eval_scores(y, s, with_ece=False):
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=np.float64)
    if len(np.unique(y)) < 2:
        out = {k: float("nan") for k in
               ("auroc", "auprc", "fpr_at_95tpr", "f1", "precision", "recall")}
        if with_ece:
            out["ece"] = float("nan")
        return out
    f1, prec, rec = _best_f1(y, s)
    out = {
        "auroc": float(roc_auc_score(y, s)),
        "auprc": float(average_precision_score(y, s)),
        "fpr_at_95tpr": _fpr_at_tpr(y, s),
        "f1": f1, "precision": prec, "recall": rec,
    }
    if with_ece:
        # clip to [0,1] for ECE if these are decision scores rather than probs
        p = s if (s.min() >= 0 and s.max() <= 1) else 1 / (1 + np.exp(-s))
        out["ece"] = _ece(y, p)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Active-PU driver (Stages 3–7) — also covers static PU via rounds=0
# ─────────────────────────────────────────────────────────────────────────────
# Leakage discipline: three disjoint index sets over the pool rows.
#   pos_idx  — labeled positives P (all target).
#   pool_idx — acquirable unlabeled U-pool (nnPU sees these as unlabeled inputs;
#              acquisition and reliable-negative mining draw only from here).
#   test_idx — held-out U-test; NEVER used for training, as unlabeled data, or
#              for acquisition. Every metric is computed on test_idx alone, so a
#              row's label can never influence its own score.
# Per-round: standardise on current train rows (P ∪ labeled ∪ remaining pool)
# only — U_test stats are never seen. π is re-estimated each round from the
# remaining hidden-positive count in the unlabeled pool.
def _estimate_prior(y_true, pool_idx, labeled_pos, unlabeled):
    """π = estimated P(y=1) among the remaining unlabeled pool rows."""
    pool_set = set(pool_idx.tolist())
    n_pos_pool = int((y_true[pool_idx] == 1).sum())
    n_pos_labeled = sum(1 for j in labeled_pos if j in pool_set)
    n_pos_left = max(0, n_pos_pool - n_pos_labeled)
    n_u = len(unlabeled)
    if n_u == 0:
        return 0.5
    return float(np.clip(n_pos_left / n_u, 0.05, 0.95))


def _standardize_train(feat, train_idx):
    """Z-score using train rows only (never U_test)."""
    mu = feat[train_idx].mean(0)
    sd = feat[train_idx].std(0) + _EPS
    return (feat - mu) / sd


def run_active_pu(
    feat, emb, y_true, pos_idx, pool_idx, test_idx, prior=None, score_mat=None,
    strategy="uxd", rounds=PU_ROUNDS, budget=PU_BUDGET,
    reliable_frac=PU_RELIABLE_FRAC, hidden=PU_HIDDEN, seed=PU_SEED, device="cpu",
    init_mode="per_round",
):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true).astype(int)
    if score_mat is None:
        score_mat = feat

    labeled_pos = set(pos_idx.tolist())
    labeled_neg = set()
    unlabeled = set(pool_idx.tolist())

    history = []
    t0 = time.time()
    n_rounds_run = 0
    model = None
    prev_unl = len(unlabeled)
    for r in range(rounds + 1):             # round 0 = initial fit, then acquisitions
        p_ids = np.array(sorted(labeled_pos))
        n_ids = np.array(sorted(labeled_neg))
        u_ids = np.array(sorted(unlabeled))
        # train rows = P + all labeled from pool + remaining unlabeled (never U_test)
        train_idx = np.concatenate([p_ids, n_ids, u_ids]).astype(int)
        fz = _standardize_train(feat, train_idx)
        cur_prior = _estimate_prior(y_true, pool_idx, labeled_pos, unlabeled)
        X_u = fz[u_ids] if len(u_ids) else fz[p_ids]

        if init_mode == "warm_start" and model is not None:
            i_seed, i_model = seed, model
        elif init_mode == "fixed":
            i_seed, i_model = seed, None
        else:  # per_round — fresh random init each round
            i_seed, i_model = seed + r, None
        model = train_nnpu(
            fz[p_ids], X_u, cur_prior,
            X_n=(fz[n_ids] if len(n_ids) else None),
            in_dim=fz.shape[1], hidden=hidden, seed=seed,
            init_seed=i_seed, init_model=i_model, device=device,
        )
        prob_test = pu_proba(model, fz[test_idx], device=device)
        m = eval_scores(y_true[test_idx], prob_test)
        m.update({
            "round": r,
            "n_labeled": int(len(labeled_pos) + len(labeled_neg)),
            "n_pos": int(len(labeled_pos)),
            "n_neg": int(len(labeled_neg)),
            "n_unlabeled": int(len(unlabeled)),
            "prior": round(cur_prior, 4),
        })
        history.append(m)

        if r == rounds or len(u_ids) == 0:
            break

        prob_u = pu_proba(model, fz[u_ids], device=device)
        picks = acquire(strategy, prob_u=prob_u, unl_idx=u_ids,
                        labeled_idx=np.array(sorted(labeled_pos | labeled_neg)),
                        emb=emb, feat=fz, score_mat=fz, model=model,
                        budget=budget, rng=rng, device=device)
        assert len(picks) == len(set(picks.tolist())), "duplicate picks in round"
        for j in picks:
            assert j in unlabeled, f"pick {j} not in unlabeled pool"
            unlabeled.discard(int(j))
            (labeled_pos if y_true[j] == 1 else labeled_neg).add(int(j))
        assert len(unlabeled) == prev_unl - len(picks), (
            f"unlabeled pool did not shrink: {prev_unl} -> {len(unlabeled)} "
            f"(picked {len(picks)})")
        prev_unl = len(unlabeled)
        n_rounds_run += 1

    # ── Stage 7: final calibrated supervised classifier ────────────────────
    p_ids = np.array(sorted(labeled_pos))
    n_ids = np.array(sorted(labeled_neg))
    u_ids = np.array(sorted(unlabeled))
    train_idx = np.concatenate([p_ids, n_ids, u_ids]).astype(int)
    fz = _standardize_train(feat, train_idx)
    reliable_extra = np.array([], dtype=int)
    if reliable_frac > 0 and len(u_ids) > 0:
        prob_pool = pu_proba(model, fz[u_ids], device=device)
        k = max(1, int(round(reliable_frac * len(u_ids))))
        reliable_extra = u_ids[np.argsort(prob_pool)[:k]]

    final = _final_classifier(fz, p_ids, n_ids, reliable_extra, seed)
    if final is not None:
        clf, iso = final
        raw = clf.decision_function(fz[test_idx])
        cal = iso.transform(raw) if iso is not None else 1 / (1 + np.exp(-raw))
    else:
        cal = pu_proba(model, fz[test_idx], device=device)
    final_metrics = eval_scores(y_true[test_idx], cal, with_ece=True)

    prob_neg = pu_proba(model, fz[n_ids], device=device) if len(n_ids) else np.array([])
    hard_neg = int((prob_neg >= 0.5).sum())
    return {
        "strategy": strategy,
        "history": history,
        "final": final_metrics,
        "n_rounds": n_rounds_run,
        "n_labeled": int(len(labeled_pos) + len(labeled_neg)),
        "n_pos_labeled": int(len(labeled_pos)),
        "n_neg_labeled": int(len(labeled_neg)),
        "n_hard_neg": hard_neg,
        "n_reliable_neg": int(len(reliable_extra)),
        "runtime_s": round(time.time() - t0, 2),
    }


def _final_classifier(feat, pos_idx, neg_idx, reliable_idx, seed):
    """Logistic classifier + isotonic calibration on labeled + reliable negatives.
    Returns (clf, iso) or None if only one class is available."""
    neg_all = np.concatenate([neg_idx, reliable_idx]).astype(int) if (
        len(neg_idx) or len(reliable_idx)) else np.array([], dtype=int)
    if len(pos_idx) == 0 or len(neg_all) == 0:
        return None
    X = np.concatenate([feat[pos_idx], feat[neg_all]], axis=0)
    y = np.concatenate([np.ones(len(pos_idx)), np.zeros(len(neg_all))]).astype(int)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(X, y)
    iso = None
    if len(np.unique(y)) == 2 and len(y) >= 10:
        try:
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(clf.decision_function(X), y)
        except Exception:
            iso = None
    return clf, iso


# ─────────────────────────────────────────────────────────────────────────────
# Non-active baselines — all evaluated on the same held-out test_idx
# ─────────────────────────────────────────────────────────────────────────────
def _fit_logistic(feat, train_idx, y_train, test_idx, y_true, seed):
    """Standardise on train rows, fit balanced logistic, score test_idx."""
    mu, sd = feat[train_idx].mean(0), feat[train_idx].std(0) + _EPS
    fz = (feat - mu) / sd
    if len(np.unique(y_train)) < 2:
        return eval_scores(y_true[test_idx], np.zeros(len(test_idx)), with_ece=True)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
    clf.fit(fz[train_idx], y_train)
    return eval_scores(y_true[test_idx], clf.decision_function(fz[test_idx]), with_ece=True)


def run_naive_pn(feat, y_true, pos_idx, pool_idx, test_idx, seed=PU_SEED):
    """Naive PN: treat the unlabeled pool as negative, fit logistic P(+) vs pool(−).
    The classic biased-PU baseline. Trained on P∪pool, evaluated on held-out test."""
    y_true = np.asarray(y_true).astype(int)
    train_idx = np.concatenate([pos_idx, pool_idx])
    y_train = np.concatenate([np.ones(len(pos_idx)), np.zeros(len(pool_idx))])
    return _fit_logistic(feat, train_idx, y_train, test_idx, y_true, seed)


def run_supervised_skyline(feat, y_true, pos_idx, pool_idx, test_idx, seed=PU_SEED):
    """True skyline: the SAME logistic classifier given every pool label. Trained on
    P(+) plus the pool with its TRUE labels, evaluated on the disjoint held-out test
    — a genuine ceiling (full supervision on the acquisition pool)."""
    y_true = np.asarray(y_true).astype(int)
    train_idx = np.concatenate([pos_idx, pool_idx])
    y_train = np.concatenate([np.ones(len(pos_idx)), y_true[pool_idx]])
    return _fit_logistic(feat, train_idx, y_train, test_idx, y_true, seed)


def run_raw_detector(score_col, y_true, test_idx):
    """Raw single-estimator detector: threshold one overlap score directly on the
    held-out test rows, no training. `score_col` is aligned with the pool rows."""
    y_true = np.asarray(y_true).astype(int)
    return eval_scores(y_true[test_idx], np.asarray(score_col)[test_idx], with_ece=True)
