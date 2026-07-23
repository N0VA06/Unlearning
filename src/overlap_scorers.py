"""Overlap estimators for target–forget separation (Active-PU pipeline, Stage 2).

Each estimator maps a text ``x`` to a single scalar "how target-like is x after
LoRA adaptation". They are computed **independently** (no fusion) so the same
Active-PU framework can be run once per estimator and compared fairly.

Estimators
----------
* ``ppl_ratio``   : mean-token log-likelihood gap, LoRA vs base.
* ``minkpp``      : Min-K%++ gap on the most-surprising tokens (calibrated).
* ``embedding``   : cosine of LoRA hidden state to a centroid built from P
                    (the labeled-positive / reference target pool).
* ``lira_lite``   : one-sided likelihood ratio — observed loss vs the OUT
                    (non-member) distribution from shadow LoRAs.
* ``lira_full``   : two-sided Gaussian likelihood ratio (IN vs OUT) from shadow
                    LoRAs, when a sample appears IN some shadow's train subset.

Leakage note: scores are computed on the unlabeled pool U (held-out target +
forget) and on P — none of which are in the LoRA-train set — so the adapter
never memorised the scored rows. See ``data_loader.make_pu_split``.
"""

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from config import BASE_MODEL, MAX_SEQ_LEN, MINK_FRAC
from src.hidden_states import (
    get_tokenizer, load_base_model, load_lora_model, extract_last_layer,
)

_EPS = 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# Shared token-level extraction (used by PPL ratio and Min-K++)
# ─────────────────────────────────────────────────────────────────────────────
def _token_logps(model, tokenizer, texts, device, need_calib=False,
                 batch_size=8, desc="tok"):
    """Per-text next-token statistics under ``model``.

    Returns a list (len == len(texts)); each entry is a dict with:
      ``logp``  : np.array (T-1,) of log p(x_t | x_<t) for the realised tokens.
      ``mu``    : np.array (T-1,) expected log-prob over the vocab (calib only).
      ``sigma`` : np.array (T-1,) std of log-prob over the vocab (calib only).
    Empty sequences yield empty arrays.
    """
    model.eval()
    out_list = []
    for i in tqdm(range(0, len(texts), batch_size), desc=desc):
        batch = texts[i:i + batch_size]
        enc = tokenizer(
            batch, padding=True, truncation=True,
            max_length=MAX_SEQ_LEN, return_tensors="pt",
        ).to(device)
        input_ids = enc["input_ids"]
        attn      = enc["attention_mask"]
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attn).logits

        shift_logits = logits[:, :-1, :]          # (B, T-1, V)
        shift_labels = input_ids[:, 1:]           # (B, T-1)
        shift_mask   = attn[:, 1:].bool()         # (B, T-1)

        log_probs = torch.log_softmax(shift_logits, dim=-1)
        token_logp = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)  # (B,T-1)

        mu = sigma = None
        if need_calib:
            probs = log_probs.exp()
            mu = (probs * log_probs).sum(dim=-1)                       # (B,T-1)
            var = (probs * (log_probs - mu.unsqueeze(-1)) ** 2).sum(dim=-1)
            sigma = var.clamp_min(0).sqrt()

        token_logp = token_logp.cpu().float().numpy()
        m = shift_mask.cpu().numpy()
        if need_calib:
            mu_np = mu.cpu().float().numpy()
            sigma_np = sigma.cpu().float().numpy()
        for b in range(len(batch)):
            valid = m[b]
            entry = {"logp": token_logp[b][valid]}
            if need_calib:
                entry["mu"] = mu_np[b][valid]
                entry["sigma"] = sigma_np[b][valid]
            out_list.append(entry)
    return out_list


def _class_means(scores, labels, name):
    labels = np.asarray(labels).astype(int)
    t = scores[labels == 1]
    f = scores[labels == 0]
    t = t[~np.isnan(t)]
    f = f[~np.isnan(f)]
    mt = float(t.mean()) if len(t) else float("nan")
    mf = float(f.mean()) if len(f) else float("nan")
    print(f"[overlap:{name}]   mean target={mt:.4f}  mean forget={mf:.4f}  "
          f"(higher target ⇒ separable)")
    return mt, mf


# ─────────────────────────────────────────────────────────────────────────────
# Feature bundle: base + LoRA token log-probs computed once, reused by PPL/MinK++
# ─────────────────────────────────────────────────────────────────────────────
def token_logp_features(u_df: pd.DataFrame, checkpoint_path: Path, device,
                        need_calib=True, batch_size=8) -> dict:
    """Compute per-text token log-probs under base and LoRA models for U."""
    tokenizer = get_tokenizer()
    texts = u_df["text"].tolist()

    print("[overlap] base token log-probs")
    base = load_base_model(device)
    base_feats = _token_logps(base, tokenizer, texts, device,
                              need_calib=need_calib, batch_size=batch_size,
                              desc="tok-base")
    del base
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("[overlap] LoRA token log-probs")
    lora = load_lora_model(checkpoint_path, device)
    lora_feats = _token_logps(lora, tokenizer, texts, device,
                              need_calib=need_calib, batch_size=batch_size,
                              desc="tok-lora")
    del lora
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {"base": base_feats, "lora": lora_feats}


# ─────────────────────────────────────────────────────────────────────────────
# 1) PPL ratio  =  mean-token logp(LoRA) − mean-token logp(base)
# ─────────────────────────────────────────────────────────────────────────────
def ppl_ratio_from_features(feats: dict) -> np.ndarray:
    base, lora = feats["base"], feats["lora"]
    n = len(base)
    scores = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        lb, ll = base[i]["logp"], lora[i]["logp"]
        if len(lb) == 0 or len(ll) == 0:
            continue
        scores[i] = float(ll.mean() - lb.mean())   # = log PPL_base − log PPL_lora
    return scores


# ─────────────────────────────────────────────────────────────────────────────
# 2) Min-K%++ gap
# ─────────────────────────────────────────────────────────────────────────────
def _minkpp_single(logp, mu, sigma, frac) -> float:
    """Min-K%++ score under one model: average of the calibrated log-probs at the
    k% most-surprising tokens (lowest normalised score). Higher ⇒ member-like."""
    if len(logp) == 0:
        return float("nan")
    z = (logp - mu) / (sigma + _EPS)               # per-token calibrated score
    k = max(1, int(np.ceil(frac * len(z))))
    lowest = np.sort(z)[:k]                          # k most-surprising tokens
    return float(lowest.mean())


def minkpp_from_features(feats: dict, frac=MINK_FRAC) -> np.ndarray:
    """Gap of Min-K%++ scores: LoRA minus base (target-likeness gained)."""
    base, lora = feats["base"], feats["lora"]
    n = len(base)
    scores = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        b, l = base[i], lora[i]
        if len(l["logp"]) == 0 or len(b["logp"]) == 0:
            continue
        s_lora = _minkpp_single(l["logp"], l["mu"], l["sigma"], frac)
        s_base = _minkpp_single(b["logp"], b["mu"], b["sigma"], frac)
        scores[i] = s_lora - s_base
    return scores


# ─────────────────────────────────────────────────────────────────────────────
# 3) Embedding similarity to the P (reference target) centroid
# ─────────────────────────────────────────────────────────────────────────────
def compute_embedding_scores(u_df, p_df, checkpoint_path, device) -> np.ndarray:
    """Cosine similarity of each U row's LoRA hidden state to the centroid of the
    P (labeled-positive / reference target) pool. P must be disjoint from U and
    from the LoRA-train set (guaranteed by ``make_pu_split``)."""
    if not (p_df["quadrant_label"] == 1).all():
        raise ValueError("P (reference) pool must be strictly label==1")
    p_keys = set(zip(p_df["source"], p_df["id"]))
    u_keys = set(zip(u_df["source"], u_df["id"]))
    if p_keys & u_keys:
        raise ValueError("P and U share (source,id) keys — would cause self-influence")

    tokenizer = get_tokenizer()
    lora = load_lora_model(checkpoint_path, device)
    emb_p = extract_last_layer(lora, tokenizer, p_df["text"].tolist(), device,
                               desc="emb-P").astype(np.float64)
    emb_u = extract_last_layer(lora, tokenizer, u_df["text"].tolist(), device,
                               desc="emb-U").astype(np.float64)
    del lora
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    def _norm(x):
        return x / (np.linalg.norm(x, axis=1, keepdims=True) + _EPS)

    emb_p, emb_u = _norm(emb_p), _norm(emb_u)
    centroid = emb_p.mean(axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + _EPS)
    scores = emb_u @ centroid
    _class_means(scores, u_df["quadrant_label"].values, "embedding")
    return scores.astype(np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# 4/5) LiRA (lite = one-sided OUT; full = two-sided IN/OUT) from shadow LoRAs
# ─────────────────────────────────────────────────────────────────────────────
def _mean_nll(model, tokenizer, texts, device, batch_size=8, desc="nll"):
    """Per-text mean negative log-likelihood under ``model``."""
    feats = _token_logps(model, tokenizer, texts, device, need_calib=False,
                         batch_size=batch_size, desc=desc)
    out = np.full(len(texts), np.nan, dtype=np.float64)
    for i, f in enumerate(feats):
        if len(f["logp"]) > 0:
            out[i] = float(-f["logp"].mean())
    return out


_LIRA_CLIP = 30.0   # clip z-scores / log-ratios to keep outliers finite


def _load_lora_retry(ckpt, device, tries=6, wait=1.0):
    """Load a LoRA adapter, tolerating brief filesystem-visibility races right
    after a shadow was trained+saved (adapter_config.json not yet visible)."""
    cfg = Path(ckpt) / "adapter_config.json"
    for i in range(tries):
        if cfg.exists():
            return load_lora_model(ckpt, device)
        time.sleep(wait)
    if not cfg.exists():
        raise FileNotFoundError(f"adapter_config.json never appeared at {ckpt}")
    return load_lora_model(ckpt, device)


def compute_lira_scores(
    u_df, target_checkpoint, shadow_checkpoints, shadow_key_sets, device,
    mode="lite", batch_size=8,
) -> np.ndarray:
    """LiRA-style target-membership score for every row in U.

    The per-example statistic is the **adaptation benefit** relative to the base
    model,  s(x, θ) = base_nll(x) − θ_nll(x)  (higher ⇒ the adapter fits x better
    ⇒ more target-like). Using a base reference makes the statistic comparable
    across adapters of different strength (the main adapter trains on fewer rows
    than the shadows), which a raw-loss LiRA gets wrong.

    For each x we form:
      * OUT distribution — s(x, shadow) over shadows that did NOT train on x.
      * IN  distribution — s(x, shadow) over shadows that DID train on x
                           (only exists for some target rows; never for forget).

    Every U row is a non-member of the *main* adapter (it trained only on the
    held-back lora_train rows), so a classic observed-vs-OUT LiRA is confounded
    by adapter strength. We therefore key membership off the shadows:

    mode : 'lite' → ensemble OUT benefit  mean_{s: x∉train(s)} s(x, shadow_s).
                    A variance-reduced PPL ratio: how much do target-trained
                    models that did NOT see x still reduce its loss (generalise)?
           'full' → OUT-calibrated Gaussian gap  (s_ref − μ_out) / σ_out, with
                    s_ref = mean IN-shadow benefit when x was seen by ≥2 shadows
                    (memorisation signal), else the main adapter's benefit.

    Higher ⇒ more target-like. Variances are floored (relative to the global OUT
    scale) and full scores clipped to ±30.
    """
    if len(shadow_checkpoints) == 0:
        raise ValueError("LiRA needs at least one shadow checkpoint")

    tokenizer = get_tokenizer()
    texts = u_df["text"].tolist()
    keys = list(zip(u_df["source"], u_df["id"]))
    n = len(texts)

    base = load_base_model(device)
    base_nll = _mean_nll(base, tokenizer, texts, device, batch_size, desc="lira-base")
    del base
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    target = load_lora_model(target_checkpoint, device)
    tgt_nll = _mean_nll(target, tokenizer, texts, device, batch_size, desc="lira-obs")
    del target
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    shadow_nll = np.full((len(shadow_checkpoints), n), np.nan, dtype=np.float64)
    for s, ckpt in enumerate(shadow_checkpoints):
        m = _load_lora_retry(ckpt, device)
        shadow_nll[s] = _mean_nll(m, tokenizer, texts, device, batch_size,
                                  desc=f"lira-shadow{s}")
        del m
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # adaptation-benefit statistic (higher = more target-like)
    obs_stat    = base_nll - tgt_nll                       # (n,)
    shadow_stat = base_nll[None, :] - shadow_nll           # (S, n)

    # global OUT scale for variance flooring
    out_mask = np.ones_like(shadow_stat, dtype=bool)
    for s in range(len(shadow_checkpoints)):
        for i in range(n):
            if keys[i] in shadow_key_sets[s]:
                out_mask[s, i] = False
    finite_out = shadow_stat[out_mask & ~np.isnan(shadow_stat)]
    global_sd = float(np.std(finite_out)) if finite_out.size else 1.0
    sd_min = max(0.05 * global_sd, 1e-3)

    def _clip(v):
        return float(np.clip(v, -_LIRA_CLIP, _LIRA_CLIP))

    scores = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        in_s, out_s = [], []
        for s in range(len(shadow_checkpoints)):
            v = shadow_stat[s, i]
            if np.isnan(v):
                continue
            (in_s if keys[i] in shadow_key_sets[s] else out_s).append(v)
        if len(out_s) < 2:
            continue
        mu_out = float(np.mean(out_s))
        sd_out = max(float(np.std(out_s)), sd_min)
        if mode == "lite":
            scores[i] = mu_out                       # ensemble OUT benefit
        else:  # full — OUT-calibrated gap, IN benefit as observed when available
            s_ref = float(np.mean(in_s)) if len(in_s) >= 2 else obs_stat[i]
            if np.isnan(s_ref):
                continue
            scores[i] = _clip((s_ref - mu_out) / sd_out)

    _class_means(scores, u_df["quadrant_label"].values, f"lira_{mode}")
    return scores
