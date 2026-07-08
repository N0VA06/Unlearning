"""Active-learning sample selection for the LoRA training subset.

We replace random sampling with **informative = uncertain + diverse**:

  * uncertainty signals (from the base GPT-2, before any LoRA training):
      - mean per-token cross-entropy loss   (high  → model finds the text surprising)
      - mean per-token next-token entropy   (high  → model is unsure)
      - gradient norm of ∇‖θ‖ on the LM loss (high  → strong adaptation signal)

  * diversity:
      - K-means clusters on base GPT-2 mean-pooled last-layer embeddings
        (k = number of training rows to select); one sample per cluster

The selected training subset has one representative per cluster (diverse)
chosen as the **highest-uncertainty** member of that cluster (informative).

All signals come from the **untouched base model**, so selection is causally
independent of the eventual LoRA adapter — no circularity, no leakage.
"""

from typing import Optional

import numpy as np
import torch
from tqdm import tqdm
from sklearn.cluster import KMeans

from config import MAX_SEQ_LEN, PROBE_BATCH
from src.hidden_states import get_tokenizer, load_base_model, extract_last_layer


def _minmax(x: np.ndarray) -> np.ndarray:
    rng = float(x.max() - x.min())
    if rng < 1e-12:
        return np.zeros_like(x)
    return (x - x.min()) / rng


def compute_uncertainty_signals(
    base_model,
    tokenizer,
    texts,
    device,
    include_grad_norm: bool = True,
    batch_size: int = PROBE_BATCH,
) -> dict:
    """Return per-sample uncertainty signals from the base model.

    Outputs (each is an ndarray of length len(texts)):
      'loss'      — mean per-token cross-entropy on the LM loss
      'entropy'   — mean per-token next-token entropy of the softmax
      'grad_norm' — L2 norm of ∇‖θ‖ on the LM loss (None if include_grad_norm is False)
    """
    n = len(texts)
    losses    = np.zeros(n, dtype=np.float64)
    entropies = np.zeros(n, dtype=np.float64)

    # Pass 1 — batched forward for loss + entropy
    base_model.eval()
    for i in tqdm(range(0, n, batch_size), desc="signals-batched"):
        batch = texts[i:i + batch_size]
        enc = tokenizer(
            batch, padding=True, truncation=True,
            max_length=MAX_SEQ_LEN, return_tensors="pt",
        ).to(device)
        input_ids = enc["input_ids"]
        attn      = enc["attention_mask"]

        with torch.no_grad():
            logits = base_model(input_ids=input_ids, attention_mask=attn).logits

        # Shift for next-token prediction
        shift_logits = logits[:, :-1].contiguous()      # (B, T-1, V)
        shift_labels = input_ids[:, 1:].contiguous()     # (B, T-1)
        shift_mask   = attn[:, 1:].contiguous().float()  # (B, T-1)
        denom = shift_mask.sum(dim=1).clamp_min(1.0)

        # Per-token cross-entropy
        ce = torch.nn.functional.cross_entropy(
            shift_logits.transpose(1, 2), shift_labels,
            reduction="none",
        )                                                 # (B, T-1)
        per_text_loss = (ce * shift_mask).sum(dim=1) / denom

        # Per-token entropy
        probs     = torch.softmax(shift_logits, dim=-1)
        log_probs = torch.log_softmax(shift_logits, dim=-1)
        ent       = -(probs * log_probs).sum(dim=-1)      # (B, T-1)
        per_text_entropy = (ent * shift_mask).sum(dim=1) / denom

        losses[i:i + len(batch)]    = per_text_loss.cpu().numpy()
        entropies[i:i + len(batch)] = per_text_entropy.cpu().numpy()

    if not include_grad_norm:
        return {"loss": losses, "entropy": entropies, "grad_norm": None}

    # Pass 2 — per-sample forward + backward for gradient norm.
    grad_norms = np.zeros(n, dtype=np.float64)
    base_model.train()
    for j, txt in enumerate(tqdm(texts, desc="signals-gradnorm")):
        enc = tokenizer(
            txt, truncation=True, max_length=MAX_SEQ_LEN, return_tensors="pt",
        ).to(device)
        input_ids = enc["input_ids"]
        base_model.zero_grad(set_to_none=True)
        out = base_model(input_ids=input_ids, labels=input_ids)
        out.loss.backward()
        total = 0.0
        for p in base_model.parameters():
            if p.grad is not None:
                total += float((p.grad ** 2).sum().item())
        grad_norms[j] = total ** 0.5
        base_model.zero_grad(set_to_none=True)
        del out
    base_model.eval()

    return {"loss": losses, "entropy": entropies, "grad_norm": grad_norms}


def compose_uncertainty(signals: dict, weights: Optional[dict] = None) -> np.ndarray:
    """Combine the three signals into a single per-sample score in roughly
    [0, 1]. Each component is min-max normalized first so they're comparable."""
    weights = weights or {"loss": 1.0, "entropy": 1.0, "grad_norm": 1.0}
    composite, total_w = None, 0.0
    for name, w in weights.items():
        v = signals.get(name)
        if v is None:
            continue
        norm = _minmax(np.asarray(v, dtype=np.float64))
        composite = norm * w if composite is None else composite + norm * w
        total_w  += w
    if composite is None or total_w <= 0:
        raise ValueError("No usable uncertainty signals supplied.")
    return composite / total_w


def select_uncertain_diverse(
    embeddings: np.ndarray,
    uncertainty: np.ndarray,
    n_select: int,
    seed: int = 42,
) -> np.ndarray:
    """K-means clusters on `embeddings` for diversity; within each cluster
    keep the row with highest `uncertainty`. Returns sorted indices."""
    n = embeddings.shape[0]
    if n_select <= 0:
        raise ValueError(f"n_select must be positive, got {n_select}")
    if uncertainty.shape[0] != n:
        raise ValueError(
            f"uncertainty length {uncertainty.shape[0]} != n_rows {n}"
        )
    if n_select >= n:
        return np.arange(n)
    if n_select == 1:
        return np.array([int(np.argmax(uncertainty))])

    km = KMeans(n_clusters=n_select, random_state=seed, n_init=10).fit(embeddings)

    selected, used = [], set()
    for c in range(n_select):
        in_cluster = np.where(km.labels_ == c)[0]
        if in_cluster.size == 0:
            continue
        order = in_cluster[np.argsort(uncertainty[in_cluster])[::-1]]
        for cand in order:
            if int(cand) not in used:
                selected.append(int(cand))
                used.add(int(cand))
                break

    # Backfill empty clusters with the next most-uncertain unused row.
    if len(selected) < n_select:
        leftover = sorted(set(range(n)) - used, key=lambda i: -float(uncertainty[i]))
        for cand in leftover[: n_select - len(selected)]:
            selected.append(int(cand))

    return np.array(sorted(selected), dtype=np.int64)


def gather_selection_inputs(
    texts,
    device,
    include_grad_norm: bool = True,
) -> dict:
    """Convenience: load base GPT-2, return embeddings + uncertainty signals.

    Caller is responsible for the (small) GPU memory; the function frees the
    base model before returning."""
    tokenizer = get_tokenizer()
    base = load_base_model(device)
    print(f"[sample_select] extracting embeddings (base GPT-2, {len(texts)} texts)")
    embeddings = extract_last_layer(base, tokenizer, texts, device, desc="select-emb")
    print(f"[sample_select] computing uncertainty signals (loss, entropy"
          + (", grad_norm" if include_grad_norm else "")
          + ")")
    signals = compute_uncertainty_signals(
        base, tokenizer, texts, device,
        include_grad_norm=include_grad_norm,
    )
    del base
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    signals["embeddings"] = embeddings
    return signals
