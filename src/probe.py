"""Layerwise activation-drift analysis.

For every transformer layer we compute, per text, the cosine distance between
the base GPT-2 hidden state and the LoRA-adapted GPT-2 hidden state:

    drift_l(x) = 1 - cos( h_base_l(x) , h_lora_l(x) )

Aggregated per class, this answers the question

    "Where did LoRA move representations, and did it move target rows
     more than forget rows?"

This is a diagnostic about how training changed the model — not about whether
target/forget are separable at any given layer (the earlier probe-AUROC view).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.hidden_states import (
    get_tokenizer, load_base_model, load_lora_model, extract_all_layers,
)


def _cosine_drift(h_base: np.ndarray, h_lora: np.ndarray) -> np.ndarray:
    """h_base, h_lora: (N, D)   →  per-row cosine distance (N,)."""
    bn = h_base / (np.linalg.norm(h_base, axis=1, keepdims=True) + 1e-9)
    ln = h_lora / (np.linalg.norm(h_lora, axis=1, keepdims=True) + 1e-9)
    return 1.0 - np.sum(bn * ln, axis=1)


def _l2_drift(h_base: np.ndarray, h_lora: np.ndarray) -> np.ndarray:
    """Normalized L2 distance:  ||h_lora - h_base|| / ||h_base||."""
    diff = h_lora - h_base
    return np.linalg.norm(diff, axis=1) / (np.linalg.norm(h_base, axis=1) + 1e-9)


def run_layerwise_drift(eval_df: pd.DataFrame, checkpoint_path: Path, device) -> dict:
    """Compute per-layer mean and std cosine drift for target / forget rows."""
    tok = get_tokenizer()
    texts  = eval_df["text"].tolist()
    labels = eval_df["quadrant_label"].values.astype(int)

    print(f"[drift] base hidden states on {len(texts)} eval texts")
    base = load_base_model(device)
    feats_base = extract_all_layers(base, tok, texts, device, desc="drift-base")
    del base
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"[drift] LoRA hidden states on {len(texts)} eval texts")
    lora = load_lora_model(checkpoint_path, device)
    feats_lora = extract_all_layers(lora, tok, texts, device, desc="drift-lora")
    del lora
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    assert feats_base.shape == feats_lora.shape, \
        f"base/lora feature shapes differ: {feats_base.shape} vs {feats_lora.shape}"

    n_layers = feats_base.shape[1]
    target_mask = labels == 1
    forget_mask = labels == 0

    target_mean = np.zeros(n_layers, dtype=np.float64)
    target_std  = np.zeros(n_layers, dtype=np.float64)
    forget_mean = np.zeros(n_layers, dtype=np.float64)
    forget_std  = np.zeros(n_layers, dtype=np.float64)
    target_mean_l2 = np.zeros(n_layers, dtype=np.float64)
    forget_mean_l2 = np.zeros(n_layers, dtype=np.float64)

    for layer in range(n_layers):
        d_cos = _cosine_drift(feats_base[:, layer, :], feats_lora[:, layer, :])
        d_l2  = _l2_drift    (feats_base[:, layer, :], feats_lora[:, layer, :])

        target_mean[layer] = float(d_cos[target_mask].mean()) if target_mask.any() else float("nan")
        target_std[layer]  = float(d_cos[target_mask].std())  if target_mask.any() else float("nan")
        forget_mean[layer] = float(d_cos[forget_mask].mean()) if forget_mask.any() else float("nan")
        forget_std[layer]  = float(d_cos[forget_mask].std())  if forget_mask.any() else float("nan")
        target_mean_l2[layer] = float(d_l2[target_mask].mean()) if target_mask.any() else float("nan")
        forget_mean_l2[layer] = float(d_l2[forget_mask].mean()) if forget_mask.any() else float("nan")

    delta = target_mean - forget_mean
    best_diff_layer = int(np.nanargmax(delta))

    print(f"[drift] per-layer cosine drift")
    print(f"  target  : {', '.join(f'{x:.4f}' for x in target_mean)}")
    print(f"  forget  : {', '.join(f'{x:.4f}' for x in forget_mean)}")
    print(f"  Δ(t-f)  : {', '.join(f'{x:+.4f}' for x in delta)}")
    print(f"  largest target-minus-forget drift at layer {best_diff_layer} (Δ={delta[best_diff_layer]:+.4f})")

    return {
        "n_layers":               int(n_layers),
        "pool":                   "mean",
        "metric":                 "cosine_distance",
        "target_drift_per_layer": [float(x) for x in target_mean],
        "target_std_per_layer":   [float(x) for x in target_std],
        "forget_drift_per_layer": [float(x) for x in forget_mean],
        "forget_std_per_layer":   [float(x) for x in forget_std],
        "target_l2_per_layer":    [float(x) for x in target_mean_l2],
        "forget_l2_per_layer":    [float(x) for x in forget_mean_l2],
        "delta_per_layer":        [float(x) for x in delta],
        "best_diff_layer":        best_diff_layer,
        "best_diff_value":        float(delta[best_diff_layer]),
    }
