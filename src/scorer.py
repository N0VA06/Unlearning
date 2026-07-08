import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from config import BASE_MODEL, MAX_SEQ_LEN
from src.hidden_states import (
    get_tokenizer, load_lora_model, extract_last_layer,
)


def _ppl_single(text: str, model, tokenizer, device) -> float:
    enc = tokenizer(
        text,
        truncation=True,
        max_length=MAX_SEQ_LEN,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(device)
    if input_ids.shape[1] == 0:
        return float("nan")

    with torch.no_grad():
        out = model(input_ids=input_ids, labels=input_ids)
    loss = out.loss

    if loss is None:
        return float("nan")
    loss_val = float(loss.item())
    if math.isnan(loss_val) or math.isinf(loss_val):
        return float("nan")
    try:
        return math.exp(loss_val)
    except OverflowError:
        return float("nan")


def compute_ppl_ratio_scores(df: pd.DataFrame, checkpoint_path: Path, device) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[scorer] loading base GPT-2 (no adapter)")
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float32)
    assert not hasattr(base_model, "peft_config"), "Base model must not have LoRA adapter loaded"
    base_model.to(device).eval()

    print(f"[scorer] loading GPT-2 + LoRA adapter from {checkpoint_path}")
    lora_base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float32)
    lora_model = PeftModel.from_pretrained(lora_base, str(checkpoint_path))
    lora_model.to(device).eval()

    texts = df["text"].tolist()
    n = len(texts)
    ppl_base = np.full(n, np.nan, dtype=np.float64)
    ppl_lora = np.full(n, np.nan, dtype=np.float64)

    for i, txt in enumerate(tqdm(texts, desc="ppl_base")):
        ppl_base[i] = _ppl_single(txt, base_model, tokenizer, device)
    for i, txt in enumerate(tqdm(texts, desc="ppl_lora")):
        ppl_lora[i] = _ppl_single(txt, lora_model, tokenizer, device)

    score_ppl = np.full(n, np.nan, dtype=np.float64)
    valid = (~np.isnan(ppl_base)) & (~np.isnan(ppl_lora))
    score_ppl[valid] = np.log(ppl_base[valid] + 1e-8) - np.log(ppl_lora[valid] + 1e-8)

    labels      = df["quadrant_label"].values
    target_mask = (labels == 1)
    forget_mask = (labels == 0)

    def _mean(arr, mask):
        sub = arr[mask]
        sub = sub[~np.isnan(sub)]
        return float(np.mean(sub)) if len(sub) > 0 else float("nan")

    print(f"[scorer]   mean score_ppl (target) = {_mean(score_ppl, target_mask):.4f}")
    print(f"[scorer]   mean score_ppl (forget) = {_mean(score_ppl, forget_mask):.4f}")
    print(f"[scorer]   NaN count: target={int(np.isnan(score_ppl[target_mask]).sum())}, "
          f"forget={int(np.isnan(score_ppl[forget_mask]).sum())}")

    del base_model, lora_model, lora_base
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "ppl_base":  ppl_base,
        "ppl_lora":  ppl_lora,
        "score_ppl": score_ppl,
    }


def compute_embedding_scores(
    eval_df: pd.DataFrame,
    centroid_source_df: pd.DataFrame,
    checkpoint_path: Path,
    device,
) -> dict:
    """Score every row in `eval_df` against a target centroid built from
    `centroid_source_df` (must be strictly label=1).

    The embedding source is the LoRA-adapted GPT-2's mean-pooled last-layer
    hidden state — *not* a sentence-transformer. This makes the embedding
    score and the activation-drift analysis read from the same hidden state
    stream, so any geometry change LoRA induced affects both views
    consistently.

    Disjointness on (source, id) between centroid source and eval set is
    asserted before any extraction to prevent self-influence in the centroid.
    """
    if not (centroid_source_df["quadrant_label"] == 1).all():
        raise ValueError("centroid_source_df must contain only label=1 rows")

    src_keys  = set(zip(centroid_source_df["source"].tolist(),
                        centroid_source_df["id"].tolist()))
    eval_keys = set(zip(eval_df["source"].tolist(), eval_df["id"].tolist()))
    overlap_keys = src_keys & eval_keys
    if overlap_keys:
        raise ValueError(
            f"centroid_source and eval set share {len(overlap_keys)} (source,id) key(s) — "
            f"this would cause self-influence. Make them disjoint."
        )

    tokenizer  = get_tokenizer()
    lora_model = load_lora_model(checkpoint_path, device)

    print(f"[scorer]   extracting LoRA last-layer embeddings  "
          f"(centroid={len(centroid_source_df)}, eval={len(eval_df)})")
    embeddings_centroid_src = extract_last_layer(
        lora_model, tokenizer,
        centroid_source_df["text"].tolist(), device,
        desc="emb-centroid",
    ).astype(np.float64)
    embeddings_eval = extract_last_layer(
        lora_model, tokenizer,
        eval_df["text"].tolist(), device,
        desc="emb-eval",
    ).astype(np.float64)

    del lora_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    assert embeddings_centroid_src.shape[0] == len(centroid_source_df)
    assert embeddings_eval.shape[0] == len(eval_df)

    # L2-normalize so dot product == cosine similarity
    def _normalize(x):
        return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)

    embeddings_centroid_src = _normalize(embeddings_centroid_src)
    embeddings_eval         = _normalize(embeddings_eval)

    centroid = embeddings_centroid_src.mean(axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-9)

    score_emb = embeddings_eval @ centroid

    labels      = eval_df["quadrant_label"].values
    target_mask = (labels == 1)
    forget_mask = (labels == 0)

    cos_t = float("nan")
    cos_f = float("nan")
    if target_mask.any():
        v = embeddings_eval[target_mask].mean(axis=0)
        cos_t = float(np.dot(v / (np.linalg.norm(v) + 1e-9), centroid))
    if forget_mask.any():
        v = embeddings_eval[forget_mask].mean(axis=0)
        cos_f = float(np.dot(v / (np.linalg.norm(v) + 1e-9), centroid))

    print(f"[scorer]   mean score_emb (held-out target) = {score_emb[target_mask].mean():.4f}")
    print(f"[scorer]   mean score_emb (forget)          = {score_emb[forget_mask].mean():.4f}")
    print(f"[scorer]   cos(mean held-out target, centroid) = {cos_t:.4f}")
    print(f"[scorer]   cos(mean forget,           centroid) = {cos_f:.4f}")

    return {
        "embeddings": embeddings_eval,
        "score_emb":  score_emb.astype(np.float64),
        "centroid":   centroid,
        "source":     "lora_last_layer",
    }


def compute_combined_scores(score_ppl: np.ndarray, score_emb: np.ndarray) -> np.ndarray:
    if score_ppl.shape != score_emb.shape:
        raise ValueError(f"score_ppl shape {score_ppl.shape} != score_emb shape {score_emb.shape}")

    valid = (~np.isnan(score_ppl)) & (~np.isnan(score_emb))
    score_combined = np.full_like(score_ppl, np.nan, dtype=np.float64)

    if valid.sum() == 0:
        return score_combined

    def _minmax(x):
        rng = x.max() - x.min()
        return (x - x.min()) / (rng + 1e-9)

    ppl_n = _minmax(score_ppl[valid])
    emb_n = _minmax(score_emb[valid])
    combined_v = 0.4 * ppl_n + 0.6 * emb_n

    score_combined[valid] = combined_v
    return score_combined
