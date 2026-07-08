"""Shared helper for extracting pooled hidden states from transformer models.

Used by three downstream consumers:
  * sample_select  — base GPT-2 last-layer embeddings for k-medoids selection
  * scorer         — LoRA last-layer mean-pooled embeddings for centroid + scoring
  * probe          — base AND LoRA all-layer mean-pooled embeddings for drift
"""

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from config import BASE_MODEL, MAX_SEQ_LEN, PROBE_BATCH, PROBE_POOL


def get_tokenizer():
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_base_model(device):
    m = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float32).to(device)
    m.eval()
    return m


def load_lora_model(checkpoint_path, device):
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float32)
    m = PeftModel.from_pretrained(base, str(checkpoint_path)).to(device)
    m.eval()
    return m


def _pool(hidden, attn_mask, mode: str):
    """hidden: (B, T, D)  attn_mask: (B, T)  → (B, D)"""
    if mode == "last":
        last_idx = attn_mask.sum(dim=1) - 1
        return hidden[torch.arange(hidden.shape[0]), last_idx]
    if mode == "mean":
        m = attn_mask.unsqueeze(-1).float()
        return (hidden * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
    raise ValueError(f"unknown pool mode: {mode}")


def extract_all_layers(
    model, tokenizer, texts, device,
    pool: str = PROBE_POOL,
    batch_size: int = PROBE_BATCH,
    desc: str = "extract",
) -> np.ndarray:
    """Return ndarray of shape (N, n_layers+1, hidden_dim) of pooled hidden
    states for every layer (embedding output + transformer blocks)."""
    model.eval()
    all_feats = []
    for i in tqdm(range(0, len(texts), batch_size), desc=desc):
        batch = texts[i:i + batch_size]
        enc = tokenizer(
            batch, padding=True, truncation=True,
            max_length=MAX_SEQ_LEN, return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            out = model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                output_hidden_states=True,
            )
        per_layer = [
            _pool(h, enc["attention_mask"], pool).cpu().float().numpy()
            for h in out.hidden_states
        ]
        all_feats.append(np.stack(per_layer, axis=1))  # (B, L+1, D)
    return np.concatenate(all_feats, axis=0)


def extract_last_layer(
    model, tokenizer, texts, device,
    pool: str = PROBE_POOL,
    batch_size: int = PROBE_BATCH,
    desc: str = "extract-last",
) -> np.ndarray:
    """Return ndarray of shape (N, hidden_dim) — the final hidden state only,
    pooled per text. Lighter than extract_all_layers when only the final
    embedding is needed."""
    model.eval()
    all_feats = []
    for i in tqdm(range(0, len(texts), batch_size), desc=desc):
        batch = texts[i:i + batch_size]
        enc = tokenizer(
            batch, padding=True, truncation=True,
            max_length=MAX_SEQ_LEN, return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            out = model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                output_hidden_states=True,
            )
        last = out.hidden_states[-1]
        pooled = _pool(last, enc["attention_mask"], pool)
        all_feats.append(pooled.cpu().float().numpy())
    return np.concatenate(all_feats, axis=0)
