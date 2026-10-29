"""Shadow LoRA adapters for LiRA overlap estimation.

LiRA needs a distribution of "reference" models to tell whether a sample's loss
under the target adapter is unusually low. We approximate this by training a
handful of *shadow* LoRA adapters, each on a random subset of the target class.
For any sample we can then form an OUT (non-member) loss distribution — and, for
target samples that happen to land in a shadow's subset, an IN distribution.

This is compute-heavy (LIRA_N_SHADOWS adapters per quadrant) and is therefore a
standalone utility, invoked explicitly (see experiments/run_overlap_scorers.py
--train-shadows), not part of the default pipeline.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model

from config import (
    BASE_MODEL, CHECKPOINTS, QUADRANTS, MAX_SEQ_LEN,
    LORA_R, LORA_ALPHA, LORA_DROPOUT, TRAIN_EPOCHS, LR, BATCH_SIZE,
    LIRA_N_SHADOWS, LIRA_SHADOW_FRAC, LIRA_SEED,
)
from src.lora_trainer import TextDataset


def _shadow_dir(qid: str, s: int) -> Path:
    return CHECKPOINTS / "shadows" / f"{qid}_{QUADRANTS[qid]['name']}_shadow{s}"


def _train_one(train_df: pd.DataFrame, ckpt_dir: Path, device):
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float32)
    lora_cfg = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, target_modules=["c_attn"],
        lora_dropout=LORA_DROPOUT, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.to(device).train()

    dataset = TextDataset(train_df["text"].tolist(), tokenizer, MAX_SEQ_LEN)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    steps = TRAIN_EPOCHS * len(train_df)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=max(1, min(30, steps // 10)),
        num_training_steps=steps,
    )

    step = 0
    while step < steps:
        for batch in loader:
            if step >= steps:
                break
            out = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
            )
            out.loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def train_shadow_loras(qid: str, pool_df: pd.DataFrame,
                       n_shadows: int = LIRA_N_SHADOWS,
                       frac: float = LIRA_SHADOW_FRAC,
                       seed: int = LIRA_SEED):
    """Train ``n_shadows`` shadow adapters on random subsets of ``pool_df``.

    IMPORTANT (leakage): ``pool_df`` must be the **labeled-positive** pool only
    (lora_train ∪ P) — never the held-out target rows that live in U. Sampling
    U rows would let LiRA's IN branch read the very label it is meant to predict.
    The caller (run_overlap_scorers) enforces this; here we assert label==1.

    Returns a list of (checkpoint_dir: Path, train_keys: set[(source,id)]).
    Existing shadow checkpoints are reused (skipped) if present.
    """
    if "quadrant_label" in pool_df and not (pool_df["quadrant_label"] == 1).all():
        raise ValueError("shadow pool_df must contain only labeled positives (label==1)")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    target_df = pool_df.reset_index(drop=True)
    n_take = max(1, int(round(len(target_df) * frac)))

    shadows = []
    for s in range(n_shadows):
        rng = np.random.default_rng(seed + s)
        pick = np.sort(rng.choice(len(target_df), size=n_take, replace=False))
        sub = target_df.iloc[pick].reset_index(drop=True)
        keys = set(zip(sub["source"].tolist(), sub["id"].tolist()))
        ckpt_dir = _shadow_dir(qid, s)

        if not (ckpt_dir / "adapter_config.json").exists():
            print(f"[shadow] {qid} shadow {s}/{n_shadows}: training on {len(sub)} target rows")
            _train_one(sub, ckpt_dir, device)
        else:
            print(f"[shadow] {qid} shadow {s}: reuse {ckpt_dir}")

        with (ckpt_dir / "shadow_keys.json").open("w") as f:
            json.dump([list(k) for k in keys], f)
        shadows.append((ckpt_dir, keys))
    return shadows


def load_shadow_loras(qid: str, n_shadows: int = LIRA_N_SHADOWS):
    """Load existing shadow checkpoints + their key sets (no training)."""
    shadows = []
    for s in range(n_shadows):
        ckpt_dir = _shadow_dir(qid, s)
        kpath = ckpt_dir / "shadow_keys.json"
        if not (ckpt_dir / "adapter_config.json").exists() or not kpath.exists():
            continue
        with kpath.open("r") as f:
            keys = set(tuple(k) for k in json.load(f))
        shadows.append((ckpt_dir, keys))
    return shadows
