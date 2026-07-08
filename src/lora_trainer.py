from pathlib import Path

import torch
import pandas as pd
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model
from tqdm import tqdm

from config import (
    BASE_MODEL, CHECKPOINTS, QUADRANTS, MAX_SEQ_LEN,
    LORA_R, LORA_ALPHA, LORA_DROPOUT, TRAIN_EPOCHS, LR, BATCH_SIZE,
    SPLIT_RATIO, SPLIT_SEED,
)


class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        input_ids      = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        labels         = input_ids.clone()
        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
        }


def _checkpoint_dir(qid: str) -> Path:
    """Checkpoint directory whose name encodes the split params so a checkpoint
    trained with one split can never be silently reused under a different split."""
    qmeta = QUADRANTS[qid]
    tag = f"r{int(round(SPLIT_RATIO * 100)):02d}_s{SPLIT_SEED}_e{TRAIN_EPOCHS}"
    return CHECKPOINTS / f"{qid}_{qmeta['name']}_{tag}"


def train_lora(qid: str, train_df: pd.DataFrame) -> Path:
    """Train a LoRA adapter on the supplied target subset (must be strictly label=1)."""
    qmeta     = QUADRANTS[qid]
    ckpt_dir  = _checkpoint_dir(qid)

    if (ckpt_dir / "adapter_config.json").exists():
        print(f"[SKIP] Checkpoint found for {qid}, loading existing adapter at {ckpt_dir}")
        return ckpt_dir

    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[lora_trainer] {qid}: device={device}, ckpt_dir={ckpt_dir}")

    train_df = train_df.reset_index(drop=True)
    if len(train_df) == 0:
        raise ValueError(f"No target records to train on for quadrant {qid}")
    assert (train_df["quadrant_label"] == 1).all(), "Training data contains non-target records"

    train_steps = TRAIN_EPOCHS * len(train_df)
    print(f"[lora_trainer] {qid}: training on {len(train_df)} target records (label=1 only), "
          f"{TRAIN_EPOCHS} epoch(s) = {train_steps} steps")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float32)
    lora_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["c_attn"],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.to(device)
    model.train()

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total     = sum(p.numel() for p in model.parameters())
    print(f"[lora_trainer] {qid}: trainable params = {n_trainable:,} / {n_total:,} "
          f"({100.0 * n_trainable / n_total:.3f}%)")

    dataset = TextDataset(train_df["text"].tolist(), tokenizer, MAX_SEQ_LEN)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    warmup_steps = max(1, min(30, train_steps // 10))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=train_steps,
    )

    step = 0
    pbar = tqdm(total=train_steps, desc=f"train[{qid}]")
    running_loss = 0.0
    log_window = max(1, train_steps // 10)
    while step < train_steps:
        for batch in loader:
            if step >= train_steps:
                break
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            out  = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = out.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            running_loss += loss.item()
            step += 1
            pbar.update(1)

            if step % log_window == 0:
                avg = running_loss / log_window
                print(f"[lora_trainer] {qid}: step {step}/{train_steps}, "
                      f"avg loss (last {log_window}) = {avg:.4f}")
                running_loss = 0.0
    pbar.close()

    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)

    import json
    split_info = {
        "qid":          qid,
        "split_ratio":  float(SPLIT_RATIO),
        "split_seed":   int(SPLIT_SEED),
        "train_epochs": int(TRAIN_EPOCHS),
        "train_steps":  int(train_steps),
        "n_train":      int(len(train_df)),
        "train_keys":   [list(t) for t in zip(train_df["source"].tolist(),
                                               train_df["id"].tolist())],
    }
    with (ckpt_dir / "split_info.json").open("w") as f:
        json.dump(split_info, f, indent=2)

    print(f"[lora_trainer] {qid}: saved adapter to {ckpt_dir}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return ckpt_dir
