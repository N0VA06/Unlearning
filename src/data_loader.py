import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import DATA_QUAD, QUADRANTS, SPLIT_SEED, SPLIT_RATIO


REQUIRED_COLS = ["id", "source", "text", "quadrant_label", "role"]


def load_quadrant(qid: str) -> pd.DataFrame:
    if qid not in QUADRANTS:
        raise KeyError(f"Unknown quadrant id: {qid}. Valid ids: {list(QUADRANTS.keys())}")

    qmeta = QUADRANTS[qid]
    path = DATA_QUAD / qmeta["dir"] / f"{qmeta['name']}_combined.json"

    if not path.exists():
        raise FileNotFoundError(f"Expected data file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        records = json.load(f)

    if not isinstance(records, list) or len(records) == 0:
        raise ValueError(f"Data file {path} is empty or not a JSON list")

    df = pd.DataFrame(records)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Data file {path} missing required columns: {missing}")

    df = df[REQUIRED_COLS].copy()
    df["quadrant_label"] = df["quadrant_label"].astype(int)

    labels_present = set(df["quadrant_label"].unique().tolist())
    if 0 not in labels_present or 1 not in labels_present:
        raise ValueError(
            f"Quadrant {qid} must contain both class labels 0 and 1; found {labels_present}"
        )

    n_total   = len(df)
    n_target  = int((df["quadrant_label"] == 1).sum())
    n_forget  = int((df["quadrant_label"] == 0).sum())
    char_len  = df["text"].astype(str).str.len()
    mean_t    = float(char_len[df["quadrant_label"] == 1].mean())
    mean_f    = float(char_len[df["quadrant_label"] == 0].mean())
    max_t     = int(char_len[df["quadrant_label"] == 1].max())
    max_f     = int(char_len[df["quadrant_label"] == 0].max())

    print(f"[data_loader] {qid} ({qmeta['label']}) loaded from {path.name}")
    print(f"[data_loader]   total records      = {n_total}")
    print(f"[data_loader]   target (label=1)   = {n_target}")
    print(f"[data_loader]   forget (label=0)   = {n_forget}")
    print(f"[data_loader]   target text chars  : mean={mean_t:.1f}, max={max_t}")
    print(f"[data_loader]   forget text chars  : mean={mean_f:.1f}, max={max_f}")

    return df


def make_informative_split(
    df: pd.DataFrame,
    target_signals: dict,
    ratio: float = SPLIT_RATIO,
    seed: int = SPLIT_SEED,
):
    """Pick training target rows using **uncertain + diverse** selection.

    `target_signals` is the dict returned by
    `sample_select.gather_selection_inputs`. It must contain
    `embeddings` (N, D) plus uncertainty signals (`loss`, `entropy`,
    `grad_norm`). All arrays are aligned with the rows of `df` where
    `quadrant_label == 1` (in df-positional order).

    Returns
    -------
    train_target_idx : np.ndarray   df-positional indices of LoRA training rows
    eval_idx         : np.ndarray   df-positional indices of held-out target + all forget
    """
    from src.sample_select import compose_uncertainty, select_uncertain_diverse

    if not (0.0 < ratio < 1.0):
        raise ValueError(f"ratio must be in (0,1); got {ratio}")

    labels = df["quadrant_label"].values.astype(int)
    target_pos = np.where(labels == 1)[0]
    forget_pos = np.where(labels == 0)[0]
    if len(target_pos) == 0 or len(forget_pos) == 0:
        raise ValueError("Both target (label=1) and forget (label=0) records are required.")

    embeddings = target_signals.get("embeddings")
    if embeddings is None:
        raise ValueError("target_signals must include 'embeddings'")
    if embeddings.shape[0] != len(target_pos):
        raise ValueError(
            f"embeddings has {embeddings.shape[0]} rows "
            f"but there are {len(target_pos)} label=1 rows in df"
        )

    n_train = int(round(len(target_pos) * ratio))
    if n_train < 1 or n_train >= len(target_pos):
        raise ValueError(
            f"split produced empty train or held-out set; "
            f"ratio={ratio}, n_target={len(target_pos)}"
        )

    uncertainty = compose_uncertainty(target_signals)
    in_target_idx = select_uncertain_diverse(embeddings, uncertainty, n_train, seed=seed)
    train_target_idx = np.sort(target_pos[in_target_idx])

    heldout_target_idx = np.array(
        sorted(set(target_pos.tolist()) - set(train_target_idx.tolist())),
        dtype=np.int64,
    )
    eval_idx = np.sort(np.concatenate([heldout_target_idx, forget_pos]))

    assert len(np.intersect1d(train_target_idx, heldout_target_idx)) == 0, \
        "train/heldout target overlap"
    assert len(np.intersect1d(train_target_idx, eval_idx)) == 0, \
        "train target leaks into eval set"
    assert set(df.iloc[train_target_idx]["quadrant_label"].unique().tolist()) == {1}, \
        "train_target_idx must contain only label=1 rows"
    assert set(df.iloc[eval_idx]["quadrant_label"].unique().tolist()) == {0, 1}, \
        "eval_idx must contain both classes"

    print(f"[data_loader]   informative split (uncertain+diverse): "
          f"train_target={len(train_target_idx)}  "
          f"heldout_target={len(heldout_target_idx)}  forget={len(forget_pos)}  "
          f"eval_total={len(eval_idx)} (positives={len(heldout_target_idx)})")
    print(f"[data_loader]   uncertainty range across all label=1: "
          f"min={uncertainty.min():.3f}, max={uncertainty.max():.3f}, "
          f"selected mean={uncertainty[in_target_idx].mean():.3f}, "
          f"unselected mean={np.delete(uncertainty, in_target_idx).mean():.3f}")

    return train_target_idx, eval_idx


def _normalize_text(t: str) -> str:
    return " ".join(t.lower().split())


def audit_split(df: pd.DataFrame, train_target_idx: np.ndarray, eval_idx: np.ndarray) -> dict:
    """Strict leakage audit. Returns a dict suitable for serializing to JSON.

    Asserts disjointness at FOUR levels — row index, (source, id), exact text,
    normalized text — and prints the result so any leakage shows up in logs."""
    print("\n" + "─" * 70)
    print("[audit] Strict leakage audit  (must show 0 overlap on every check)")
    print("─" * 70)

    train_df = df.iloc[train_target_idx]
    eval_df  = df.iloc[eval_idx]

    # 1) Row-position disjointness
    row_overlap = np.intersect1d(train_target_idx, eval_idx)
    print(f"[audit] row-position overlap       : {len(row_overlap)}  (must be 0)")
    assert len(row_overlap) == 0, f"row-position leakage: {row_overlap[:5]}..."

    # 2) (source, id) disjointness
    train_keys = set(zip(train_df["source"].tolist(), train_df["id"].tolist()))
    eval_keys  = set(zip(eval_df["source"].tolist(),  eval_df["id"].tolist()))
    key_overlap = train_keys & eval_keys
    print(f"[audit] (source,id) overlap        : {len(key_overlap)}  (must be 0)")
    assert len(key_overlap) == 0, f"(source,id) leakage: {list(key_overlap)[:3]}..."

    # 3) Exact text disjointness
    train_texts = set(train_df["text"].tolist())
    eval_texts  = set(eval_df["text"].tolist())
    text_overlap = train_texts & eval_texts
    print(f"[audit] exact-text overlap         : {len(text_overlap)}  (must be 0)")
    assert len(text_overlap) == 0, "exact-text leakage between train and eval"

    # 4) Normalized text (whitespace+case-insensitive) disjointness
    train_norm = set(_normalize_text(t) for t in train_df["text"])
    eval_norm  = set(_normalize_text(t) for t in eval_df["text"])
    norm_overlap = train_norm & eval_norm
    print(f"[audit] normalized-text overlap    : {len(norm_overlap)}  (must be 0)")
    assert len(norm_overlap) == 0, "normalized-text leakage between train and eval"

    # 5) Train must be strictly label=1
    train_label_set = set(train_df["quadrant_label"].unique().tolist())
    print(f"[audit] train_target label set     : {train_label_set}  (must be {{1}})")
    assert train_label_set == {1}, f"non-target rows in training set: {train_label_set}"

    # 6) Eval must contain both classes
    eval_label_set = set(eval_df["quadrant_label"].unique().tolist())
    print(f"[audit] eval label set             : {eval_label_set}  (must be {{0,1}})")
    assert eval_label_set == {0, 1}, f"eval set is degenerate: {eval_label_set}"

    # 7) No forget records ever in train
    forget_in_train = (train_df["quadrant_label"] == 0).sum()
    print(f"[audit] forget records in train    : {int(forget_in_train)}  (must be 0)")
    assert forget_in_train == 0

    # 8) Sample IDs for manual inspection
    sample_train = list(zip(train_df["source"].tolist(), train_df["id"].tolist()))[:3]
    sample_eval_heldout = list(zip(
        eval_df[eval_df["quadrant_label"] == 1]["source"].tolist(),
        eval_df[eval_df["quadrant_label"] == 1]["id"].tolist(),
    ))[:3]
    sample_eval_forget = list(zip(
        eval_df[eval_df["quadrant_label"] == 0]["source"].tolist(),
        eval_df[eval_df["quadrant_label"] == 0]["id"].tolist(),
    ))[:3]
    print(f"[audit] sample train (source,id)         : {sample_train}")
    print(f"[audit] sample held-out target (source,id): {sample_eval_heldout}")
    print(f"[audit] sample forget         (source,id): {sample_eval_forget}")
    print("─" * 70 + "\n")

    return {
        "row_overlap":            int(len(row_overlap)),
        "source_id_overlap":      int(len(key_overlap)),
        "exact_text_overlap":     int(len(text_overlap)),
        "normalized_text_overlap": int(len(norm_overlap)),
        "train_label_set":        sorted(int(x) for x in train_label_set),
        "eval_label_set":         sorted(int(x) for x in eval_label_set),
        "forget_in_train":        int(forget_in_train),
        "n_train_target":         int(len(train_df)),
        "n_heldout_target":       int((eval_df["quadrant_label"] == 1).sum()),
        "n_forget":               int((eval_df["quadrant_label"] == 0).sum()),
        "sample_train":           [list(t) for t in sample_train],
        "sample_eval_heldout":    [list(t) for t in sample_eval_heldout],
        "sample_eval_forget":     [list(t) for t in sample_eval_forget],
    }


