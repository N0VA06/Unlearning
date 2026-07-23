import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import DATA_QUAD, QUADRANTS, SPLIT_SEED, SPLIT_RATIO, LORA_RATIO, P_RATIO


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


def load_train_keys(checkpoint_dir: Path) -> set:
    """Read the exact (source, id) keys a LoRA adapter was trained on from its
    split_info.json. Lets a downstream split stay consistent with an already
    trained adapter (no retrain)."""
    info_path = Path(checkpoint_dir) / "split_info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"No split_info.json in {checkpoint_dir}")
    with info_path.open("r") as f:
        info = json.load(f)
    return set(tuple(k) for k in info["train_keys"])


def make_pu_split(
    df: pd.DataFrame,
    target_signals: dict = None,
    lora_ratio: float = LORA_RATIO,
    p_ratio: float = P_RATIO,
    seed: int = SPLIT_SEED,
    lora_train_keys: set = None,
):
    """Three-way target split for leakage-free Positive-Unlabeled learning.

    The target class (label==1) is partitioned into:
      * ``lora_train`` — rows used ONLY to fit the LoRA adapter (adaptation).
      * ``P``          — labeled positives / reference target, **disjoint** from
                         lora_train, so overlap scores on P are not inflated by
                         the adapter having memorised them.
      * ``heldout``    — the remaining target rows (hidden positives in U).
    All forget rows (label==0) go into the unlabeled pool U alongside heldout.

    Parameters
    ----------
    lora_train_keys : set of (source, id), optional
        If given (e.g. from an existing checkpoint's ``split_info.json`` via
        :func:`load_train_keys`), exactly those rows become ``lora_train`` so the
        split matches an already-trained adapter — no retraining required.
        Otherwise ``lora_train`` is chosen by the same uncertain+diverse active
        selection as :func:`make_informative_split` (needs ``target_signals``).

    Returns
    -------
    dict with df-positional index arrays: ``lora_train_idx``, ``p_idx``,
    ``heldout_target_idx``, ``forget_idx``, ``u_idx``.
    """
    labels = df["quadrant_label"].values.astype(int)
    target_pos = np.where(labels == 1)[0]
    forget_pos = np.where(labels == 0)[0]
    if len(target_pos) == 0 or len(forget_pos) == 0:
        raise ValueError("Both target (label=1) and forget (label=0) records are required.")

    n_target = len(target_pos)

    # 1) LoRA-train subset
    if lora_train_keys is not None:
        keys = list(zip(df["source"].tolist(), df["id"].tolist()))
        lora_train_idx = np.array(
            [i for i in target_pos if keys[i] in lora_train_keys], dtype=np.int64
        )
        if len(lora_train_idx) == 0:
            raise ValueError("None of lora_train_keys matched target rows in df.")
    else:
        if target_signals is None or "embeddings" not in target_signals:
            raise ValueError("target_signals with 'embeddings' required when "
                             "lora_train_keys is not provided.")
        from src.sample_select import compose_uncertainty, select_uncertain_diverse
        n_lora = max(1, int(round(n_target * lora_ratio)))
        uncertainty = compose_uncertainty(target_signals)
        in_target = select_uncertain_diverse(
            target_signals["embeddings"], uncertainty, n_lora, seed=seed
        )
        lora_train_idx = np.sort(target_pos[in_target])

    # 2) P (labeled positives / reference) — random, representative, disjoint
    remaining = np.array(sorted(set(target_pos.tolist()) - set(lora_train_idx.tolist())),
                         dtype=np.int64)
    n_p = max(1, int(round(n_target * p_ratio)))
    n_p = min(n_p, max(1, len(remaining) - 1))  # keep at least one held-out positive
    rng = np.random.default_rng(seed)
    p_idx = np.sort(rng.choice(remaining, size=n_p, replace=False))

    # 3) held-out target (hidden positives in U)
    heldout_target_idx = np.array(
        sorted(set(remaining.tolist()) - set(p_idx.tolist())), dtype=np.int64
    )
    forget_idx = np.sort(forget_pos)
    u_idx = np.sort(np.concatenate([heldout_target_idx, forget_idx]))

    # Disjointness guarantees
    assert len(np.intersect1d(lora_train_idx, p_idx)) == 0, "P leaks into lora_train"
    assert len(np.intersect1d(lora_train_idx, u_idx)) == 0, "lora_train leaks into U"
    assert len(np.intersect1d(p_idx, u_idx)) == 0, "P leaks into U"
    assert set(df.iloc[lora_train_idx]["quadrant_label"].unique().tolist()) == {1}
    assert set(df.iloc[p_idx]["quadrant_label"].unique().tolist()) == {1}

    print(f"[pu_split] lora_train={len(lora_train_idx)}  P(pos)={len(p_idx)}  "
          f"heldout_target={len(heldout_target_idx)}  forget={len(forget_idx)}  "
          f"U={len(u_idx)}")

    return {
        "lora_train_idx":     lora_train_idx,
        "p_idx":              p_idx,
        "heldout_target_idx": heldout_target_idx,
        "forget_idx":         forget_idx,
        "u_idx":              u_idx,
    }


