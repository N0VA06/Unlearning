import os
from pathlib import Path

ROOT        = Path(__file__).parent
DATA_QUAD   = ROOT / "data" / "quadrants"
CHECKPOINTS = ROOT / "checkpoints"
RESULTS     = ROOT / "results"
PLOTS       = ROOT / "plots"

# Allow pointing to a local snapshot dir (useful for offline / air-gapped runs).
BASE_MODEL   = os.environ.get("BASE_MODEL_OVERRIDE", "openai-community/gpt2")
MAX_SEQ_LEN  = 128
LORA_R       = 8
LORA_ALPHA   = 16
LORA_DROPOUT = 0.05
TRAIN_EPOCHS = 1
LR           = 2e-4
BATCH_SIZE   = 1
UMAP_SEED    = 42
SPLIT_SEED   = 42
SPLIT_RATIO  = 0.2
PROBE_BATCH  = 16
PROBE_FOLDS  = 5
PROBE_C      = 1.0
PROBE_POOL   = "mean"   # 'mean' | 'last' — how to pool token hidden states per text

# ── Overlap-estimator / PU-split knobs ──────────────────────────────────────
# The target class is split three ways to avoid leakage in the PU setup:
#   LoRA-train (adaptation only) : LORA_RATIO of target, active-selected
#   P (labeled positives / ref)  : P_RATIO   of target, disjoint from LoRA-train
#   held-out target (hidden pos) : the remainder → goes into U with all forget
LORA_RATIO   = SPLIT_RATIO   # keep back-compat with the existing 20% adaptation split
P_RATIO      = 0.2           # fraction of target reserved as labeled positives / reference

MINK_FRAC    = 0.2           # Min-K++: fraction of most-surprising tokens averaged
LIRA_N_SHADOWS   = 8         # number of shadow LoRA adapters for full LiRA
LIRA_SHADOW_FRAC = 0.5       # fraction of target each shadow LoRA trains on
LIRA_SEED        = 42

# ── Active nnPU (Positive-Unlabeled) knobs ──────────────────────────────────
# Classifier features are the low-dimensional overlap SCORES (honest: the raw
# 768-d LoRA embedding trivialises the task via dataset-identity, see README).
PU_SCORE_COLS    = ["ppl_ratio", "minkpp", "emb_cos", "lira_lite"]
PU_PRIOR         = "known"      # positive class prior in U: 'known' (from split) or a float
PU_ROUNDS        = 8            # active-learning acquisition rounds
PU_BUDGET        = 12           # oracle labels acquired per round
PU_EPOCHS        = 200          # nnPU training epochs per refit
PU_LR            = 1e-2         # AdamW learning rate for the nnPU head
PU_HIDDEN        = 0            # nnPU head width (0 ⇒ linear; >0 ⇒ 1-hidden MLP)
PU_MLP_HIDDEN    = 32           # width used by the 'pu_mlp' baseline
PU_WEIGHT_DECAY  = 1e-4
PU_RELIABLE_FRAC = 0.10         # fraction of lowest-prob U treated as reliable negatives
PU_NEG_LAMBDA    = 1.0          # weight on the supervised loss for confirmed negatives
PU_SEED          = 42
PU_SEEDS         = 5            # repeats for mean ± std in the method comparison

QUADRANTS = {
    "q1": {
        "name":  "gsm8k_vs_svamp",
        "dir":   "q1_math__gsm8k_vs_svamp",
        "label": "Math (Q1)",
    },
    "q2": {
        "name":  "humaneval_vs_mbpp",
        "dir":   "q2_code__humaneval_vs_mbpp",
        "label": "Code (Q2)",
    },
    "q3": {
        "name":  "squad_vs_nq",
        "dir":   "q3_qa__squad_vs_nq",
        "label": "QA (Q3)",
    },
    "q4": {
        "name":  "agnews_vs_reuters",
        "dir":   "q4_news__agnews_vs_reuters",
        "label": "News (Q4)",
    },
}

COLORS = {
    "target":   "#1D9E75",
    "forget":   "#D85A30",
    "combined": "#534AB7",
    "ppl":      "#BA7517",
    "emb":      "#185FA5",
}
