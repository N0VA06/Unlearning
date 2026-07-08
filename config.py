from pathlib import Path

ROOT        = Path(__file__).parent
DATA_QUAD   = ROOT / "data" / "quadrants"
CHECKPOINTS = ROOT / "checkpoints"
RESULTS     = ROOT / "results"
PLOTS       = ROOT / "plots"

BASE_MODEL   = "openai-community/gpt2"
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
