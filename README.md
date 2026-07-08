# Model Unlearning — Data Identification Experiment

Can we tell **target** records from **forget** records after training a small LoRA adapter on a subset of the target class?

This repo implements a diagnostic pipeline that tests two scoring signals — a **perplexity ratio** (base GPT-2 vs LoRA-adapted GPT-2) and an **embedding similarity** score (LoRA hidden states) — across four **same-domain** dataset pairs. LoRA is trained on an actively selected 20% of target rows; all metrics are computed on a held-out eval set (remaining target + all forget). Strict leakage audits run before every training step.

---

## Table of contents

- [Experiment design](#experiment-design)
- [Quick start](#quick-start)
- [Pipeline overview](#pipeline-overview)
- [Scoring methods](#scoring-methods)
- [Code walkthrough](#code-walkthrough)
- [Example results (Q1 — Math)](#example-results-q1--math)
- [Cross-quadrant summary](#cross-quadrant-summary)
- [Configuration](#configuration)
- [Project layout](#project-layout)
- [Leakage controls](#leakage-controls)
- [Length matching (`download.py`)](#length-matching-downloadpy)
- [Caveats](#caveats)

---

## Experiment design

Each quadrant pairs two datasets from the **same broad domain** so that text length or surface format cannot trivially separate classes. Every combined JSON record has:

```json
{ "id": "...", "source": "gsm8k", "text": "...", "quadrant_label": 1, "role": "target" }
```

`quadrant_label = 1` → target (LoRA training source), `0` → forget (never trained on).

| Q | Domain | Target (`label=1`) | Forget (`label=0`) |
|---|--------|--------------------|--------------------|
| Q1 | Math word problems | GSM8K | SVAMP |
| Q2 | Python code | HumanEval | MBPP |
| Q3 | Extractive QA | SQuAD | NQ-open |
| Q4 | Short news | AG News | XSum summaries |

**Train / eval split**

- **Train (20% of target):** chosen by active learning (uncertain + diverse), not random.
- **Eval:** all forget rows + held-out target rows (80% of target).
- **Positive class for metrics:** held-out target (`label=1`) vs forget (`label=0`).

---

## Quick start

```bash
pip install -r requirements.txt
python download.py                       # one-time: build data/raw/ and data/quadrants/
python experiments/run_identification.py # train LoRA, score, metrics, plots (per quadrant)
```

Requires Python 3.10+, PyTorch, and optionally CUDA. Checkpoints skip re-training if `checkpoints/q{n}_*_r20_s42_e1/` already exists.

To change eval size for Q1/Q2, edit `download.py` → `python download.py` → remove affected `checkpoints/q{n}_*` → `python experiments/run_identification.py`.

---

## Pipeline overview

```
data/quadrants/q{n}/*_combined.json
        │
        ▼
  load_quadrant()                    ← src/data_loader.py
        │
        ▼
  gather_selection_inputs()          ← src/sample_select.py (base GPT-2 only)
        │   embeddings, loss, entropy, grad_norm
        ▼
  make_informative_split()           ← k-means diversity + max-uncertainty per cluster
        │
        ▼
  audit_split()                      ← zero overlap on idx, (source,id), text
        │
        ▼
  train_lora()                       ← src/lora_trainer.py (target rows only)
        │
        ├──────────────────────────────────┐
        ▼                                  ▼
  compute_ppl_ratio_scores()      compute_embedding_scores()
        │                                  │
        └──────────┬───────────────────────┘
                   ▼
        compute_combined_scores()    ← 0.4·PPL + 0.6·emb (min-max)
                   │
        ┌──────────┼──────────┐
        ▼          ▼          ▼
  compute_all_metrics()   plot_all()   run_layerwise_drift()
        │                                  │
        ▼                                  ▼
  results/*.json              plots/{qid}/*.png
```

Driver: [`experiments/run_identification.py`](experiments/run_identification.py) loops `q1`…`q4`, saves JSON artifacts, prints a metrics table and an observations block at the end.

---

## Scoring methods

| Method | Formula / definition | Higher score means… |
|--------|----------------------|---------------------|
| **PPL ratio** | `log(PPL_base + ε) − log(PPL_lora + ε)` | LoRA fits the row better than base |
| **Embedding** | `cos(h_lora(x), centroid)` where `h` is mean-pooled last-layer LoRA hidden state; centroid = normalized mean of train-subset embeddings | Closer to the LoRA train distribution |
| **Combined** | `0.4 · minmax(PPL) + 0.6 · minmax(emb)` on the eval set | Better trade-off of both signals |

Implementation: [`src/scorer.py`](src/scorer.py). Embedding uses **LoRA GPT-2 hidden states only** — no separate sentence-transformer — so embedding scores and layerwise drift read from the same representation stream.

**Metrics** ([`src/metrics.py`](src/metrics.py)): AUROC, average precision, best F1 (threshold sweep), Kolmogorov–Smirnov statistic, Cohen's d, KDE overlap coefficient, precision@K for K ∈ {10, 25, 50, 100}. Results written to `results/{qid}_{name}_metrics.json`.

---

## Code walkthrough

### `config.py`

Central paths (`DATA_QUAD`, `CHECKPOINTS`, `RESULTS`, `PLOTS`) and hyperparameters: `BASE_MODEL = openai-community/gpt2`, `MAX_SEQ_LEN = 128`, LoRA `r=8, α=16`, `SPLIT_RATIO = 0.2`, `SPLIT_SEED = 42`, `TRAIN_EPOCHS = 1`. The `QUADRANTS` dict maps each `qid` to its directory name and display label.

### `download.py`

Builds `data/raw/*.json` from Hugging Face datasets and assembles matched quadrants under `data/quadrants/`. For each pair:

1. Truncate text to per-pair caps (`TRUNC`: 200 / 400 / 150 / 200 chars).
2. Drop stubs below `LOWER` character bounds (`q1` 60, `q2` 80, `q3` 40, `q4` 80).
3. Restrict both classes to the **length overlap band** `[max(min_t, min_f), min(max_t, max_f)]`.
4. **Stratified bin sampling** — equal target/forget counts per narrow length bin (`N_BINS_PER`).
5. Print **length-only AUROC** as a sanity check (should be ≈ 0.5).

Q4 forget uses **XSum summaries** instead of Reuters-21578 (deprecated HF script); length is still matched.

### `src/data_loader.py`

- **`load_quadrant(qid)`** — loads `{name}_combined.json`, validates columns `id, source, text, quadrant_label, role`, prints class counts and mean text length.
- **`make_informative_split(df, target_signals)`** — maps active-learning indices back to DataFrame positions; eval = held-out target ∪ all forget.
- **`audit_split(df, train_idx, eval_idx)`** — asserts disjointness at four levels; returns a JSON-serializable receipt.

### `src/sample_select.py`

Active learning on **base GPT-2 only** (no LoRA yet):

- **Uncertainty:** mean of min-max normalized `loss`, `entropy`, `grad_norm` (per-token CE, next-token entropy, ‖∇θ LM-loss‖₂).
- **Diversity:** K-means on mean-pooled last-layer embeddings (`k = n_train` clusters).
- **Selection:** highest-uncertainty row per cluster (`select_uncertain_diverse`).

`gather_selection_inputs()` loads base model, extracts embeddings + signals, frees GPU memory.

### `src/lora_trainer.py`

- LoRA on GPT-2 `c_attn` modules via PEFT (`LoraConfig`).
- Trains **only** on rows with `quadrant_label == 1`; runtime assertion enforces this.
- Checkpoint dir encodes split: `checkpoints/q{n}_{name}_r20_s42_e1/` — prevents silent reuse under a different split.
- Writes `split_info.json` listing every `(source, id)` in the train set.
- Skips training if adapter already on disk.

### `src/hidden_states.py`

Shared extraction for selection, scoring, and drift: batched forward passes, `PROBE_POOL = "mean"` token pooling, `extract_last_layer` and `extract_all_layers` (embedding + all transformer blocks).

### `src/scorer.py`

- **`compute_ppl_ratio_scores`** — loads base (no adapter) and LoRA models separately; asserts base has no `peft_config`.
- **`compute_embedding_scores`** — asserts `(source, id)` disjointness between centroid source (train subset) and eval before extraction.
- **`compute_combined_scores`** — min-max both signals on valid (non-NaN) rows.

### `src/probe.py`

Layerwise **activation drift** on the eval set:

```
drift_l(x) = 1 − cos(h_base_l(x), h_lora_l(x))
```

Aggregates mean ± std per layer for target vs forget; `delta_l = target_drift_l − forget_drift_l`. Positive Δ at layer *l* means LoRA moved target representations more than forget at that depth.

### `src/plotting.py`

Seven plots per quadrant + `summary_all_quadrants.png` (4×3 ROC grid). Colors from `config.COLORS`: target green `#1D9E75`, forget orange `#D85A30`.

| Plot | File | What it shows |
|------|------|----------------|
| 1 | `plot1_score_distributions.png` | KDE per class per method; Youden threshold |
| 2 | `plot2_roc_curves.png` | ROC for PPL, embedding, combined |
| 3 | `plot3_pr_curves.png` | Precision–recall + class baseline |
| 4 | `plot4_2d_scatter.png` | Raw PPL vs embedding, colored by label |
| 5 | `plot5_umap.png` | UMAP of LoRA embeddings + convex hulls |
| 6 | `plot6_threshold_sweep.png` | P/R/F1 vs combined threshold |
| 7 | `plot7_layerwise_drift.png` | Per-layer drift curves + Δ(target−forget) |

### `experiments/run_identification.py`

End-to-end orchestration: for each quadrant, load → select → audit → train → score → metrics → plots → drift → aggregate summary tables and observations (best/worst quadrant, PPL vs embedding winner, near-random warnings, drift table, overlap matrix). Observations are **computed from results**, never hardcoded.

---

## Example results (Q1 — Math)

After a full run, Q1 (GSM8K vs SVAMP) produces diagnostics under `plots/q1_gsm8k_vs_svamp/`. Below are the plots from the latest run on **n = 247** eval rows (110 held-out target + 137 forget; 137 matched rows per class in the combined JSON).

**AUROC (latest run):** PPL 0.645 · Embedding 0.528 · Combined **0.678**

### Plot 1 — Score distributions

KDE of each scoring method by class. Green = target (held-out), orange = forget. Dashed line = Youden threshold.

![Q1 score distributions](plots/q1_gsm8k_vs_svamp/plot1_score_distributions.png)

With the larger eval set, embedding is near-random; PPL and combined still show modest separation. Loosen or tighten `N_BINS_PER` / `LOWER` in `download.py` to trade eval size vs length-AUROC.

### Plot 2 — ROC curves

![Q1 ROC curves](plots/q1_gsm8k_vs_svamp/plot2_roc_curves.png)

### Plot 3 — Precision–recall

![Q1 PR curves](plots/q1_gsm8k_vs_svamp/plot3_pr_curves.png)

### Plot 4 — PPL vs embedding scatter

Each point is one eval row; marginal rug/KDE on axes.

![Q1 2D scatter](plots/q1_gsm8k_vs_svamp/plot4_2d_scatter.png)

### Plot 5 — UMAP of LoRA embeddings

2D projection of mean-pooled LoRA last-layer states; convex hulls per class.

![Q1 UMAP](plots/q1_gsm8k_vs_svamp/plot5_umap.png)

### Plot 6 — Combined threshold sweep

Precision, recall, and F1 vs threshold on the min-max combined score.

![Q1 threshold sweep](plots/q1_gsm8k_vs_svamp/plot6_threshold_sweep.png)

### Plot 7 — Layerwise activation drift

Top: mean cosine drift (base vs LoRA) per layer for target and forget. Bottom: Δ(target − forget) — where LoRA differentiated classes.

![Q1 layerwise drift](plots/q1_gsm8k_vs_svamp/plot7_layerwise_drift.png)

Plots for Q2–Q4 follow the same naming under `plots/q2_humaneval_vs_mbpp/`, `plots/q3_squad_vs_nq/`, `plots/q4_agnews_vs_reuters/`.

---

## Cross-quadrant summary

ROC curves for all four quadrants × three methods (from `plot_summary`):

![Summary ROC grid](plots/summary_all_quadrants.png)

**AUROC by quadrant** (from `results/*_metrics.json` on the current checkpoints):

| Quadrant | PPL | Embedding | Combined |
|----------|-----|-----------|----------|
| Q1 Math | 0.645 | 0.528 | 0.678 |
| Q2 Code | 0.832 | 0.817 | **0.884** |
| Q3 QA | 0.807 | 0.811 | **0.875** |
| Q4 News | 0.786 | 0.621 | 0.741 |

Q2 and Q3 show the strongest combined AUROC. Q1 has the largest eval among the math/code pairs but weaker separation after relaxed length matching. Re-run `python download.py` then `experiments/run_identification.py` (delete `checkpoints/q{n}_*` to retrain LoRA) after changing data knobs.

---

## Configuration

| Knob | Default | Role |
|------|---------|------|
| `BASE_MODEL` | `openai-community/gpt2` | Backbone + LoRA host |
| `MAX_SEQ_LEN` | `128` | Token truncation |
| `LORA_R` / `LORA_ALPHA` | `8` / `16` | Adapter capacity |
| `TRAIN_EPOCHS` | `1` | Training passes over train subset |
| `LR` | `2e-4` | AdamW |
| `SPLIT_RATIO` | `0.2` | Fraction of target used for LoRA |
| `SPLIT_SEED` | `42` | K-means + selection reproducibility |
| `PROBE_BATCH` | `16` | Hidden-state batch size |
| `PROBE_POOL` | `"mean"` | Token pooling (`mean` or `last`) |

Edit [`config.py`](config.py) for model and split settings. **Eval size** is set in [`download.py`](download.py):

| Knob | Q1 (current) | Q2 (current) | Effect |
|------|--------------|--------------|--------|
| `N_BINS_PER` | 10 | 8 | Fewer bins → more rows per class (watch length-AUROC) |
| `LOWER` | 60 | 80 | Lower → keep shorter texts |
| `TRUNC` | 200 | 400 | Upper cap on characters |
| `TARGET_N` | 1000 | 500 | Max rows per class after matching |

Eval size ≈ `0.8 × n_target + n_forget` with matched `n_target ≈ n_forget`. Q2 is capped by HumanEval (164 problems) and overlap with MBPP.

---

## Project layout

```
modelunlearning/
├── config.py
├── download.py
├── requirements.txt
├── experiments/
│   └── run_identification.py      # main entry point
├── src/
│   ├── data_loader.py             # load, split, audit
│   ├── sample_select.py           # active-learning selection
│   ├── hidden_states.py           # pooled hidden-state extraction
│   ├── lora_trainer.py
│   ├── scorer.py
│   ├── metrics.py
│   ├── probe.py                   # layerwise drift
│   └── plotting.py
├── data/
│   ├── raw/                       # per-dataset JSON exports
│   └── quadrants/q{n}_*/           # *_combined.json per pair
├── checkpoints/q{n}_*_r20_s42_e1/ # LoRA adapter + split_info.json
├── results/                       # metrics, audit, drift JSONs
└── plots/                         # per-quadrant PNGs + summary grid
```

**JSON artifacts per quadrant**

| File | Contents |
|------|----------|
| `results/{qid}_{name}_metrics.json` | Full metric dict per method |
| `results/{qid}_{name}_audit.json` | Leakage audit receipts + sample IDs |
| `results/{qid}_{name}_drift.json` | Per-layer drift means, Δ, L2 backup |

---

## Leakage controls

Inflated AUROC from train/eval leakage is the main failure mode. Every run enforces:

1. **Deterministic split** — `SPLIT_SEED` fixes which target rows train vs held-out.
2. **`audit_split`** — zero overlap on row index, `(source, id)`, exact text, normalized text (printed before training).
3. **LoRA trains on `label=1` only** — enforced in `train_lora` and at the call site.
4. **PPL scores only the eval set** — no memorization of train texts in the metric.
5. **Embedding centroid from train subset only** — `(source, id)` disjoint from eval asserted in `compute_embedding_scores`.
6. **Checkpoint names encode split** — `r{ratio}_s{seed}_e{epochs}`; `split_info.json` lists trained keys.
7. **Base model has no adapter** — asserted before PPL scoring.

---

## Length matching (`download.py`)

Without controls, pairs like SST-2 vs MBPP made length alone near-perfect for classification. The current pipeline:

- Truncates to `TRUNC[qid]`, drops below `LOWER[qid]`.
- Keeps only the shared length band between classes.
- Samples equal counts per narrow bin inside that band.

Typical **length-only AUROC** after matching: ≈ 0.50 ± 0.02 (Q3/Q4); Q1/Q2 may reach ~0.55 when bins are coarser for larger eval sets. `download.py` prints the value for each quadrant at the end of a data build.

---

## Caveats

- **Q2 sample size** — HumanEval has only 164 problems; relaxed matching yields ~91 rows per class (~164 eval). Further gains need looser bins or a larger target corpus, not just `TARGET_N`.
- **Q1 trade-off** — Looser length matching increases eval size (e.g. 149 → 247) but can weaken AUROC and push embedding toward random.
- **Template residue** — Length is matched; dataset-specific formatting (answer markers, code style, QA spans) may still differ. Inspect `*_audit.json` sample rows if AUROC looks unexpected.
- **Membership framing** — LoRA is optimized for the target distribution. The 80/20 split tests generalization beyond the train subset, not cold membership on an unrelated domain.
- **Q4 forget style** — XSum summaries vs AG News snippets; length is controlled but forget text is abstractive summary prose.

---

## Dependencies

See [`requirements.txt`](requirements.txt): `torch`, `transformers`, `peft`, `scikit-learn`, `scipy`, `matplotlib`, `seaborn`, `umap-learn`, `pandas`, `numpy`, `tqdm`. Dataset download additionally uses `datasets` (installed when running `download.py`).
