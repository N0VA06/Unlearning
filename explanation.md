# Concepts, Current Status, and Next Steps

A plain-language reference for everything in this repo: what each method is, why it
exists, what the current results actually prove, and how to structure the next
experiment.

---

## Table of contents

- [Part 1 — What this experiment actually is](#part-1--what-this-experiment-actually-is)
- [Part 2 — The concepts](#part-2--the-concepts)
  - [Positive-Unlabeled (PU) learning](#positive-unlabeled-pu-learning)
  - [nnPU — non-negative PU risk](#nnpu--non-negative-pu-risk)
  - [The class prior π](#the-class-prior-π)
  - [Active learning](#active-learning)
  - [Acquisition strategies](#acquisition-strategies)
  - [The baselines](#the-baselines)
  - [The features (overlap estimators)](#the-features-overlap-estimators)
  - [The 7-stage pipeline](#the-7-stage-pipeline)
  - [Metrics](#metrics)
  - [Leakage discipline](#leakage-discipline)
- [Part 3 — Honest assessment of current results](#part-3--honest-assessment-of-current-results)
- [Part 4 — Can I move to unlearning now?](#part-4--can-i-move-to-unlearning-now)
- [Part 5 — How to structure the next experiment](#part-5--how-to-structure-the-next-experiment)
- [Part 6 — Decision guide](#part-6--decision-guide)

---

## Part 1 — What this experiment actually is

**One sentence:** given a model that was fine-tuned on some data, can we look at a
big pile of unlabeled text and figure out which rows the model was trained on?

The setup per quadrant:

| Set | Meaning | Size (Q1) |
|-----|---------|-----------|
| `lora_train` | target rows used to fit the LoRA adapter | 400 |
| `P` | labeled positives — known target rows, **not** used for LoRA training | 400 |
| `U_pool` | unlabeled pool we may query an oracle about | 1600 |
| `U_test` | held-out unlabeled rows, **evaluation only** | 1600 |

`U_pool` and `U_test` each contain a mix of hidden target rows (`label=1`) and forget
rows (`label=0`). The job is to score every `U_test` row by "how likely is this
target?" and measure AUROC.

**Important framing note.** The LoRA adapter is trained on **target rows only**. The
"forget" rows are never trained on — they are distractors from a related distribution.
So this is **membership / provenance identification**, not unlearning. This matters a
lot for Part 4.

---

## Part 2 — The concepts

### Positive-Unlabeled (PU) learning

**Problem.** Normal binary classification needs positives *and* negatives. Here you
have:

- a small set of **confirmed positives** `P` (rows you know are target),
- a big **unlabeled** set `U` that is a *mixture* of positives and negatives.

You never get a clean negative set. Treating all of `U` as negative is wrong — a
sizeable fraction of `U` really is positive.

**Analogy.** You have 400 confirmed photos of your dog and 3200 unlabeled photos.
Some unlabeled photos are your dog; most are other dogs. You cannot just say
"unlabeled = not my dog."

**Why it matters here.** In a real unlearning audit, a data owner can point at a few
documents they know are theirs. They cannot enumerate every document that *isn't*.
PU is the natural formulation.

---

### nnPU — non-negative PU risk

The classical PU trick (uPU) rewrites the negative risk using the positive set:

```
R_neg  ≈  E_U[ℓ(g(x), −)]  −  π · E_P[ℓ(g(x), −)]
```

Meaning: "risk over unlabeled, minus the part contributed by the hidden positives."

**The problem.** With a flexible model, that estimate can go **negative**, which is
impossible for a real risk. The model exploits it and overfits badly.

**nnPU fix (Kiryo et al., 2017).** If the empirical negative risk drops below zero,
stop descending on it and instead push it back up:

```python
if neg_risk >= -beta:
    return pos_risk + neg_risk        # normal case
return gamma * (-neg_risk)            # non-negative correction
```

In this repo: `src/active_pu.py` → `_nnpu_risk()`, with a sigmoid surrogate loss.

The scorer `PUNet` is:
- **linear** when `PU_HIDDEN = 0` (default for the active loop),
- a **1-hidden-layer MLP** when `PU_HIDDEN > 0` (`PU_MLP_HIDDEN = 32`, used by
  `pu_static_mlp`).

---

### The class prior π

π = fraction of the unlabeled pool that is actually positive.

nnPU **requires** π. Get it wrong and performance degrades:

- π too high → model calls almost everything positive
- π too low → model is over-conservative

In this repo `PU_PRIOR = "known"` makes `_estimate_prior()` recompute π every round
from the number of hidden positives remaining in the pool. Setting `PU_PRIOR` to a
float instead pins it to a fixed value.

> ⚠️ **Caveat worth stating in any writeup:** with `PU_PRIOR = "known"` the prior is
> read from ground-truth labels. That is an *oracle prior*. It is not train/test
> leakage (no `U_test` label reaches the model), but a real deployment would have to
> estimate π (KM2, TIcE, EM) or show robustness to getting it wrong. The float option
> makes that sweep a one-line change.

---

### Active learning

Instead of labeling data randomly, **pick which examples to label** so each label
teaches the model as much as possible.

The loop:

```
1. Train model on what you have
2. Score the unlabeled pool
3. Pick the top-B most "useful" rows  ← acquisition strategy
4. Ask the oracle for their true labels
5. Remove them from the pool, add to the labeled set
6. Repeat
```

In this repo: `PU_ROUNDS = 10` rounds, budget = `min(PU_BUDGET, 0.06·|U_pool|)`
labels per round. The model is **retrained from scratch** each round.

**Why it matters.** Labels are expensive. If 40 well-chosen labels get you the same
accuracy as 400 random ones, that is a real result.

---

### Acquisition strategies

All six live in `acquire()` in `src/active_pu.py`.

#### 1. `random`
Pick B rows uniformly at random. The control condition. Surprisingly strong — if a
fancy strategy cannot beat random, it is not adding value.

#### 2. `uncertainty`
Pick rows where the model is least confident, i.e. probability closest to 0.5:

```python
score = 1.0 - abs(2.0 * prob - 1.0)     # maximised at p = 0.5
```

**Intuition.** Label the points sitting right on the decision boundary; they should
pin the boundary down fastest.

**Failure mode.** All B picks can come from the *same* ambiguous cluster — redundant
labels. And on a noisy overlap region it can pull in mislabel-prone junk. You saw
exactly this earlier on Q4.

#### 3. `diversity` (k-center greedy / coreset)
Ignores the model entirely. Repeatedly pick the candidate **farthest** (cosine
distance) from everything already selected or labeled:

```python
dist = (1.0 - cand @ e[labeled_idx].T).min(axis=1)
for _ in range(budget):
    j = argmax(dist)
    picks.append(j)
    dist = minimum(dist, 1.0 - cand @ cand[j])   # update coverage
```

**Intuition.** Cover the input space evenly, like placing cell towers so no house is
far from one.

**Note.** This is the only place the raw 768-d LoRA embedding is used. The classifier
itself never sees it (see [features](#the-features-overlap-estimators)).

#### 4. `badge` (Batch Active learning by Diverse Gradient Embeddings)
Combines uncertainty *and* diversity in one shot.

1. Compute each candidate's **gradient embedding** under its pseudo-label. For a
   sigmoid scorer the last-layer gradient is `(p − ŷ)·x`.
2. Run **k-means++ seeding** on those gradient vectors.

```python
p     = pu_proba(model, feat[cand_idx])
yhat  = (p >= 0.5)
G     = (p - yhat)[:, None] * feat[cand_idx]     # gradient embedding
seeds = kmeanspp_seed(G, budget, rng)
```

**Why it works.** Gradient *magnitude* encodes uncertainty (confident points have
near-zero gradient), and gradient *direction* encodes what kind of update the point
would cause. k-means++ picks points that are both high-magnitude and
directionally spread — uncertain *and* diverse, no hand-tuned trade-off.

#### 5. `committee` (Query-by-Committee / QBC)
Treat each of the four overlap scores as a separate "expert." Rank-normalise each
column to [0,1], then take the **standard deviation across experts**:

```python
ranks = argsort(argsort(score_mat, axis=0), axis=0) / (n - 1)
score = ranks.std(axis=1)          # high = the estimators disagree
```

**Intuition.** If PPL-ratio says "definitely target" and Min-K++ says "definitely
forget," that row is genuinely informative. Consensus rows teach you nothing.

#### 6. `uxd` (Uncertainty × Diversity)
The naive product of the two signals:

```python
score = uncertainty(prob_u) * diversity(emb, unl_idx, labeled_idx)
```

Simpler than BADGE; usually a bit worse because a straight product has no principled
scaling between the two terms.

**Summary table**

| Strategy | Uses model? | Uses embeddings? | Batch-diverse? |
|----------|:-----------:|:----------------:|:--------------:|
| random | no | no | trivially |
| uncertainty | yes | no | **no** |
| diversity | no | yes | yes |
| badge | yes | no (uses features) | yes |
| committee | no | no (uses score matrix) | no |
| uxd | yes | yes | partly |

---

### The baselines

You need these to know whether Active PU is doing anything.

| Method | What it is | Why it's here |
|--------|------------|---------------|
| `raw_<score>` | Threshold one overlap score directly. No training. | Floor. If a single number beats your classifier, the classifier is pointless. |
| `naive_pn` | Logistic regression, treating **all** of `U_pool` as negative | The classic "wrong but strong" PU baseline |
| `pu_static_linear` | nnPU, linear head, **zero** acquired labels (round 0 only) | Isolates the value of acquisition |
| `pu_static_mlp` | Same, MLP head | Isolates the value of model capacity |
| `active_<strategy>` | nnPU + 10 rounds of acquisition + Stage-7 head | The proposed method |
| `supervised_skyline` | Logistic trained with **every** `U_pool` label revealed | **Ceiling.** Nothing should beat it. |

**The skyline is the most important row in the table.** It tells you how good these
features can *possibly* be. If skyline is 0.6, no clever classifier will hit 0.9 —
your features or your data pair are the bottleneck.

---

### The features (overlap estimators)

The classifier input is a 4-dimensional vector per row:

| Feature | Signal | Intuition |
|---------|--------|-----------|
| `ppl_ratio` | log-perplexity under base GPT-2 minus under LoRA GPT-2 | Trained-on text gets *easier* for the adapted model |
| `minkpp` | Min-K%++ over the most surprising 20% of tokens, calibrated by per-token μ/σ | Memorisation shows up in the tail, not the mean |
| `emb_cos` | cosine of the row's LoRA embedding to the **P-centroid** | Trained-on text drifts toward the target cluster |
| `lira_lite` | one-sided LiRA: benefit vs an ensemble of shadow LoRAs that never saw this row | Calibrates "is this benefit unusual?" against a null distribution |

**Why not just use the 768-d embedding?** Because it cheats. The raw embedding
separates *datasets* (formatting, vocabulary, style), not *membership*. On Q2 it gave
AUROC ≈ 1.0 with zero active-learning signal — a meaningless result. The four score
features leave a genuinely hard boundary. The embedding is used **only** for
diversity/coreset acquisition.

---

### The 7-stage pipeline

| Stage | Name | What happens |
|-------|------|--------------|
| 1–2 | Setup + scoring | Train LoRA on `lora_train`; compute the 4 overlap scores for every `P ∪ U` row |
| 3 | nnPU | Fit `PUNet` on `P` (positive) + `U_pool` (unlabeled) with prior π |
| 4 | Acquisition | Score `U_pool`, query top-B, oracle-label, move to labeled set |
| 5 | Reliable negatives | Take the lowest-probability 10% of the remaining pool as soft negatives |
| 6 | Hard negatives | Acquired forget rows that scored *high* — the informative boundary cases |
| 7 | Final classifier | Calibrated logistic (+ isotonic) on labeled positives + confirmed + reliable negatives |

**Reporting subtlety that matters.** `pu_static_*` reports the **round-0 nnPU score**,
while `active_*` reports the **Stage-7 calibrated head**. That is not always
apples-to-apples — Stage-7 helps when you have real acquired negatives and can *hurt*
when you don't. Both numbers are now written to the JSON (`last_nnpu` and
`stage7_final`) precisely so this is visible.

---

### Metrics

| Metric | Reads as | When to trust it |
|--------|----------|------------------|
| **AUROC** | probability a random target ranks above a random forget | Primary. Threshold-free. |
| **AUPRC** | area under precision-recall | Better than AUROC when classes are imbalanced |
| **F1 / Precision / Recall** | at a fixed 0.5 threshold | Only meaningful once AUROC is decent and the model is calibrated |
| **FPR@95TPR** | false-positive rate when you catch 95% of targets | The deployment-relevant number: "to find 95% of my data, how much junk do I flag?" |
| **ECE** | expected calibration error | Are the probabilities honest? Stage-7 isotonic targets this |

**Rule of thumb.** With AUROC ≈ 0.6, F1 looks terrible no matter what — don't tune the
threshold, fix the ranking first.

---

### Leakage discipline

Audited and currently clean:

| Guarantee | Enforced where |
|-----------|----------------|
| `lora_train ∩ P = ∅`, `lora_train ∩ U = ∅`, `P ∩ U = ∅` | `make_pu_split()` asserts |
| Row / (source,id) / exact-text / normalised-text disjointness | `audit_split()` |
| `U ∩ shadow_train = ∅` | `run_overlap_scorers.py` raises `RuntimeError` |
| `emb_cos` centroid built from `P` only | `compute_embedding_scores()` asserts |
| Metrics computed on `U_test` only | `run_active_pu()` |
| Acquisition draws from `U_pool` only; picks are removed | `unlabeled.discard(j)` + assertions |
| Feature z-scoring uses train rows only, never `U_test` | `_standardize_train()` |

Verify any time with:

```bash
python experiments/run_active_pu.py --audit --quadrants q2
```

**Known non-leakage caveats:**
1. **Oracle prior π** — computed from true labels (see above).
2. **Isotonic calibration** is fit on the training scores, so `ECE` is optimistic;
   AUROC on `U_test` is unaffected.
3. The **oracle labeler** in the active loop is by design — those rows never enter
   `U_test`.

---

## Part 3 — Honest assessment of current results

Current AUROC on held-out `U_test`, 5 seeds:

| Method | Q1 Math | Q2 Code | Q3 QA | Q4 News |
|--------|:-------:|:-------:|:-----:|:-------:|
| best raw detector | 0.921 | 0.767 | 0.934 | 0.981 |
| `naive_pn` | 0.922 | 0.795 | 0.926 | 0.974 |
| `pu_static_mlp` | **0.924** | **0.816** | **0.936** | **0.981** |
| best `active_*` | 0.924 | 0.808 | 0.933 | 0.981 |
| `supervised_skyline` | 0.926 | 0.805 | 0.937 | 0.982 |

### What is genuinely good

- **The pipeline is correct and leak-free.** Splits, shadows, standardisation,
  evaluation — all audited.
- **Data scale is adequate**: 3.6k–6k rows per quadrant, 1.6k–2.7k held-out test
  rows, 5 seeds. Standard deviations are ≤ 0.01. These are not small-N artifacts.
- **The features work.** `ppl_ratio` and `lira_lite` alone hit 0.92–0.98 on
  Q1/Q3/Q4. Membership signal is clearly present and measurable.
- **Reproducible and well-instrumented**: dual metrics, three init modes,
  label-efficiency curves, per-quadrant plots.

### What is a problem for the scientific claim

**There is no headroom left.** On Q1, Q3, and Q4:

```
static nnPU (0 acquired labels)  ≈  active PU (400 labels)  ≈  skyline (all labels)
        0.924                            0.924                      0.926
```

The gap between "zero labels" and "every label" is **0.002 AUROC**. Active learning
cannot demonstrate value in that regime — there is nothing to learn. This is a
textbook **ceiling effect**.

Compare to the earlier hard configuration, where static was 0.72 and active reached
0.81 — *that* was a real active-learning result, just on a pool too small to trust.

So the current state is:

| | Earlier | Now |
|--|---------|-----|
| Data size | too small (270–1600) | good (3.6k–6k) |
| Task difficulty | too hard (skyline 0.59) or too easy | **too easy** (skyline 0.93+) |
| Active PU demonstrable? | noisy | **no headroom** |

You have solved the engineering. You have not yet found the operating regime where
the method earns its keep.

**Q2 is the exception** — skyline 0.805, static MLP 0.816, active 0.808. Still tight,
but it is the one quadrant that is not saturated.

---

## Part 4 — Can I move to unlearning now?

Short answer: **not directly, and here is the structural reason.**

### The blocker

Look at what the LoRA adapter is trained on:

```python
assert (train_df["quadrant_label"] == 1).all(), "Training data contains non-target records"
```

The model **only ever trains on target rows**. The "forget" rows are never in any
training set — they are a related-but-unseen distractor distribution.

**Therefore there is nothing to unlearn.** You cannot remove knowledge the model
never acquired. What you have built is the *detector* that would tell an auditor
which rows a model memorised — a genuine and useful component, but a precursor to
unlearning, not unlearning itself.

### What unlearning actually requires

| Step | Current repo | Needed for unlearning |
|------|--------------|-----------------------|
| Train | LoRA on **target only** | Train on `D_retain ∪ D_forget` — the model must *know* the forget set |
| Identify | Active PU finds target rows | Same, but now it finds the rows to remove |
| Unlearn | *does not exist* | Gradient ascent / NPO / SimNPO / task-vector negation on `D_forget` |
| Evaluate | AUROC on `U_test` | Forget quality **and** retain utility **and** MIA resistance |

### On "learning and unlearning at the same time in the same space"

This is a real and interesting research direction — joint optimisation of

```
L = L_retain(θ)  −  λ · L_forget(θ)      (roughly; naive ascent diverges)
```

with modern variants (NPO, SimNPO, RMU, task arithmetic) handling the instability of
the naive negative term. Doing it in one LoRA subspace is a legitimate framing.

But it presupposes a model that has learned both. **You need to restructure the
training setup first.** Jumping to joint learn/unlearn on the current checkpoints
would be unlearning data the model never saw.

---

## Part 5 — How to structure the next experiment

Two viable paths. They are not mutually exclusive; **Path A is a ~2 day detour that
makes Path B's story much stronger.**

---

### Path A — Rescue the Active-PU claim (recommended first)

**Goal:** find the regime where active labeling measurably beats static.

**Why:** right now your headline method is indistinguishable from a baseline that
uses zero labels. One controlled experiment fixes that.

#### A1. Sweep task difficulty, keep size fixed

Build 3 difficulty tiers per domain at constant ~4k rows:

| Tier | Q1 Math forget | Q2 Code forget | Expected skyline |
|------|----------------|----------------|------------------|
| easy | hendrycks_math (current) | CSN code-only (current) | 0.93+ |
| medium | ASDiv + SVAMP mix | MBPP + APPS | 0.80–0.88 |
| hard | SVAMP only, same template | MBPP only, same format | 0.65–0.78 |

Then plot **`active − static` AUROC gap vs skyline**. The expected shape is an
inverted U: no gain when trivial, no gain when hopeless, **maximum gain in the
middle**. That plot is the paper figure.

#### A2. Shrink the label budget

Currently `P = 400` labeled positives and 400 acquired. That is generous. Try:

```python
P_RATIO   = 0.05      # 100 positives instead of 400
PU_BUDGET = 10        # 10 labels/round instead of 40
PU_ROUNDS = 10
```

Active learning is about **label scarcity**. Give it scarcity.

#### A3. Fix the reporting asymmetry

Make static and active use the *same* head. Either:
- report `last_nnpu` for both, or
- run Stage 7 on the static model too.

Right now static gets round-0 nnPU and active gets Stage-7 — a confound.

#### A4. Match capacity
`pu_static_mlp` uses a 32-unit MLP; the active loop uses a linear head. Run
`active_*` with `PU_HIDDEN = 32` so the comparison is fair.

#### A5. Drop the oracle prior
`PU_PRIOR` already accepts a float, so a sensitivity curve costs almost nothing:

```python
PU_PRIOR = 0.30    # then 0.375 (true), 0.45, 0.50 — plot AUROC vs assumed π
```

Report that curve, or swap in a label-free estimator (KM2 / TIcE / EM). This closes
the one legitimate methodological criticism of the current setup.

**Deliverable:** a figure showing active PU beats static by a clear margin in the
medium-difficulty, low-budget regime, with a label-free prior.

---

### Path B — Move to actual unlearning

**Goal:** use the identifier to drive removal, then measure what removal cost you.

#### B1. Restructure the data

```
D_train = D_retain ∪ D_forget      ← LoRA trains on BOTH
D_holdout                          ← never trained on, for utility eval
```

Concretely: keep the quadrant pairs, but train the adapter on target **and** a
designated forget subset. Suggested split:

| Set | Role | Size |
|-----|------|------|
| `D_retain` | stays in the model | ~2000 |
| `D_forget` | must be removed | ~500 |
| `D_holdout` | clean utility probe | ~1000 |
| `P` | known-forget seeds for the auditor | ~50 |

#### B2. Identify with the existing pipeline

Run Active PU with `P` = a handful of known forget rows. Output: a ranked list of
suspected `D_forget` members. **This is where your current work plugs in.** Report
precision@k — because unlearning the wrong rows is the expensive failure mode.

#### B3. Unlearn

Compare at least three methods so the result is not a single-point claim:

| Method | One-line description |
|--------|----------------------|
| **Retrain-from-scratch** | Gold standard. Train fresh on `D_retain` only. Expensive but it is the ground truth. |
| **Gradient ascent (GA)** | Maximise loss on `D_forget`. Simple, unstable, degrades utility. |
| **GA + retain** | Ascent on forget, descent on retain. The standard practical baseline. |
| **NPO** | Preference-optimisation framing; far more stable than raw GA. |
| **Task-vector negation** | `θ_unlearned = θ − α·(θ_forget-tuned − θ_base)` |

#### B4. Evaluate on three axes simultaneously

This is the part people get wrong — reporting only one axis is meaningless.

| Axis | Metric | Target |
|------|--------|--------|
| **Forget quality** | MIA AUROC on `D_forget` (reuse this repo's scorers!) | → 0.5 (indistinguishable from never-seen) |
| **Retain utility** | perplexity / accuracy on `D_retain` and `D_holdout` | unchanged vs original |
| **Efficiency** | wall-clock vs retrain-from-scratch | ≪ retrain |

Plot the **forget-quality vs retain-utility trade-off curve** by sweeping the
unlearning strength. That curve is the deliverable — a single point is not.

**Nice synergy:** your four overlap estimators become the MIA evaluation suite. That
is a genuinely clean story — *the same signal that finds the data verifies its
removal.*

#### B5. Only then: joint learn + unlearn

Once B1–B4 work, the simultaneous version is a natural extension:

```
L = L_new-task(θ)  +  λ_r · L_retain(θ)  −  λ_f · L_forget(θ)
```

Research questions worth asking:
- Does learning a new task while unlearning **interfere**, or does the new gradient
  act as a regulariser that *protects* retain utility?
- Do the forget and learn directions occupy the same LoRA subspace? (Measure the
  principal-angle overlap between the two updates — that is the "same space"
  question, made concrete and measurable.)
- Is joint cheaper than sequential unlearn-then-learn, at equal quality?

That subspace-overlap measurement is, in my view, the most publishable idea in this
whole direction.

---

## Part 6 — Decision guide

| Your goal | Do this |
|-----------|---------|
| Publish the Active-PU work as its own contribution | **Path A.** You need a headroom regime; right now there is none. |
| Active PU is just a component; unlearning is the thesis | **Path B**, and reframe A's results as "identification is solved on easy pairs." |
| Limited time, want one strong result | **Path A steps A1–A3** (≈2 days) → then Path B. The difficulty-sweep figure justifies why identification is non-trivial. |
| Want the most novel angle | **B5** — subspace overlap between learn and unlearn updates. |

### My recommendation

1. **Do not throw away the current results.** They prove the pipeline is correct,
   leak-free, and that membership signal exists at scale. That is a real finding —
   just report it honestly as *"on well-separated pairs, identification saturates and
   active labeling is unnecessary."* Negative/ceiling results stated clearly are more
   credible than overclaimed ones.

2. **Run Path A1 + A2** (difficulty sweep at low label budget). Cheap, reuses
   everything, and produces the inverted-U figure that makes the active-learning
   claim defensible.

3. **Then go to Path B.** Restructure so the model actually trains on the forget set.
   Without that, "unlearning" is not well-defined here.

4. **Save joint learn+unlearn (B5) for last** — it is the most interesting question,
   but it only makes sense once single-direction unlearning is measurable.

### Concrete next command

If you take the recommendation, the first concrete step is a medium-difficulty,
low-budget rebuild:

```bash
# 1. medium-difficulty pairs (SVAMP-only Q1, MBPP-only Q2) at ~4k rows
DATA_SCALE=large python download.py

# 2. scarce labels:  P_RATIO=0.05, PU_BUDGET=10 in config.py

# 3. retrain + re-evaluate
rm -rf checkpoints/q*_ checkpoints/shadows results/*_pu_features.npz
python experiments/run_identification.py
python experiments/run_overlap_scorers.py --with-lira --train-shadows
python experiments/run_active_pu.py --rebuild-cache
```

Then check one number: **`supervised_skyline − pu_static_mlp`**. If that gap is
> 0.05, you have headroom and active learning has a chance to show value. If it is
still ~0.00, make the pair harder and repeat.

---

## Appendix — Glossary

| Term | Meaning |
|------|---------|
| **PU learning** | Learning from Positive and Unlabeled data only |
| **nnPU** | Non-negative PU risk estimator; prevents negative-risk overfitting |
| **π (prior)** | Fraction of the unlabeled pool that is truly positive |
| **Acquisition function** | Rule for choosing which unlabeled rows to label next |
| **BADGE** | Batch AL via Diverse Gradient Embeddings — k-means++ on gradient embeddings |
| **QBC** | Query-by-Committee — query where independent models disagree |
| **k-center / coreset** | Diversity acquisition; maximise minimum distance coverage |
| **Skyline** | Fully-supervised upper bound (all pool labels revealed) |
| **MIA** | Membership Inference Attack — did this row appear in training? |
| **LiRA** | Likelihood Ratio Attack; calibrates membership signal against shadow models |
| **Min-K%++** | MIA score over the k% most-surprising tokens, per-token calibrated |
| **Shadow model** | Reference model trained on a known subset, used to build a null distribution |
| **Reliable negatives** | Lowest-scoring unlabeled rows, treated as pseudo-negatives |
| **Hard negatives** | Acquired rows that scored high but turned out negative — boundary cases |
| **Ceiling effect** | Task so easy that all methods saturate and differences vanish |
| **Retain / forget set** | Data that must be kept / removed in an unlearning task |
| **NPO** | Negative Preference Optimisation — a stable alternative to gradient ascent |
