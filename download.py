"""Build the four data quadrants used by the identification experiment.

Pairs are intentionally drawn from the SAME broad domain so that text length /
surface formatting cannot trivially separate target from forget:

    Q1  Math word problems     GSM8K (+test)  (target)   vs  SVAMP+ASDiv  (forget)
    Q2  Python code            HumanEval+APPS (target)   vs  MBPP (full)  (forget)
    Q3  Extractive QA          SQuAD          (target)   vs  NQ-open      (forget)
    Q4  News                   AG News        (target)   vs  XSum / DBpedia (forget)

For every pair we:
  1. Truncate each record's `text` to a per-pair character cap.
  2. Drop records shorter than a per-pair lower bound.
  3. Run length-quintile stratified sampling so target and forget classes have
     IDENTICAL length-bin counts in the combined record. This is the strongest
     guarantee we can give that `len(text)` alone cannot classify.
  4. Print the AUROC of `len(text)` as a sanity check — should be near 0.5.

Scale profiles (env ``DATA_SCALE``):
  * ``large`` (default) — bigger pools for Active-PU (≈1–3k per class where sources allow)
  * ``small`` — legacy compact sizes used in early prototypes
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
from datasets import load_dataset, concatenate_datasets
from sklearn.metrics import roc_auc_score

BASE  = Path("data")
RAW   = BASE / "raw"
QUADS = BASE / "quadrants"

# ── Scale profile ────────────────────────────────────────────────────────────
# ``large`` emphasises Active-PU label efficiency (bigger U_pool / U_test).
# ``small`` keeps the original compact quadrants for quick smoke tests.
DATA_SCALE = os.environ.get("DATA_SCALE", "large").strip().lower()
if DATA_SCALE not in {"large", "small"}:
    raise SystemExit(f"DATA_SCALE must be 'large' or 'small', got {DATA_SCALE!r}")

if DATA_SCALE == "large":
    TRUNC      = {"q1": 200, "q2": 400, "q3": 150, "q4": 200}
    LOWER      = {"q1":  50, "q2":  60, "q3":  40, "q4":  70}
    TARGET_N   = {"q1": 2000, "q2": 2000, "q3": 3000, "q4": 3000}
    # Many bins keep length-AUROC near 0.5; global subsample (not per-bin cap)
    # still allows ≥3000 combined for Q1/Q2.
    N_BINS_PER = {"q1": 15, "q2": 12, "q3": 20, "q4": 20}
    RAW_CAP    = {"q3": 30000, "q4": 30000}
    Q1_MIN_COMBINED = 3000
    Q2_MIN_COMBINED = 3000
else:
    TRUNC      = {"q1": 200, "q2": 400, "q3": 150, "q4": 200}
    LOWER      = {"q1":  60, "q2":  80, "q3":  40, "q4":  80}
    TARGET_N   = {"q1": 1000, "q2":  500, "q3": 1000, "q4": 1000}
    N_BINS_PER = {"q1": 10,   "q2":  8,   "q3": 30,   "q4": 30}
    RAW_CAP    = {"q3": 8000, "q4": 8000}
    Q1_MIN_COMBINED = 0
    Q2_MIN_COMBINED = 0

SEED = 42

QUAD_DIRS = {
    "q1": QUADS / "q1_math__gsm8k_vs_svamp",
    "q2": QUADS / "q2_code__humaneval_vs_mbpp",
    "q3": QUADS / "q3_qa__squad_vs_nq",
    "q4": QUADS / "q4_news__agnews_vs_reuters",
}

for d in [RAW, *QUAD_DIRS.values()]:
    d.mkdir(parents=True, exist_ok=True)

print(f"[download] DATA_SCALE={DATA_SCALE}  TARGET_N={TARGET_N}  N_BINS={N_BINS_PER}")


def save_json(records, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    print(f"    saved {len(records):>5} records  →  {path}")


def truncate(text: str, max_chars: int) -> str:
    return text.strip().replace("\r", "")[:max_chars]


def length_stratified_sample(rec_t, rec_f, n_bins, max_per_class=None, seed=SEED):
    """Match target/forget length distributions in two stages:

    1. Restrict both classes to the OVERLAP band of their length distributions
       (`[max(min_t, min_f), min(max_t, max_f)]`). Records outside the band have
       no counterpart and would force length-AUROC away from 0.5.
    2. Within the overlap band, bin lengths and keep the same number of target
       and forget records per bin. ``max_per_class`` is applied as a *global*
       subsample after bin matching (not a per-bin starve cap), so peaked
       length distributions can still reach large pools without collapsing to
       a single-bin length confound.
    """
    rng = np.random.default_rng(seed)
    if not rec_t or not rec_f:
        return [], []

    lt = np.array([len(r["text"]) for r in rec_t])
    lf = np.array([len(r["text"]) for r in rec_f])

    lo_band = int(max(lt.min(), lf.min()))
    hi_band = int(min(lt.max(), lf.max()))
    if lo_band >= hi_band:
        return [], []
    mask_t = (lt >= lo_band) & (lt <= hi_band)
    mask_f = (lf >= lo_band) & (lf <= hi_band)
    rec_t = [r for r, m in zip(rec_t, mask_t) if m]
    rec_f = [r for r, m in zip(rec_f, mask_f) if m]
    lt = lt[mask_t]
    lf = lf[mask_f]
    if len(lt) == 0 or len(lf) == 0:
        return [], []

    edges = np.linspace(lo_band - 1, hi_band + 1, n_bins + 1)
    out_t, out_f = [], []

    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        t_in = np.where((lt > lo) & (lt <= hi))[0]
        f_in = np.where((lf > lo) & (lf <= hi))[0]
        take = min(len(t_in), len(f_in))
        if take == 0:
            continue
        t_sel = rng.choice(t_in, size=take, replace=False)
        f_sel = rng.choice(f_in, size=take, replace=False)
        out_t.extend([rec_t[j] for j in t_sel])
        out_f.extend([rec_f[j] for j in f_sel])

    if max_per_class is not None and len(out_t) > max_per_class:
        pick = rng.choice(len(out_t), size=max_per_class, replace=False)
        out_t = [out_t[j] for j in pick]
        out_f = [out_f[j] for j in pick]
    return out_t, out_f


def length_auroc(rec_t, rec_f) -> float:
    if not rec_t or not rec_f:
        return float("nan")
    lens   = np.array([len(r["text"]) for r in rec_t] +
                      [len(r["text"]) for r in rec_f])
    labels = np.array([1] * len(rec_t) + [0] * len(rec_f))
    return float(roc_auc_score(labels, lens))


def make_quadrant(qid, target_recs, forget_recs, t_name, f_name, min_combined=0):
    out_dir = QUAD_DIRS[qid]
    # Canonical name (e.g. "gsm8k_vs_svamp") stays stable even if a fallback
    # forget source had to be used. data_loader.load_quadrant() reads this name.
    canon_name = out_dir.name.split("__", 1)[1]
    print(f"\n[{qid}] {t_name} vs {f_name}  (canonical filename: {canon_name}_combined.json)")
    print(f"  raw counts: target={len(target_recs)}  forget={len(forget_recs)}")

    target_recs = [r for r in target_recs if len(r["text"]) >= LOWER[qid]]
    forget_recs = [r for r in forget_recs if len(r["text"]) >= LOWER[qid]]
    print(f"  after lower-bound (≥{LOWER[qid]} chars): "
          f"target={len(target_recs)}  forget={len(forget_recs)}")

    n_bins = N_BINS_PER[qid]
    matched_t, matched_f = length_stratified_sample(
        target_recs, forget_recs,
        n_bins=n_bins,
        max_per_class=TARGET_N[qid],
    )
    # If still short of the size floor, retry with fewer bins (still stratified).
    if min_combined and (len(matched_t) + len(matched_f)) < min_combined and n_bins > 4:
        for retry_bins in (max(4, n_bins // 2), 4, 3):
            print(f"  matched {len(matched_t)+len(matched_f)} < {min_combined} — "
                  f"retrying with n_bins={retry_bins}")
            matched_t, matched_f = length_stratified_sample(
                target_recs, forget_recs,
                n_bins=retry_bins,
                max_per_class=TARGET_N[qid],
            )
            if (len(matched_t) + len(matched_f)) >= min_combined:
                break
    print(f"  after length-stratified sampling: "
          f"target={len(matched_t)}  forget={len(matched_f)}")

    if not matched_t or not matched_f:
        raise RuntimeError(f"[{qid}] empty class after matching — relax bounds")

    lt = np.array([len(r["text"]) for r in matched_t])
    lf = np.array([len(r["text"]) for r in matched_f])
    print(f"  text length (chars):")
    print(f"    target  : mean={lt.mean():6.1f}  median={int(np.median(lt)):4d}  "
          f"min={int(lt.min()):4d}  max={int(lt.max()):4d}")
    print(f"    forget  : mean={lf.mean():6.1f}  median={int(np.median(lf)):4d}  "
          f"min={int(lf.min()):4d}  max={int(lf.max()):4d}")

    auc = length_auroc(matched_t, matched_f)
    flag = " ✓ near random" if abs(auc - 0.5) < 0.05 else " ⚠ length-confound risk"
    print(f"  length-AUROC: {auc:.3f}{flag}")

    target_out = [{**r, "quadrant_label": 1, "role": "target"} for r in matched_t]
    forget_out = [{**r, "quadrant_label": 0, "role": "forget"} for r in matched_f]
    combined   = target_out + forget_out

    save_json(combined,   out_dir / f"{canon_name}_combined.json")
    save_json(target_out, out_dir / f"{t_name}_target.json")
    save_json(forget_out, out_dir / f"{f_name}_forget.json")
    return len(matched_t), len(matched_f)


def _clean(s):
    if s is None:
        return ""
    return str(s).replace("\n", " ").strip()


# ─────────────────────────────────────────────────────────────────────────────
# Q1 — GSM8K (target) vs SVAMP+ASDiv (forget)  -- math word problems
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Loading sources ===")
gsm_parts = []
for split in ("train", "test"):
    try:
        gsm_parts.append(load_dataset("openai/gsm8k", "main", split=split))
    except Exception as e:
        print(f"  gsm8k/{split} failed: {type(e).__name__}: {str(e)[:120]}")
gsm = concatenate_datasets(gsm_parts).shuffle(seed=SEED) if gsm_parts else None
if gsm is None:
    raise RuntimeError("Could not load GSM8K")
gsm_records = []
for i, row in enumerate(gsm):
    q = _clean(row["question"])
    a = _clean(row["answer"]).split("####")[0]   # strip the fingerprintable "#### N" tail
    text = truncate(f"Q: {q}\nA: {a}", TRUNC["q1"])
    gsm_records.append({"id": i, "source": "gsm8k", "text": text})
save_json(gsm_records, RAW / f"gsm8k_{len(gsm_records)}.json")

# Forget: SVAMP + full ASDiv under large scale so Q1 can reach ≥3000 combined.
# (Soft ASDiv-only capped the pool ~650 and could not test the size hypothesis.)
math_forget = []
svamp = load_dataset("ChilleD/SVAMP", split="train").shuffle(seed=SEED)
for i, row in enumerate(svamp):
    body = _clean(row.get("Body", ""))
    ques = _clean(row.get("Question", ""))
    ans  = _clean(row.get("Answer", ""))
    text = truncate(f"Q: {body} {ques}\nA: {ans}".strip(), TRUNC["q1"])
    math_forget.append({"id": i, "source": "svamp", "text": text})

def _try_asdiv(cap=None):
    for name, split in [("EleutherAI/asdiv", "validation"), ("asdiv", "validation"),
                        ("asdiv", "test")]:
        try:
            ds = load_dataset(name, split=split)
        except Exception as e:
            print(f"  source failed: {name}/{split}: {type(e).__name__}: {str(e)[:100]}")
            continue
        out = []
        for i, row in enumerate(ds):
            body = _clean(row.get("body", row.get("Body", "")))
            ques = _clean(row.get("question", row.get("Question", "")))
            ans  = _clean(row.get("answer", row.get("Answer", "")))
            text = truncate(f"Q: {body} {ques}\nA: {ans}".strip(), TRUNC["q1"])
            out.append({"id": 10_000 + i, "source": "asdiv", "text": text})
            if cap is not None and len(out) >= cap:
                break
        if len(out) >= 100:
            print(f"  loaded ASDiv via {name}/{split}  rows={len(out)}"
                  + (f" (cap={cap})" if cap else ""))
            return out
    return []

def _try_math_qa(cap=2000):
    """Extra math-word-problem forget pad (AllenAI MathQA)."""
    for name in ("allenai/math_qa", "math_qa"):
        try:
            ds = load_dataset(name, split="train").shuffle(seed=SEED)
        except Exception as e:
            print(f"  source failed: {name}: {type(e).__name__}: {str(e)[:120]}")
            continue
        out, seen = [], set()
        for i, row in enumerate(ds):
            prob = _clean(row.get("Problem", row.get("problem", "")))
            ans  = _clean(row.get("Rationale", row.get("correct", row.get("answer", ""))))
            text = truncate(f"Q: {prob}\nA: {ans}", TRUNC["q1"])
            if len(text) < LOWER["q1"] or text in seen:
                continue
            seen.add(text)
            out.append({"id": 20_000 + i, "source": "mathqa", "text": text})
            if len(out) >= cap:
                break
        if len(out) >= 200:
            print(f"  loaded MathQA forget pad via {name}  rows={len(out)}")
            return out
    return []


def _try_hendrycks_math(cap=3000):
    """Competition-math forget pad (longer answers → fills GSM8K length bins)."""
    out, seen = [], set()
    configs = ["algebra", "counting_and_probability", "geometry",
               "intermediate_algebra", "number_theory", "prealgebra", "precalculus"]
    for cfg in configs:
        try:
            ds = load_dataset("EleutherAI/hendrycks_math", cfg, split="train")
        except Exception as e:
            print(f"  source failed: hendrycks_math/{cfg}: {type(e).__name__}: {str(e)[:80]}")
            continue
        for i, row in enumerate(ds):
            prob = _clean(row.get("problem", ""))
            ans  = _clean(row.get("solution", ""))
            text = truncate(f"Q: {prob}\nA: {ans}", TRUNC["q1"])
            if len(text) < LOWER["q1"] or text in seen:
                continue
            seen.add(text)
            out.append({"id": 30_000 + len(out), "source": "hendrycks_math", "text": text})
            if len(out) >= cap:
                print(f"  loaded hendrycks_math forget pad  rows={len(out)}")
                return out
    if len(out) >= 200:
        print(f"  loaded hendrycks_math forget pad  rows={len(out)}")
        return out
    return []


if DATA_SCALE == "large":
    math_forget.extend(_try_asdiv(cap=None))
    math_forget.extend(_try_math_qa(cap=4000))
    math_forget.extend(_try_hendrycks_math(cap=4000))
    seen, deduped = set(), []
    for r in math_forget:
        if r["text"] in seen:
            continue
        seen.add(r["text"])
        deduped.append(r)
    math_forget = deduped
    print(f"  math forget after dedup: {len(math_forget)}")

save_json(math_forget, RAW / f"math_forget_{len(math_forget)}.json")

# ─────────────────────────────────────────────────────────────────────────────
# Q2 — HumanEval (+ APPS intro) vs MBPP (full) — Python code
# HumanEval alone is only 164 rows; under DATA_SCALE=large we pad the target
# with APPS "introductory" problems so Active-PU has a real U_pool.
# ─────────────────────────────────────────────────────────────────────────────
he = load_dataset("openai_humaneval", split="test")
he_records = [
    {
        "id":     i,
        "source": "humaneval",
        "text":   truncate(
            f"{_clean(row.get('prompt', ''))}\n{_clean(row.get('canonical_solution', ''))}",
            TRUNC["q2"],
        ),
    }
    for i, row in enumerate(he)
]

def _iter_codesearchnet(n_max, id_offset, skip=0):
    """Yield up to ``n_max`` CodeSearchNet Python rows, skipping the first ``skip`` kept ones."""
    try:
        ds = load_dataset("code_search_net", "python", split="train")
    except Exception as e:
        print(f"  source failed: code_search_net/python: {type(e).__name__}: {str(e)[:120]}")
        return []
    ds = ds.shuffle(seed=SEED)
    out, seen = [], set()
    kept = 0
    for i, row in enumerate(ds):
        doc = _clean(row.get("func_documentation_string", row.get("docstring", "")))
        code = _clean(row.get("func_code_string", row.get("whole_func_string", "")))
        if not code:
            continue
        text = truncate(f"{doc}\n{code}".strip() if doc else code, TRUNC["q2"])
        if len(text) < LOWER["q2"] or text in seen:
            continue
        seen.add(text)
        if kept < skip:
            kept += 1
            continue
        out.append({"id": id_offset + i, "source": "codesearchnet", "text": text})
        if len(out) >= n_max:
            break
    return out


def _try_codesearchnet_pad(n_max=3000):
    """Pad HumanEval with CodeSearchNet Python (docstring + function body)."""
    out = _iter_codesearchnet(n_max=n_max, id_offset=200_000, skip=0)
    print(f"  loaded CodeSearchNet python pad (target)  rows={len(out)}")
    return out


def _try_codesearchnet_forget(n_max=3000, skip=4000):
    """Forget pad from CSN — code-ONLY (no docstring), disjoint from target pad.

    Target pad uses docstring+code; forget pad drops the docstring so the two
    sides are not identical CSN clones (that previously collapsed skyline≈0.60).
    """
    try:
        ds = load_dataset("code_search_net", "python", split="train")
    except Exception as e:
        print(f"  source failed: code_search_net/python (forget): {type(e).__name__}: {str(e)[:120]}")
        return []
    ds = ds.shuffle(seed=SEED)
    out, seen = [], set()
    kept = 0
    for i, row in enumerate(ds):
        code = _clean(row.get("func_code_string", row.get("whole_func_string", "")))
        if not code:
            continue
        text = truncate(code, TRUNC["q2"])   # code only — different surface from target
        if len(text) < LOWER["q2"] or text in seen:
            continue
        seen.add(text)
        if kept < skip:
            kept += 1
            continue
        out.append({"id": 300_000 + i, "source": "codesearchnet_code", "text": text})
        if len(out) >= n_max:
            break
    print(f"  loaded CodeSearchNet code-only pad (forget)  rows={len(out)}")
    return out


def _try_apps_intro(n_max=3000):
    """Pad HumanEval with APPS introductory solutions (same coding-task family)."""
    try:
        ds = load_dataset("codeparrot/apps", split="train")
    except Exception as e:
        print(f"  source failed: codeparrot/apps: {type(e).__name__}: {str(e)[:120]}")
        return []
    ds = ds.shuffle(seed=SEED)
    out, seen = [], set()
    for i, row in enumerate(ds):
        diff = str(row.get("difficulty", "")).lower()
        if diff and "intro" not in diff and diff not in {"introductory", "0"}:
            if diff in {"interview", "competition", "1", "2", "3"}:
                continue
        prompt = _clean(row.get("question", row.get("problem", "")))
        sols = row.get("solutions", "")
        if isinstance(sols, str):
            try:
                import json as _json
                sols = _json.loads(sols)
            except Exception:
                sols = [sols]
        if isinstance(sols, list) and sols:
            sol = _clean(sols[0] if isinstance(sols[0], str) else str(sols[0]))
        else:
            sol = _clean(sols)
        text = truncate(f"{prompt}\n{sol}", TRUNC["q2"])
        if len(text) < LOWER["q2"] or text in seen:
            continue
        seen.add(text)
        out.append({"id": 100_000 + i, "source": "apps", "text": text})
        if len(out) >= n_max:
            break
    print(f"  loaded APPS introductory pad  rows={len(out)}")
    return out

code_target = list(he_records)
if DATA_SCALE == "large":
    pad = _try_apps_intro(n_max=TARGET_N["q2"] * 2)
    if len(pad) < 500:
        pad = _try_codesearchnet_pad(n_max=TARGET_N["q2"] * 2)
    code_target.extend(pad)
save_json(code_target, RAW / f"code_target_{len(code_target)}.json")

# MBPP — use full config (larger than sanitized) + all splits
def _load_mbpp():
    parts = []
    for cfg in ("full", "sanitized"):
        for split in ("train", "test", "validation", "prompt"):
            try:
                parts.append(load_dataset("google-research-datasets/mbpp", cfg, split=split))
            except Exception:
                continue
        if parts:
            break
    if not parts:
        raise RuntimeError("Could not load MBPP")
    mbpp = concatenate_datasets(parts)
    sample = mbpp[0]
    desc_key = "prompt" if "prompt" in sample else "text"
    records, seen = [], set()
    for i, row in enumerate(mbpp):
        text = truncate(
            f"# {_clean(row[desc_key])}\n{_clean(row['code'])}",
            TRUNC["q2"],
        )
        if text in seen:
            continue
        seen.add(text)
        records.append({"id": i, "source": "mbpp", "text": text})
    return records

mbpp_records = _load_mbpp()
if DATA_SCALE == "large":
    # Pad forget to unlock ≥3000 combined. Use code-only CSN (disjoint skip) so
    # forget ≠ target CSN docstring+code clones.
    need = max(0, TARGET_N["q2"] * 2 - len(mbpp_records))
    csn_forget = _try_codesearchnet_forget(
        n_max=max(need, TARGET_N["q2"] * 2),
        skip=TARGET_N["q2"] * 2,
    )
    tgt_texts = {r["text"] for r in code_target}
    csn_forget = [r for r in csn_forget if r["text"] not in tgt_texts]
    mbpp_records = mbpp_records + csn_forget
save_json(mbpp_records, RAW / f"mbpp_{len(mbpp_records)}.json")

# ─────────────────────────────────────────────────────────────────────────────
# Q3 — SQuAD (target) vs NQ-open (forget) -- extractive QA
# ─────────────────────────────────────────────────────────────────────────────
n_q3 = RAW_CAP["q3"]
squad = load_dataset("rajpurkar/squad", split="train").shuffle(seed=SEED)
squad = squad.select(range(min(n_q3, len(squad))))
squad_records = []
seen = set()
for i, row in enumerate(squad):
    q = _clean(row["question"])
    a = ""
    if row.get("answers") and row["answers"].get("text"):
        a = _clean(row["answers"]["text"][0])
    text = truncate(f"Q: {q}\nA: {a}", TRUNC["q3"])
    if text in seen:
        continue
    seen.add(text)
    squad_records.append({"id": i, "source": "squad", "text": text})
save_json(squad_records, RAW / f"squad_{len(squad_records)}.json")

nq = load_dataset("nq_open", split="train").shuffle(seed=SEED)
nq = nq.select(range(min(n_q3, len(nq))))
nq_records = []
seen = set()
for i, row in enumerate(nq):
    q = _clean(row["question"])
    a = ""
    if row.get("answer"):
        a = _clean(row["answer"][0]) if isinstance(row["answer"], list) else _clean(row["answer"])
    text = truncate(f"Q: {q}\nA: {a}", TRUNC["q3"])
    if text in seen:
        continue
    seen.add(text)
    nq_records.append({"id": i, "source": "nq_open", "text": text})
save_json(nq_records, RAW / f"nq_open_{len(nq_records)}.json")

# ─────────────────────────────────────────────────────────────────────────────
# Q4 — AG News (target) vs Reuters-21578 (forget) -- short news
# ─────────────────────────────────────────────────────────────────────────────
n_q4 = RAW_CAP["q4"]
ag = load_dataset("ag_news", split="train").shuffle(seed=SEED)
ag = ag.select(range(min(n_q4, len(ag))))
ag_records = []
for i, row in enumerate(ag):
    text = truncate(_clean(row["text"]), TRUNC["q4"])
    ag_records.append({"id": i, "source": "ag_news", "text": text})
save_json(ag_records, RAW / f"ag_news_{len(ag_records)}.json")

# Reuters-21578 on HF Hub is dataset-script-based and no longer loadable with
# current `datasets` library versions. We fall back to short news content
# whose *native* length distribution overlaps AG News (~80-200 chars) — so the
# overlap band stays non-empty after truncation. XSum summaries are short
# news TL;DRs that match AG News scale; DBpedia-14 entity descriptions are a
# secondary fallback.
def _load_news_forget():
    # Each candidate returns (records_list, source_tag) or None.

    def _try_xsum_summaries():
        try:
            ds = load_dataset("EdinburghNLP/xsum", split="train").shuffle(seed=SEED)
            ds = ds.select(range(min(n_q4, len(ds))))
        except Exception as e:
            print(f"  source failed: EdinburghNLP/xsum: {type(e).__name__}: {str(e)[:120]}")
            return None
        records, seen = [], set()
        for i, row in enumerate(ds):
            summary = _clean(row.get("summary", ""))
            if not (LOWER["q4"] <= len(summary) <= TRUNC["q4"]):
                continue
            if summary in seen:
                continue
            seen.add(summary)
            records.append({"id": i, "source": "xsum_summary", "text": summary})
        return (records, "xsum_summary") if len(records) > 400 else None

    def _try_dbpedia():
        try:
            ds = load_dataset("dbpedia_14", split="train").shuffle(seed=SEED)
            ds = ds.select(range(min(n_q4, len(ds))))
        except Exception as e:
            print(f"  source failed: dbpedia_14: {type(e).__name__}: {str(e)[:120]}")
            return None
        records, seen = [], set()
        for i, row in enumerate(ds):
            content = _clean(row.get("content", ""))
            if not (LOWER["q4"] <= len(content) <= TRUNC["q4"]):
                continue
            if content in seen:
                continue
            seen.add(content)
            records.append({"id": i, "source": "dbpedia14", "text": content})
        return (records, "dbpedia14") if len(records) > 400 else None

    for fn, name in [(_try_xsum_summaries, "xsum_summary"),
                     (_try_dbpedia,        "dbpedia14")]:
        result = fn()
        if result is not None:
            recs, tag = result
            print(f"  loaded news-forget via {fn.__name__}  source_tag={tag}  rows={len(recs)}")
            return recs, tag
    raise RuntimeError("No short-news fallback could be loaded.")

reuters_records, news_forget_source = _load_news_forget()
save_json(reuters_records, RAW / f"{news_forget_source}_{len(reuters_records)}.json")


# ─────────────────────────────────────────────────────────────────────────────
# Build the quadrants with length matching
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Building quadrants with length-stratified sampling ===")
sizes = {}
sizes["q1"] = make_quadrant("q1", gsm_records,   math_forget,     "gsm8k",     "svamp",
                            min_combined=Q1_MIN_COMBINED)
sizes["q2"] = make_quadrant("q2", code_target,   mbpp_records,    "humaneval",  "mbpp",
                            min_combined=Q2_MIN_COMBINED)
sizes["q3"] = make_quadrant("q3", squad_records, nq_records,      "squad",     "nq")
sizes["q4"] = make_quadrant("q4", ag_records,    reuters_records, "agnews",    news_forget_source)

print("\n=== Scale summary ===")
print(f"  DATA_SCALE = {DATA_SCALE}")
for qid, (nt, nf) in sizes.items():
    total = nt + nf
    n_lora = max(1, int(round(nt * 0.20)))
    n_p    = max(1, int(round(nt * 0.20)))
    n_hold = max(0, nt - n_lora - n_p)
    n_u    = n_hold + nf
    n_pool = n_u // 2
    n_test = n_u - n_pool
    flag = ""
    if qid == "q1" and DATA_SCALE == "large" and total < Q1_MIN_COMBINED:
        flag = f"  ⚠ below Q1_MIN_COMBINED={Q1_MIN_COMBINED}"
    if qid == "q2" and DATA_SCALE == "large" and total < Q2_MIN_COMBINED:
        flag = f"  ⚠ below Q2_MIN_COMBINED={Q2_MIN_COMBINED}"
    print(f"  {qid}: {nt}+{nf}={total}  →  ~P={n_p}  U_pool≈{n_pool}  U_test≈{n_test}{flag}")
    if flag:
        print(f"       → loosen N_BINS_PER[{qid!r}] or add more forget-source rows")

print("\n=== Final layout ===")
for p in sorted(BASE.rglob("*.json")):
    kb = p.stat().st_size / 1024
    print(f"  {str(p):<78}  {kb:>7.1f} KB")
print("\nDone.")
print("Next: delete stale checkpoints/results feature caches, then")
print("  python experiments/run_identification.py")
print("  python experiments/run_overlap_scorers.py --with-lira --train-shadows")
print("  python experiments/run_active_pu.py --rebuild-cache")
