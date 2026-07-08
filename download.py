"""Build the four data quadrants used by the identification experiment.

Pairs are intentionally drawn from the SAME broad domain so that text length /
surface formatting cannot trivially separate target from forget:

    Q1  Math word problems     GSM8K          (target)   vs  SVAMP        (forget)
    Q2  Python code            HumanEval      (target)   vs  MBPP         (forget)
    Q3  Extractive QA          SQuAD          (target)   vs  NQ-open      (forget)
    Q4  News                   AG News        (target)   vs  Reuters-21578(forget)

For every pair we:
  1. Truncate each record's `text` to a per-pair character cap.
  2. Drop records shorter than a per-pair lower bound.
  3. Run length-quintile stratified sampling so target and forget classes have
     IDENTICAL length-bin counts in the combined record. This is the strongest
     guarantee we can give that `len(text)` alone cannot classify.
  4. Print the AUROC of `len(text)` as a sanity check — should be near 0.5.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
from datasets import load_dataset
from sklearn.metrics import roc_auc_score

BASE  = Path("data")
RAW   = BASE / "raw"
QUADS = BASE / "quadrants"

# Per-pair caps and bounds (chars)
TRUNC      = {"q1": 200, "q2": 400, "q3": 150, "q4": 200}
LOWER      = {"q1":  60, "q2":  80, "q3":  40, "q4":  80}
TARGET_N   = {"q1": 1000, "q2":  500, "q3": 1000, "q4": 1000}
N_BINS_PER = {"q1": 10,   "q2":  8,  "q3": 30,   "q4": 30}
SEED       = 42

QUAD_DIRS = {
    "q1": QUADS / "q1_math__gsm8k_vs_svamp",
    "q2": QUADS / "q2_code__humaneval_vs_mbpp",
    "q3": QUADS / "q3_qa__squad_vs_nq",
    "q4": QUADS / "q4_news__agnews_vs_reuters",
}

for d in [RAW, *QUAD_DIRS.values()]:
    d.mkdir(parents=True, exist_ok=True)


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
    2. Within the overlap band, bin lengths into many narrow bins and keep the
       same number of target and forget records per bin. Many narrow bins
       suppress within-bin length variance and force the matched-sample
       length-AUROC close to 0.5.
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
    per_bin_cap = None
    if max_per_class is not None:
        per_bin_cap = max(1, max_per_class // n_bins)

    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        t_in = np.where((lt > lo) & (lt <= hi))[0]
        f_in = np.where((lf > lo) & (lf <= hi))[0]
        take = min(len(t_in), len(f_in))
        if per_bin_cap is not None:
            take = min(take, per_bin_cap)
        if take == 0:
            continue
        t_sel = rng.choice(t_in, size=take, replace=False)
        f_sel = rng.choice(f_in, size=take, replace=False)
        out_t.extend([rec_t[j] for j in t_sel])
        out_f.extend([rec_f[j] for j in f_sel])
    return out_t, out_f


def length_auroc(rec_t, rec_f) -> float:
    if not rec_t or not rec_f:
        return float("nan")
    lens   = np.array([len(r["text"]) for r in rec_t] +
                      [len(r["text"]) for r in rec_f])
    labels = np.array([1] * len(rec_t) + [0] * len(rec_f))
    return float(roc_auc_score(labels, lens))


def make_quadrant(qid, target_recs, forget_recs, t_name, f_name):
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

    matched_t, matched_f = length_stratified_sample(
        target_recs, forget_recs,
        n_bins=N_BINS_PER[qid],
        max_per_class=TARGET_N[qid],
    )
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


def _clean(s):
    if s is None:
        return ""
    return str(s).replace("\n", " ").strip()


# ─────────────────────────────────────────────────────────────────────────────
# Q1 — GSM8K (target) vs SVAMP (forget)  -- math word problems
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Loading sources ===")
gsm = load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=SEED)
gsm_records = []
for i, row in enumerate(gsm):
    q = _clean(row["question"])
    a = _clean(row["answer"]).split("####")[0]   # strip the fingerprintable "#### N" tail
    text = truncate(f"Q: {q}\nA: {a}", TRUNC["q1"])
    gsm_records.append({"id": i, "source": "gsm8k", "text": text})
save_json(gsm_records, RAW / f"gsm8k_train_{len(gsm_records)}.json")

svamp = load_dataset("ChilleD/SVAMP", split="train").shuffle(seed=SEED)
svamp_records = []
for i, row in enumerate(svamp):
    body = _clean(row.get("Body", ""))
    ques = _clean(row.get("Question", ""))
    ans  = _clean(row.get("Answer", ""))
    text = truncate(f"Q: {body} {ques}\nA: {ans}".strip(), TRUNC["q1"])
    svamp_records.append({"id": i, "source": "svamp", "text": text})
save_json(svamp_records, RAW / f"svamp_train_{len(svamp_records)}.json")

# ─────────────────────────────────────────────────────────────────────────────
# Q2 — HumanEval (target) vs MBPP (forget) — Python code, different sources,
# same task family. Length distributions are pulled into alignment by the
# overlap-band + stratified-bin matching shared by every pair.
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
save_json(he_records, RAW / f"humaneval_{len(he_records)}.json")

mbpp_train = load_dataset("google-research-datasets/mbpp", "sanitized", split="train")
mbpp_test  = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
mbpp_val   = load_dataset("google-research-datasets/mbpp", "sanitized", split="validation")
from datasets import concatenate_datasets
mbpp = concatenate_datasets([mbpp_train, mbpp_test, mbpp_val]).shuffle(seed=SEED)
sample = mbpp[0]
MBPP_DESC = "prompt" if "prompt" in sample else "text"

mbpp_records = [
    {
        "id":     i,
        "source": "mbpp",
        "text":   truncate(
            f"# {_clean(row[MBPP_DESC])}\n{_clean(row['code'])}",
            TRUNC["q2"],
        ),
    }
    for i, row in enumerate(mbpp)
]
save_json(mbpp_records, RAW / f"mbpp_sanitized_{len(mbpp_records)}.json")

# ─────────────────────────────────────────────────────────────────────────────
# Q3 — SQuAD (target) vs NQ-open (forget) -- extractive QA
# ─────────────────────────────────────────────────────────────────────────────
squad = load_dataset("rajpurkar/squad", split="train").shuffle(seed=SEED).select(range(8000))
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

nq = load_dataset("nq_open", split="train").shuffle(seed=SEED).select(range(8000))
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
ag = load_dataset("ag_news", split="train").shuffle(seed=SEED).select(range(8000))
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
            ds = ds.select(range(min(8000, len(ds))))
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
            ds = ds.select(range(min(8000, len(ds))))
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
make_quadrant("q1", gsm_records,     svamp_records,   "gsm8k",     "svamp")
make_quadrant("q2", he_records,      mbpp_records,    "humaneval",   "mbpp")
make_quadrant("q3", squad_records,   nq_records,      "squad",     "nq")
make_quadrant("q4", ag_records,      reuters_records, "agnews",    news_forget_source)

print("\n=== Final layout ===")
for p in sorted(BASE.rglob("*.json")):
    kb = p.stat().st_size / 1024
    print(f"  {str(p):<78}  {kb:>7.1f} KB")
print("\nDone.")
