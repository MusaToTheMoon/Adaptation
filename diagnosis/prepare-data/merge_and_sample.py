"""
merge_and_sample.py
--------------------
Merges separate English and Arabic inference CSVs into a single file,
assigns quadrant labels, and samples 100 per quadrant (deterministic,
seed=42) — identical sampling every run on the same data.

CRITICAL: uses seed=42 + sorts by question ID before sampling so the
output is STABLE across machines and Python versions.

Quadrants:
    both_correct  — English correct AND Arabic correct
    access_gap    — English correct AND Arabic wrong   ← key group
    arabic_only   — English wrong   AND Arabic correct
    both_wrong    — English wrong   AND Arabic wrong

Expected input CSVs (one for English, one for Arabic).
Must share a common ID column and a ground_truth column.
Prediction column can be named: prediction, answer, model_answer, output
(detected automatically).

Usage examples
--------------
# ALLaM 7B
python merge_and_sample.py \\
    --english_file  allam_english.csv \\
    --arabic_file   allam_arabic.csv \\
    --model         "ALLaM 7B" \\
    --output_merged allam_merged.csv \\
    --output_sample allam_sampled_quadrants.csv

# MedGemma
python merge_and_sample.py \\
    --english_file  medgemma_english.csv \\
    --arabic_file   medgemma_arabic.csv \\
    --model         "MedGemma" \\
    --output_merged medgemma_merged.csv \\
    --output_sample medgemma_sampled_quadrants.csv

# With explicit column names if auto-detection fails:
python merge_and_sample.py \\
    --english_file  allam_english.csv \\
    --arabic_file   allam_arabic.csv \\
    --id_col        id \\
    --gt_col        ground_truth \\
    --pred_col      prediction \\
    --model         "ALLaM 7B" \\
    --output_merged allam_merged.csv \\
    --output_sample allam_sampled_quadrants.csv
"""

import csv
import re
import random
import argparse
from collections import defaultdict

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED               = 42
SAMPLES_PER_QUAD   = 100

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--english_file",   required=True)
parser.add_argument("--arabic_file",    required=True)
parser.add_argument("--model",          default="Model",
                    help="Model name written into output CSV")
parser.add_argument("--output_merged",  default="merged.csv")
parser.add_argument("--output_sample",  default="sampled_quadrants.csv")

# Optional overrides for column names
parser.add_argument("--id_col",   default=None,
                    help="Name of question ID column (auto-detected if omitted)")
parser.add_argument("--gt_col",   default=None,
                    help="Name of ground-truth column (auto-detected)")
parser.add_argument("--pred_col", default=None,
                    help="Name of prediction column (auto-detected)")
args = parser.parse_args()

# ── Helpers ───────────────────────────────────────────────────────────────────
PRED_ALIASES = ["prediction", "answer", "model_answer", "output",
                "predicted_answer", "pred", "response"]
GT_ALIASES   = ["ground_truth", "answer_key", "correct_answer",
                "label", "gt", "correct"]
ID_ALIASES   = ["id", "question_id", "qid", "idx", "index", "q_id"]

def detect_col(fieldnames, aliases, label):
    lower = {f.lower(): f for f in fieldnames}
    for alias in aliases:
        if alias in lower:
            return lower[alias]
    raise ValueError(
        f"Cannot detect {label} column in {fieldnames}. "
        f"Pass --{label.replace(' ', '_')}_col explicitly.")

def extract_letter(val):
    if not val or not isinstance(val, str): return ""
    s = val.strip().upper()
    m = re.search(r'\bANSWER\s*:\s*([A-F])\b', s)
    if m: return m.group(1)
    m = re.search(r'\b([A-F])\b', s)
    return m.group(1) if m else ""

def normalize_year(val):
    if not val: return None
    s = str(val).strip().upper()
    for pat in [r"\bY\s*([1-5])\b", r"\bYEAR\s*([1-5])\b", r"\b([1-5])\b"]:
        m = re.search(pat, s)
        if m: return int(m.group(1))
    return None

def get_quadrant(en_ok, ar_ok):
    if en_ok and ar_ok:     return "both_correct"
    if en_ok and not ar_ok: return "access_gap"
    if not en_ok and ar_ok: return "arabic_only"
    return "both_wrong"

def load_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader), reader.fieldnames

def build_input(row):
    """Return a rendered MCQ input from either an input column or question/options."""
    if not row:
        return ""
    direct = row.get("input")
    if direct and str(direct).strip():
        return str(direct).strip()

    stem = (row.get("question") or row.get("Question") or "").strip()
    option_keys = [
        ("A", "opa"), ("B", "opb"), ("C", "opc"),
        ("D", "opd"), ("E", "ope"), ("F", "opf"),
    ]
    lines = []
    for letter, key in option_keys:
        value = row.get(key)
        if value is None:
            continue
        value = str(value).strip()
        if value:
            lines.append(f"{letter}) {value}")

    if stem and lines:
        return stem + "\n\n" + "\n".join(lines)
    return stem

# ── Load English file ─────────────────────────────────────────────────────────
print(f"Loading English file: {args.english_file}")
en_rows, en_fields = load_csv(args.english_file)
print(f"  {len(en_rows)} rows,  columns: {en_fields}")

id_col  = args.id_col   or detect_col(en_fields, ID_ALIASES,   "id")
gt_col  = args.gt_col   or detect_col(en_fields, GT_ALIASES,   "ground_truth")
pred_col = args.pred_col or detect_col(en_fields, PRED_ALIASES, "prediction")
print(f"  Using → id_col='{id_col}'  gt_col='{gt_col}'  pred_col='{pred_col}'")

en_by_id = {r[id_col]: r for r in en_rows}

# ── Load Arabic file ──────────────────────────────────────────────────────────
print(f"\nLoading Arabic file: {args.arabic_file}")
ar_rows, ar_fields = load_csv(args.arabic_file)
print(f"  {len(ar_rows)} rows,  columns: {ar_fields}")

# Auto-detect prediction column in Arabic file (may have different name)
ar_pred_col = args.pred_col or detect_col(ar_fields, PRED_ALIASES, "prediction")
ar_id_col   = args.id_col   or detect_col(ar_fields, ID_ALIASES, "id")
print(f"  Using → id_col='{ar_id_col}'  pred_col='{ar_pred_col}'")

ar_pred_by_id = {r[ar_id_col]: extract_letter(r.get(ar_pred_col, ""))
                 for r in ar_rows}

ar_input_by_id = {r[ar_id_col]: build_input(r) for r in ar_rows}
ar_level_by_id = {
    r[ar_id_col]: (r.get("Level") or r.get("level") or r.get("year") or "")
    for r in ar_rows
}

# ── Merge ─────────────────────────────────────────────────────────────────────
print("\nMerging ...")
merged = []
missing_ar = 0

for row in en_rows:
    qid = row[id_col]
    gt  = extract_letter(row.get(gt_col, ""))
    if not gt:
        continue

    en_pred = extract_letter(row.get(pred_col, ""))
    ar_pred = ar_pred_by_id.get(qid, "")
    if not ar_pred:
        missing_ar += 1

    en_ok = en_pred == gt and en_pred != ""
    ar_ok = ar_pred == gt and ar_pred != ""

    level = row.get("Level") or row.get("level") or row.get("year") or ar_level_by_id.get(qid, "")
    year  = normalize_year(level)
    group = ("Early" if year in (1, 2) else "Late") if year else "Unknown"

    merged.append({
        "id":               qid,
        "quadrant":         get_quadrant(en_ok, ar_ok),
        "ground_truth":     gt,
        "prediction_english": en_pred,
        "prediction_arabic":  ar_pred,
        "en_correct":       int(en_ok),
        "ar_correct":       int(ar_ok),
        "Level":            level,
        "_group":           group,
        "_year":            year,
        # Preserve question text columns
        "input_english":    row.get("input_english") or row.get("question_english")
                            or build_input(row),
        "input_arabic":     ar_input_by_id.get(qid, ""),
        "model":            args.model,
    })

print(f"  Merged: {len(merged)} rows  (missing Arabic prediction: {missing_ar})")

# Quadrant counts before sampling
q_counts = defaultdict(int)
for r in merged: q_counts[r["quadrant"]] += 1
print("\n=== Quadrant counts (full merged) ===")
for q in ["both_correct", "access_gap", "arabic_only", "both_wrong"]:
    print(f"  {q:<15}: {q_counts[q]}")

# ── Save merged ───────────────────────────────────────────────────────────────
MERGED_FIELDS = ["id", "model", "quadrant", "en_correct", "ar_correct",
                 "ground_truth", "prediction_english", "prediction_arabic",
                 "input_english", "input_arabic", "Level"]
with open(args.output_merged, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=MERGED_FIELDS, extrasaction="ignore")
    w.writeheader()
    w.writerows(merged)
print(f"\nMerged CSV → {args.output_merged}")

# ── Sample 100 per quadrant (deterministic: seed=42 + sort by id) ─────────────
# Sorting by id BEFORE seeding ensures identical output regardless of CSV order.
random.seed(SEED)

print(f"\n=== Sampling {SAMPLES_PER_QUAD} per quadrant (seed={SEED}) ===")
quads = defaultdict(list)
for r in sorted(merged, key=lambda x: str(x["id"])):  # sort for determinism
    quads[r["quadrant"]].append(r)

sampled = []
for q in ["both_correct", "access_gap", "arabic_only", "both_wrong"]:
    pool   = quads[q]
    early  = [r for r in pool if r["_group"] == "Early"]
    late   = [r for r in pool if r["_group"] != "Early"]
    n      = min(SAMPLES_PER_QUAD, len(pool))

    if len(pool) == 0:
        print(f"  {q:<15}: WARNING — no rows available, skipping")
        continue

    # Proportional Early/Late split
    if len(pool) > 0:
        n_early = min(round(n * len(early) / len(pool)), len(early))
        n_late  = min(n - n_early, len(late))
    else:
        n_early, n_late = 0, 0

    chosen = random.sample(early, n_early) + random.sample(late, n_late)
    sampled.extend(chosen)
    print(f"  {q:<15}: sampled {len(chosen):3d}  (Early: {n_early}, Late: {n_late})")

print(f"\n  Total: {len(sampled)} questions")

# ── Save sampled ──────────────────────────────────────────────────────────────
with open(args.output_sample, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=MERGED_FIELDS, extrasaction="ignore")
    w.writeheader()
    w.writerows(sampled)
print(f"  Sampled CSV → {args.output_sample}")

# ── Summary ───────────────────────────────────────────────────────────────────
summary = args.output_sample.replace(".csv", "_summary.txt")
with open(summary, "w") as f:
    f.write(f"Model: {args.model}\n")
    f.write(f"Seed:  {SEED}\n\n")
    for q in ["both_correct", "access_gap", "arabic_only", "both_wrong"]:
        rows = [r for r in sampled if r["quadrant"] == q]
        early = sum(1 for r in rows if r["_group"] == "Early")
        late  = len(rows) - early
        f.write(f"{q}: n={len(rows)}  (Early: {early}, Late: {late})\n")
    f.write(f"\nTotal: {len(sampled)}\n")
print(f"  Summary → {summary}")
print("\nDone.")
