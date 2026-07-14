"""
merge_en_ar.py
--------------
Merges separate English and Arabic prediction CSVs into one unified file
with quadrant labels, matching the format expected by quadrant_table.py.

Input CSVs (each): id, input, prediction, ground_truth
Output CSV:        id, model, quadrant, en_correct, ar_correct,
                   ground_truth, prediction_english, prediction_arabic,
                   input_english, input_arabic

Usage:
    python merge_en_ar.py \
        --en llama_en.csv \
        --ar llama_ar.csv \
        --model "Llama-3.3-70B" \
        --out llama_predictions.csv
"""

import csv
import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--en",    required=True, help="English predictions CSV")
parser.add_argument("--ar",    required=True, help="Arabic predictions CSV")
parser.add_argument("--model", required=True, help="Display name for the model column")
parser.add_argument("--out",   required=True, help="Output merged CSV path")
args = parser.parse_args()

# ── Helpers ───────────────────────────────────────────────────────────────────
def extract_letter(val):
    """Extract answer letter A-F from prediction or ground_truth string."""
    if not val or not isinstance(val, str):
        return ""
    s = val.strip().upper()
    # Match patterns like "ANSWER: A", "Answer: B", or bare "A"
    m = re.search(r'\bANSWER\s*:\s*([A-F])\b', s)
    if m: return m.group(1)
    m = re.search(r'\b([A-F])\b', s)
    return m.group(1) if m else ""

def load_csv(path):
    rows = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row_id = row.get("id", "").strip()
            rows[row_id] = row
    return rows

# ── Load both files ───────────────────────────────────────────────────────────
print(f"Loading EN: {args.en}")
en_rows = load_csv(args.en)
print(f"  {len(en_rows)} rows")

print(f"Loading AR: {args.ar}")
ar_rows = load_csv(args.ar)
print(f"  {len(ar_rows)} rows")

# ── Match on id ───────────────────────────────────────────────────────────────
all_ids = sorted(set(en_rows.keys()) | set(ar_rows.keys()),
                 key=lambda x: int(x) if x.isdigit() else x)

only_en = set(en_rows) - set(ar_rows)
only_ar = set(ar_rows) - set(en_rows)
if only_en:
    print(f"WARNING: {len(only_en)} IDs in EN only (will have ar_correct=None)")
if only_ar:
    print(f"WARNING: {len(only_ar)} IDs in AR only (will have en_correct=None)")

matched = set(en_rows) & set(ar_rows)
print(f"  Matched on id: {len(matched)}")

# ── Build merged rows ─────────────────────────────────────────────────────────
QUADRANT = {
    (1, 0): "access_gap",
    (1, 1): "both_correct",
    (0, 0): "both_wrong",
    (0, 1): "arabic_only",
}

out_rows = []
stats = {"access_gap": 0, "both_correct": 0, "both_wrong": 0, "arabic_only": 0}

for row_id in all_ids:
    en = en_rows.get(row_id)
    ar = ar_rows.get(row_id)

    gt_letter = extract_letter((en or ar).get("ground_truth", ""))

    en_pred    = extract_letter(en["prediction"]) if en else ""
    ar_pred    = extract_letter(ar["prediction"]) if ar else ""

    en_correct = int(en_pred == gt_letter) if (en and gt_letter) else None
    ar_correct = int(ar_pred == gt_letter) if (ar and gt_letter) else None

    if en_correct is not None and ar_correct is not None:
        quadrant = QUADRANT[(en_correct, ar_correct)]
    else:
        quadrant = "unknown"

    if quadrant in stats:
        stats[quadrant] += 1

    out_rows.append({
        "id":                 row_id,
        "model":              args.model,
        "quadrant":           quadrant,
        "en_correct":         en_correct if en_correct is not None else "",
        "ar_correct":         ar_correct if ar_correct is not None else "",
        "ground_truth":       gt_letter,
        "prediction_english": en_pred,
        "prediction_arabic":  ar_pred,
        "input_english":      en["input"] if en else "",
        "input_arabic":       ar["input"] if ar else "",
    })

# ── Write output ──────────────────────────────────────────────────────────────
fieldnames = ["id", "model", "quadrant", "en_correct", "ar_correct",
              "ground_truth", "prediction_english", "prediction_arabic",
              "input_english", "input_arabic"]

with open(args.out, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(out_rows)

print(f"\nSaved → {args.out}  ({len(out_rows)} rows)")
print("\nQuadrant distribution:")
total = len(out_rows)
for q, label in [("access_gap","Access Gap (En✓ Ar✗)"),
                  ("both_correct","Both Correct (En✓ Ar✓)"),
                  ("both_wrong","Both Wrong (En✗ Ar✗)"),
                  ("arabic_only","Arabic Only (En✗ Ar✓)")]:
    n = stats[q]
    print(f"  {label:<30} {n:>5}  ({100*n/total:.1f}%)")