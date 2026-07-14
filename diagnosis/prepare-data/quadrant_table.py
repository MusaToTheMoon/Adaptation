"""
quadrant_table.py
-----------------
Reads prediction CSVs for each model and computes:
  - Quadrant distribution (count + % of total)
  - English accuracy and Arabic accuracy overall
  - EN and AR accuracy within each quadrant

Produces a console summary table and ready-to-paste LaTeX code.

Expected CSV columns (matches the MedAraBench prediction format):
  model, quadrant, en_correct, ar_correct, ...

Usage:
    # Option A: pass individual files with display names
    python quadrant_table.py \\
        --files mistral_predictions.csv:Mistral-3.2-24B \\
                llama_predictions.csv:Llama-3.3-70B \\
                allam_predictions.csv:ALLaM-7B \\
                medgemma_predictions.csv:MedGemma-27B

    # Option B: auto-discover all CSVs in a directory
    python quadrant_table.py --dir ./predictions

    # Output LaTeX to file
    python quadrant_table.py --files ... --latex quadrant_table.tex
"""

import os
import csv
import argparse
from collections import defaultdict

# ── Args ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--files", nargs="+", default=[],
                    help="path:DisplayName pairs, e.g. mistral.csv:Mistral-3.2-24B")
parser.add_argument("--dir", default=None,
                    help="Directory to auto-discover CSVs (uses filename as display name)")
parser.add_argument("--latex", default=None,
                    help="Output LaTeX to this file (also prints to stdout)")
args = parser.parse_args()

# Order matches table column layout: En✓Ar✓ | En✓Ar✗ | En✗Ar✓ | En✗Ar✗
QUADRANT_ORDER = ["both_correct", "access_gap", "arabic_only", "both_wrong"]
QUADRANT_LABELS = {
    "access_gap":   "Access Gap\\n(En\\cmark\\ Ar\\xmark)",
    "both_correct": "Both Correct\\n(En\\cmark\\ Ar\\cmark)",
    "both_wrong":   "Both Wrong\\n(En\\xmark\\ Ar\\xmark)",
    "arabic_only":  "Arabic Only\\n(En\\xmark\\ Ar\\cmark)",
}
QUADRANT_SHORT = {
    "access_gap":   "Access Gap",
    "both_correct": "Both Correct",
    "both_wrong":   "Both Wrong",
    "arabic_only":  "Arabic Only",
}

# ── Collect files ─────────────────────────────────────────────────────────────
file_pairs = []  # [(path, display_name), ...]

if args.files:
    for entry in args.files:
        if ":" in entry:
            path, name = entry.split(":", 1)
        else:
            path = entry
            name = os.path.splitext(os.path.basename(path))[0]
        file_pairs.append((path, name))

if args.dir:
    for fname in sorted(os.listdir(args.dir)):
        if fname.endswith(".csv"):
            file_pairs.append((
                os.path.join(args.dir, fname),
                os.path.splitext(fname)[0]
            ))

if not file_pairs:
    print("No files specified. Use --files path:Name ... or --dir <directory>")
    import sys; sys.exit(1)

# ── Parse CSVs ────────────────────────────────────────────────────────────────
def load_csv(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

def extract_letter(val):
    """Extract answer letter A-F from a prediction or ground_truth string."""
    import re
    if not val or not isinstance(val, str): return ""
    s = val.strip().upper()
    m = re.search(r'\bANSWER\s*:\s*([A-F])\b', s)
    if m: return m.group(1)
    m = re.search(r'\b([A-F])\b', s)
    return m.group(1) if m else ""

def int_col(row, col):
    """Safely parse 0/1 integer column, treating empty/NA as None."""
    v = row.get(col, "").strip()
    if v in ("", "NA", "None", "nan"): return None
    try: return int(float(v))
    except: return None

QUADRANT_MAP = {(1,0): "access_gap", (1,1): "both_correct",
                (0,0): "both_wrong",  (0,1): "arabic_only"}

def resolve_row(row):
    """
    Returns (quadrant, en_correct, ar_correct) for a row.
    Handles two CSV formats:
      A) Already has quadrant + en_correct + ar_correct columns (merged format)
      B) Has prediction_english + prediction_arabic + ground_truth (raw format)
    """
    enc = int_col(row, "en_correct")
    arc = int_col(row, "ar_correct")
    q   = row.get("quadrant", "").strip()

    # Format B: compute from raw predictions
    if enc is None or arc is None:
        gt   = extract_letter(row.get("ground_truth", ""))
        en_p = extract_letter(row.get("prediction_english", ""))
        ar_p = extract_letter(row.get("prediction_arabic",  ""))
        if gt and en_p: enc = int(en_p == gt)
        if gt and ar_p: arc = int(ar_p == gt)

    if enc is not None and arc is not None and not q:
        q = QUADRANT_MAP.get((enc, arc), "unknown")

    return q, enc, arc

# ── Compute stats ─────────────────────────────────────────────────────────────
results = []  # list of dicts, one per model

for path, display_name in file_pairs:
    if not os.path.exists(path):
        print(f"WARNING: {path} not found — skipping")
        continue

    rows = load_csv(path)
    total = len(rows)

    # Detect format
    sample_cols = set(rows[0].keys()) if rows else set()
    has_precomputed = "en_correct" in sample_cols and "quadrant" in sample_cols
    has_raw = "prediction_english" in sample_cols and "prediction_arabic" in sample_cols
    print(f"  {display_name}: {'precomputed' if has_precomputed else 'raw predictions'} format  ({total} rows)")

    # Per-quadrant counts and accuracy
    quad_counts   = defaultdict(int)
    quad_en_corr  = defaultdict(int)
    quad_ar_corr  = defaultdict(int)
    quad_valid    = defaultdict(int)  # rows with non-null en/ar correct

    for row in rows:
        q, enc, arc = resolve_row(row)
        if not q or q == "unknown": continue
        quad_counts[q] += 1
        if enc is not None and arc is not None:
            quad_valid[q]   += 1
            quad_en_corr[q] += enc
            quad_ar_corr[q] += arc

    # Overall accuracy — use valid_n as denominator throughout for consistency.
    # valid_n = rows where BOTH en_correct and ar_correct could be resolved.
    # Quadrant percentages use the same denominator so they sum to 100% and
    # EN Acc = both_correct% + access_gap%, AR Acc = both_correct% + arabic_only%.
    resolved = [resolve_row(r) for r in rows]
    valid_n = sum(1 for q,e,a in resolved if e is not None and a is not None)
    en_acc  = 100 * sum(e for _,e,a in resolved if e is not None and a is not None) / valid_n if valid_n else 0
    ar_acc  = 100 * sum(a for _,e,a in resolved if e is not None and a is not None) / valid_n if valid_n else 0

    if valid_n < total:
        print(f"  NOTE: {total - valid_n} rows had unresolvable predictions — excluded from accuracy stats.")

    stat = {
        "name":   display_name,
        "en_acc": en_acc,
        "ar_acc": ar_acc,
    }
    quad_total = sum(quad_counts.values())  # rows that resolved to a valid quadrant
    for q in QUADRANT_ORDER:
        n   = quad_counts.get(q, 0)
        pct = 100 * n / quad_total if quad_total else 0
        stat[f"{q}_pct"] = pct
        vn = quad_valid.get(q, 0)
        stat[f"{q}_en_acc"] = 100 * quad_en_corr[q] / vn if vn else 0
        stat[f"{q}_ar_acc"] = 100 * quad_ar_corr[q] / vn if vn else 0

    results.append(stat)

if not results:
    print("No data loaded.")
    import sys; sys.exit(1)

# ── Console table ─────────────────────────────────────────────────────────────
col_w = 14
hdr_w = 22

header_row = f"{'Model':<{hdr_w}}" + "".join(
    f"{QUADRANT_SHORT[q]:>{col_w}}" for q in QUADRANT_ORDER
) + f"{'EN Acc':>{col_w}}{'AR Acc':>{col_w}}"
sep = "-" * len(header_row)

print("\n" + sep)
print(header_row)
print(sep)
for r in results:
    quad_str = "".join(
        f"{r[f'{q}_pct']:>{col_w}.2f}%" for q in QUADRANT_ORDER
    )
    print(f"{r['name']:<{hdr_w}}" + quad_str +
          f"{r['en_acc']:>{col_w}.2f}{'%'}" +
          f"{r['ar_acc']:>{col_w}.2f}{'%'}")
print(sep)

print("\nDetailed accuracy per quadrant:")
print(sep)
hdr2 = f"{'Model':<{hdr_w}}" + "".join(
    f"{'EN|AR '+QUADRANT_SHORT[q][:10]:>{col_w+2}}" for q in QUADRANT_ORDER
)
print(hdr2)
print(sep)
for r in results:
    detail = "".join(
        f"{r[f'{q}_en_acc']:>6.2f}|{r[f'{q}_ar_acc']:<6.2f}" for q in QUADRANT_ORDER
    )
    print(f"{r['name']:<{hdr_w}}{detail}")
print(sep + "\n")

# ── LaTeX ─────────────────────────────────────────────────────────────────────
# Column headers match QUADRANT_ORDER: both_correct, access_gap, arabic_only, both_wrong
# i.e.  En✓Ar✓ | En✓Ar✗ | En✗Ar✓ | En✗Ar✗
QUAD_TEX = {
    "both_correct": r"\makecell{En\checkmark\\Ar\checkmark}",
    "access_gap":   r"\makecell{En\checkmark\\Ar\texttimes}",
    "arabic_only":  r"\makecell{En\texttimes\\Ar\checkmark}",
    "both_wrong":   r"\makecell{En\texttimes\\Ar\texttimes}",
}

lines = []
lines.append(r"% Requires: \usepackage{booktabs,makecell,pifont}")
lines.append(r"% \newcommand{\cmark}{\ding{51}}")
lines.append(r"% \newcommand{\xmark}{\ding{55}}")
lines.append(r"")
lines.append(r"\begin{table}[t]")
lines.append(r"\centering")
lines.append(r"\small")
lines.append(r"\setlength{\tabcolsep}{5pt}")
lines.append(
    r"\caption{Quadrant distribution across models. "
    r"Each cell shows count (\%) of questions where the model answers "
    r"correctly/incorrectly in English and Arabic. "
    r"EN Acc / AR Acc are overall zero-shot accuracies.}"
)
lines.append(r"\label{tab:quadrant_distribution}")

ncols = len(QUADRANT_ORDER) + 3  # model + 4 quadrants + EN acc + AR acc
col_spec = "l" + "c" * (ncols - 1)
lines.append(r"\begin{tabular}{" + col_spec + r"}")
lines.append(r"\toprule")

# Header
quad_hdrs = " & ".join(QUAD_TEX[q] for q in QUADRANT_ORDER)
lines.append(r"Model & " + quad_hdrs + r" & EN Acc & AR Acc \\")
lines.append(r"\midrule")

# Data rows
for r in results:
    cells = [r["name"].replace("_", r"\_")]
    for q in QUADRANT_ORDER:
        cells.append(f"{r[f'{q}_pct']:.2f}\\%")
    cells.append(f"{r['en_acc']:.2f}\\%")
    cells.append(f"{r['ar_acc']:.2f}\\%")
    lines.append(" & ".join(cells) + r" \\")

lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"\end{table}")

latex_str = "\n".join(lines)
print(latex_str)

if args.latex:
    with open(args.latex, "w", encoding="utf-8") as f:
        f.write(latex_str)
    print(f"\nLaTeX saved → {args.latex}")