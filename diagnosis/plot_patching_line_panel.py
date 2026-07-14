"""
plot_patching_panel_lines.py
-----------------------
2×2 panel figure from activation patching results — LINE PLOT VERSION.
Recovery % (y-axis) vs patch layer (x-axis), with Arabic/English baselines
as horizontal reference lines.

Usage:
  python plot_patching_panel_lines.py \
      --llama_dir    ./activation_patching_llama_out \
      --mistral_dir  ./activation_patching_mistral_out \
      --allam_dir    ./activation_patching_allam_out \
      --medgemma_dir ./activation_patching_medgemma_out \
      --out_dir      ./activation_patching_panel_out
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

parser = argparse.ArgumentParser()
parser.add_argument("--model", action="append", default=[],
                    help="Repeatable 'Label=directory' entry. Overrides built-in model args when provided.")
parser.add_argument("--gemma_dir",   default="./diagnosis/activation_patching_gemma3_27b_it_out")
parser.add_argument("--mistral7b_dir", default="./diagnosis/activation_patching_mistral_7b_out")
parser.add_argument("--llama8b_dir", default="./diagnosis/activation_patching_llama31_8b_inst_out")
parser.add_argument("--falcon_dir",  default="./diagnosis/activation_patching_falcon_out")
parser.add_argument("--fanar_dir",   default="./diagnosis/activation_patching_fanar_out")
parser.add_argument("--llama_dir",    default="./activation_patching_llama_out")
parser.add_argument("--mistral_dir",  default="./activation_patching_mistral_out")
parser.add_argument("--allam_dir",    default="./activation_patching_allam_out")
parser.add_argument("--medgemma_dir", default="./activation_patching_medgemma_out")
parser.add_argument("--out_dir",      default="./activation_patching_panel_out")
args = parser.parse_args()
os.makedirs(args.out_dir, exist_ok=True)

# ── Palette (consistent with tuned-lens panel) ─────────────────────────────────
PALETTE = {
    'Base': '#0D3349',   # sky blue  — Arabic baseline
    'CoT':  '#2A9D8F',   # teal      — English baseline
    'line': '#C0392B',   # crimson   — recovery line
}

AR_COLOR   = PALETTE['Base']
EN_COLOR   = PALETTE['CoT']
LINE_COLOR = PALETTE['line']
ZERO_COLOR = '#888888'

# ── Model registry ─────────────────────────────────────────────────────────────
def parse_models(args):
    if args.model:
        out = []
        for item in args.model:
            if "=" not in item:
                raise ValueError(f"--model must be 'Label=directory', got: {item}")
            label, directory = item.split("=", 1)
            out.append((label.strip(), directory.strip()))
        return out
    return [
        ("Gemma-3-27B-IT", args.gemma_dir),
        ("Mistral-7B-Instruct-v0.3", args.mistral7b_dir),
        ("Llama-3.1-8B-Instruct", args.llama8b_dir),
        ("Falcon-H1-7B-Instruct", args.falcon_dir),
        ("Fanar-1-9B", args.fanar_dir),
    ]


MODELS = parse_models(args)

# ── Load helper ────────────────────────────────────────────────────────────────
def load_npz(out_dir):
    path = os.path.join(out_dir, "patching_results.npz")
    if not os.path.exists(path):
        return None
    d = np.load(path, allow_pickle=True)
    en_base  = d["en_base_prob"]
    ar_base  = d["ar_base_prob"]
    N        = len(en_base)
    patch_keys = [k for k in d.files
                  if k not in ("en_base_prob", "ar_base_prob", "gt_letters")]
    patch_data = {k: d[k] for k in patch_keys}
    return dict(en_base=en_base, ar_base=ar_base, N=N, patch_data=patch_data)

# ── Recovery helper ────────────────────────────────────────────────────────────
def recovery(val, ar_mean, en_mean):
    gap = en_mean - ar_mean
    return 100 * (val - ar_mean) / gap if gap > 0 else 0.0

# ── Shared y-axis range (computed across all models after loading) ─────────────
SHARED_Y_MIN = -60
SHARED_Y_MAX = 130  # fixed ceiling; Mistral over-recovery annotated, not scaled to

# ── Draw one subplot ───────────────────────────────────────────────────────────
def draw_panel(ax, data, model_name, show_ylabel, show_xlabel):
    if data is None:
        ax.set_facecolor("#f5f5f5")
        ax.text(0.5, 0.5, "Pending", ha="center", va="center",
                fontsize=15, color="#aaaaaa", transform=ax.transAxes)
        ax.set_title(model_name, fontsize=17, fontweight="bold", pad=8)
        ax.axis("off")
        return

    en_base    = data["en_base"]
    ar_base    = data["ar_base"]
    patch_data = data["patch_data"]
    N          = data["N"]

    ar_mean = ar_base.mean()
    en_mean = en_base.mean()

    # Single-layer patches for main line; span patches annotated separately
    single_keys = [k for k in patch_data if "_" not in k.replace("patch_", "")]
    span_keys   = [k for k in patch_data if k not in single_keys]

    def get_rec_sem(keys):
        means = [patch_data[k].mean() for k in keys]
        sems  = [patch_data[k].std() / np.sqrt(N) for k in keys]
        recs  = [recovery(m, ar_mean, en_mean) for m in means]
        return means, sems, recs

    single_means, single_sems, single_recs = get_rec_sem(single_keys)
    x_labels = [k.replace("patch_", "") for k in single_keys]
    xs = np.arange(len(single_keys))

    # ── Clip SE band to shared y range to avoid dominating panels ──
    gap = abs(en_mean - ar_mean)
    se_rec = [100 * s / gap for s in single_sems]
    band_lo = [max(r - s, SHARED_Y_MIN) for r, s in zip(single_recs, se_rec)]
    band_hi = [min(r + s, SHARED_Y_MAX) for r, s in zip(single_recs, se_rec)]

    # ── Main recovery line ──
    ax.plot(xs, single_recs, color=LINE_COLOR, linewidth=2.2,
            marker="o", markersize=6, zorder=3, label="Single-layer recovery %")
    ax.fill_between(xs, band_lo, band_hi,
                    color=LINE_COLOR, alpha=0.10, zorder=2)  # reduced alpha

    # ── Reference lines ──
    ax.axhline(100, color=EN_COLOR, linewidth=1.8, linestyle="--",
               alpha=0.85, label="English baseline (100%)")
    ax.axhline(0,   color=AR_COLOR, linewidth=1.8, linestyle="--",
               alpha=0.85, label="Arabic baseline (0%)")

    # ── Span patch annotations ──
    if span_keys:
        span_means, span_sems, span_recs = get_rec_sem(span_keys)
        for span_idx, (k, r) in enumerate(zip(span_keys, span_recs)):
            rounded_r = round(r)
            label = k.replace("patch_", "")
            # Stagger span labels so near-identical span patches do not overlap.
            x_text = max(xs[-1] - 2 - span_idx * 0.75, 0)
            y_text = max(r + 12 + span_idx * 14, SHARED_Y_MIN + 15)
            y_text = min(y_text, SHARED_Y_MAX - 10)
            ax.annotate(
                f"{label}: {rounded_r}%",
                xy=(xs[-1], max(r, SHARED_Y_MIN + 5)),
                xytext=(x_text, y_text),
                fontsize=10, color="#555555",
                arrowprops=dict(arrowstyle="->", color="#aaaaaa", lw=0.8),
            )

    # ── Over-recovery annotation (values clipped by shared y-axis) ──
    peak_rec = max(single_recs)
    if peak_rec > SHARED_Y_MAX:
        peak_idx = single_recs.index(peak_rec)
        ax.annotate(
            f"peak: {peak_rec:.0f}%",
            xy=(peak_idx, SHARED_Y_MAX),
            xytext=(peak_idx + 0.4, SHARED_Y_MAX - 12),
            fontsize=10, color=LINE_COLOR, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=LINE_COLOR, lw=1.0),
        )

    # ── Threshold annotation: first layer ≥ 80%, placed inside plot ──
    thresh_idx = next((i for i, r in enumerate(single_recs) if r >= 80), None)
    if thresh_idx is not None:
        ax.axvline(thresh_idx, color="#999999", linewidth=1.0,
                   linestyle=":", alpha=0.7)
        # Place label near top of panel, just right of the line
        ax.text(thresh_idx + 0.15, SHARED_Y_MAX - 12,
                f"{x_labels[thresh_idx]}\n≥80%",
                fontsize=9, color="#555555", va="top")

    # ── X-axis ticks: thin out if more than 10 labels ──
    if len(xs) > 10:
        step = 2
        tick_positions = xs[::step]
        tick_labels    = x_labels[::step]
    else:
        tick_positions = xs
        tick_labels    = x_labels

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=11, rotation=45, ha="right")
    ax.tick_params(axis="y", labelsize=11)
    ax.set_title(model_name, fontsize=15, fontweight="bold", pad=8)

    # ── Shared y-axis range ──
    ax.set_ylim(SHARED_Y_MIN, SHARED_Y_MAX)

    if show_ylabel:
        ax.set_ylabel("Recovery (% of En–Ar gap)", fontsize=15)
    if show_xlabel:
        ax.set_xlabel("Patch layer", fontsize=15)

    ax.grid(axis="y", alpha=0.18)
    ax.spines[["top", "right"]].set_visible(False)

# ── Build 2×2 panel ────────────────────────────────────────────────────────────
loaded = [(name, load_npz(d)) for name, d in MODELS]

fig = plt.figure(figsize=(max(7 * len(MODELS), 10), 7))
gs  = gridspec.GridSpec(1, len(MODELS), figure=fig, wspace=0.32)

axes = []
for idx, (model_name, data) in enumerate(loaded):
    ax = fig.add_subplot(gs[0, idx])
    axes.append(ax)
    show_ylabel = (idx == 0)
    show_xlabel = True
    draw_panel(ax, data, model_name, show_ylabel, show_xlabel)

# ── Shared legend ──────────────────────────────────────────────────────────────
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

legend_handles = [
    Line2D([0], [0], color=LINE_COLOR, linewidth=2.2, marker="o",
           markersize=7, label="Single-layer recovery %"),
    Line2D([0], [0], color=EN_COLOR, linewidth=1.8, linestyle="--",
           label="English baseline (100%)"),
    Line2D([0], [0], color=AR_COLOR, linewidth=1.8, linestyle="--",
           label="Arabic baseline (0%)"),
]

fig.legend(handles=legend_handles, loc="lower center",
           ncol=3, fontsize=17, frameon=False,      # increased legend fontsize
           handlelength=1.6, handletextpad=0.6, columnspacing=1.2,
           bbox_to_anchor=(0.5, -0.08))

fig.suptitle(
    "Causal Activation Patching: English → Arabic Hidden-State Injection",
    fontsize=18, fontweight="bold", y=0.98
)

for ext in [".pdf", ".png"]:
    out_path = os.path.join(args.out_dir, f"fig_patching_panel_lines{ext}")
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"Saved → {out_path}")

plt.close()
print("Done.")
