"""
plot_logit_lens_panel.py
------------------------
2×2 panel of logit lens delta plots across all four quadrants.
Reads logit_lens_results_<quadrant>.npz from the output directory.

Each panel shows:
  - LoRA lift over Arabic baseline (Full LoRA − Base AR, Targeted − Base AR)
  - English ceiling (Base EN − Base AR) as shaded reference

Usage:
  python plot_logit_lens_panel.py --out_dir ./logit_lens_lora_out
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import matplotlib.lines as mlines

parser = argparse.ArgumentParser()
parser.add_argument("--out_dir", default="./logit_lens_lora_out",
                    help="Directory containing logit_lens_results_<quadrant>.npz files")
parser.add_argument("--zoom_start", type=int, default=15,
                    help="First layer to show in zoomed x-axis")
args = parser.parse_args()

# ── Palette (consistent with tuned-lens and patching panels) ───────────────────
PALETTE = {
    "base_en":  "#2A9D8F",   # teal
    "base_ar":  "#87CEEB",   # sky blue
    "targeted": "#C0392B",   # crimson
    "full":     "#F07C00",   # tangerine
    "window":   "#F4C430",   # mango
    "zero":     "#888888",   # grey
}

QUADRANTS = [
    ("access_gap",   "Access Gap\n(En✓ Ar✗)",    True),
    ("both_correct", "Both Correct\n(En✓ Ar✓)",  False),
    ("both_wrong",   "Both Wrong\n(En✗ Ar✗)",    False),
    ("arabic_only",  "Arabic Only\n(En✗ Ar✓)",   False),
]

def sem(arr):
    return arr.std(axis=0) / np.sqrt(arr.shape[0])

def load(quadrant):
    path = os.path.join(args.out_dir, f"logit_lens_results_{quadrant}.npz")
    if not os.path.exists(path):
        return None
    d = np.load(path)
    return {k: d[k][:, :-1] for k in ["base_en", "base_ar", "targeted_ar", "full_ar"]}

# ── Figure 1: 2×2 Delta Panel ─────────────────────────────────────────────────
fig = plt.figure(figsize=(28, 7))   # same width as patching/tuned-lens panels
gs  = gridspec.GridSpec(1, 4, figure=fig, wspace=0.32)  # 1×4 layout, same wspace

all_data = [(q, label, is_key, load(q)) for q, label, is_key in QUADRANTS]

# Shared y-axis range
all_deltas = []
for _, _, _, data in all_data:
    if data is None: continue
    all_deltas.append(data["full_ar"].mean(0)     - data["base_ar"].mean(0))
    all_deltas.append(data["targeted_ar"].mean(0) - data["base_ar"].mean(0))
    all_deltas.append(data["base_en"].mean(0)     - data["base_ar"].mean(0))

global_max = max(arr.max() for arr in all_deltas) * 1.15
global_min = min(arr.min() for arr in all_deltas) - 0.02
global_min = min(global_min, -0.02)

positions = [(0,0), (0,1), (0,2), (0,3)]

for idx, ((quadrant, label, is_key, data), (row, col)) in enumerate(
        zip(all_data, positions)):

    ax = fig.add_subplot(gs[row, col])

    if data is None:
        ax.set_facecolor("#f5f5f5")
        ax.text(0.5, 0.5, "Pending", ha="center", va="center",
                fontsize=15, color="#aaaaaa", transform=ax.transAxes)
        ax.set_title(label, fontsize=17, fontweight="bold", pad=8)
        ax.axis("off")
        continue

    N, n_layers = data["base_en"].shape
    lx = np.arange(n_layers)
    zoom_s = min(args.zoom_start, n_layers - 5)

    delta_full     = data["full_ar"].mean(0)     - data["base_ar"].mean(0)
    delta_targeted = data["targeted_ar"].mean(0) - data["base_ar"].mean(0)
    delta_en       = data["base_en"].mean(0)     - data["base_ar"].mean(0)

    se_full     = np.sqrt(sem(data["full_ar"])**2     + sem(data["base_ar"])**2)
    se_targeted = np.sqrt(sem(data["targeted_ar"])**2 + sem(data["base_ar"])**2)

    # English ceiling
    ax.fill_between(lx[zoom_s:], np.zeros(n_layers - zoom_s),
                    delta_en[zoom_s:],
                    color=PALETTE["base_en"], alpha=0.10)
    ax.plot(lx[zoom_s:], delta_en[zoom_s:],
            lw=1.8, color=PALETTE["base_en"], ls="--", alpha=0.6)

    # LoRA deltas
    ax.plot(lx[zoom_s:], delta_full[zoom_s:],
            lw=2.2, color=PALETTE["full"],     label="Full LoRA lift")
    ax.plot(lx[zoom_s:], delta_targeted[zoom_s:],
            lw=2.2, color=PALETTE["targeted"], label="Targeted LoRA lift")
    ax.fill_between(lx[zoom_s:],
                    (delta_full - se_full)[zoom_s:],
                    (delta_full + se_full)[zoom_s:],
                    color=PALETTE["full"],     alpha=0.10)
    ax.fill_between(lx[zoom_s:],
                    (delta_targeted - se_targeted)[zoom_s:],
                    (delta_targeted + se_targeted)[zoom_s:],
                    color=PALETTE["targeted"], alpha=0.10)

    ax.axhline(0, color=PALETTE["zero"], lw=0.8, ls="-", alpha=0.5)
    ax.axvspan(24, min(39, n_layers-1), color=PALETTE["window"], alpha=0.10)
    ax.axvline(24, color=PALETTE["window"], lw=1.0, ls=":", alpha=0.7)
    ax.axvline(min(39, n_layers-1), color=PALETTE["window"], lw=1.0, ls=":", alpha=0.7)

    ax.set_ylim(global_min, global_max)
    ax.set_xlim(zoom_s, n_layers - 1)

    # Bold title for key quadrant
    weight = "bold"
    color  = "#C0392B" if is_key else "black"
    ax.set_title(f"{label}  (N={N})", fontsize=17, fontweight=weight,
                 color=color, pad=8)

    if col == 0:
        ax.set_ylabel("ΔP(correct letter) over Base AR", fontsize=15)
    ax.set_xlabel("Layer", fontsize=15)

    # Thin out ticks if many layers
    tick_locs = list(range(zoom_s if zoom_s % 2 == 0 else zoom_s+1, n_layers, 4))
    if len(tick_locs) > 10:
        tick_locs = tick_locs[::2]
    ax.set_xticks(tick_locs)
    ax.set_xticklabels([f"L{l}" for l in tick_locs], fontsize=11,
                       rotation=45, ha="right")
    ax.tick_params(axis="y", labelsize=11)
    ax.grid(axis="y", alpha=0.18)
    ax.spines[["top", "right"]].set_visible(False)

# ── Shared legend ──────────────────────────────────────────────────────────────
legend_handles = [
    mlines.Line2D([], [], color=PALETTE["full"],     lw=2.2,
                  label="Full LoRA lift (Full AR − Base AR)"),
    mlines.Line2D([], [], color=PALETTE["targeted"], lw=2.2,
                  label="Targeted LoRA lift (Targeted AR − Base AR)"),
    mlines.Line2D([], [], color=PALETTE["base_en"],  lw=1.8, ls="--", alpha=0.7,
                  label="English ceiling (Base EN − Base AR)"),
    Patch(facecolor=PALETTE["window"], alpha=0.3, label="Critical window L24--L39"),
]

fig.legend(handles=legend_handles, loc="lower center", ncol=4,
           fontsize=17, frameon=False,
           handlelength=1.6, handletextpad=0.6, columnspacing=1.2,
           bbox_to_anchor=(0.5, -0.08))

fig.suptitle(
    "LoRA Representation Lift over Arabic Baseline — All Quadrants\n"
    "Mistral-Small-3.2-24B · Direct Logit Lens · ΔP(correct answer letter)",
    fontsize=18, fontweight="bold", y=0.98
)

for ext in [".pdf", ".png"]:
    out = os.path.join(args.out_dir, f"fig_logit_lens_panel{ext}")
    plt.savefig(out, bbox_inches="tight", dpi=150)
    print(f"Saved → {out}")
plt.close()

# ── Figure 2: Overlay — all quadrants on one delta plot ───────────────────────
QUADRANT_COLORS = {
    "access_gap":   "#C0392B",
    "both_correct": "#2A9D8F",
    "both_wrong":   "#555555",
    "arabic_only":  "#87CEEB",
}
QUADRANT_LABELS = {
    "access_gap":   "Access Gap (En✓ Ar✗)",
    "both_correct": "Both Correct (En✓ Ar✓)",
    "both_wrong":   "Both Wrong (En✗ Ar✗)",
    "arabic_only":  "Arabic Only (En✗ Ar✓)",
}

fig, axes = plt.subplots(1, 2, figsize=(28, 7), sharey=False)

for ax, lora_key, lora_label in [
    (axes[0], "full_ar",     "Full LoRA"),
    (axes[1], "targeted_ar", "Targeted LoRA L24--40"),
]:
    for quadrant, _, _, data in all_data:
        if data is None: continue
        n_layers = data["base_en"].shape[1]
        lx = np.arange(n_layers)
        zoom_s = min(args.zoom_start, n_layers - 5)

        delta = data[lora_key].mean(0) - data["base_ar"].mean(0)
        se_d  = np.sqrt(sem(data[lora_key])**2 + sem(data["base_ar"])**2)
        color = QUADRANT_COLORS[quadrant]
        lw    = 2.5 if quadrant == "access_gap" else 1.5
        ls    = "-"  if quadrant == "access_gap" else "--"

        ax.plot(lx[zoom_s:], delta[zoom_s:],
                lw=lw, ls=ls, color=color,
                label=QUADRANT_LABELS[quadrant])
        ax.fill_between(lx[zoom_s:],
                        (delta - se_d)[zoom_s:],
                        (delta + se_d)[zoom_s:],
                        color=color, alpha=0.10)

    ax.axhline(0, color=PALETTE["zero"], lw=0.8, alpha=0.5)
    ax.axvspan(24, min(39, n_layers-1), color=PALETTE["window"], alpha=0.10)
    ax.axvline(24, color=PALETTE["window"], lw=1.2, ls=":", alpha=0.8)
    ax.axvline(min(39, n_layers-1), color=PALETTE["window"], lw=1.2, ls=":", alpha=0.8)
    ax.set_xlabel("Layer", fontsize=15)
    ax.set_ylabel("ΔP(correct letter) over Base AR", fontsize=15)
    ax.set_title(lora_label, fontsize=17, fontweight="bold", pad=8)
    ax.set_xlim(zoom_s, n_layers - 1)

    tick_locs = list(range(zoom_s if zoom_s % 2 == 0 else zoom_s+1, n_layers, 2))
    if len(tick_locs) > 10:
        tick_locs = tick_locs[::2]
    ax.set_xticks(tick_locs)
    ax.set_xticklabels([f"L{l}" for l in tick_locs], fontsize=11,
                       rotation=45, ha="right")
    ax.tick_params(axis="y", labelsize=11)
    ax.legend(fontsize=17, frameon=False)
    ax.grid(axis="y", alpha=0.18)
    ax.spines[["top", "right"]].set_visible(False)

fig.suptitle(
    "LoRA Lift Specificity: Access Gap vs Other Quadrants",
    fontsize=18, fontweight="bold", y=0.98
)
plt.tight_layout()

for ext in [".pdf", ".png"]:
    out = os.path.join(args.out_dir, f"fig_logit_lens_specificity{ext}")
    plt.savefig(out, bbox_inches="tight", dpi=150)
    print(f"Saved → {out}")
plt.close()

print("\nDone. Generated:")
print("  fig_logit_lens_panel       — 1×4 delta grid, all quadrants")
print("  fig_logit_lens_specificity — side-by-side full vs targeted, all quadrants overlaid")