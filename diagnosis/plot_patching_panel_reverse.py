"""
plot_patching_panel_reverse.py
--------------------------------
2×2 panel figure from REVERSE activation patching results for all four models.
Reads patching_results_reverse.npz from each model's output directory.

Metric: Degradation % — how much Arabic injection pulls English performance
toward the Arabic floor. (0% = no effect, 100% = English drops to Arabic level)

Usage:
  python plot_patching_panel_reverse.py \
      --llama_dir    ./activation_patching_llama_reverse_out \
      --mistral_dir  ./activation_patching_mistral_reverse_out \
      --allam_dir    ./activation_patching_allam_reverse_out \
      --medgemma_dir ./activation_patching_medgemma_reverse_out \
      --out_dir      ./activation_patching_panel_reverse_out
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

parser = argparse.ArgumentParser()
parser.add_argument("--llama_dir",    default="./activation_patching_llama_reverse_out")
parser.add_argument("--mistral_dir",  default="./activation_patching_mistral_reverse_out")
parser.add_argument("--allam_dir",    default="./activation_patching_allam_reverse_out")
parser.add_argument("--medgemma_dir", default="./activation_patching_medgemma_reverse_out")
parser.add_argument("--out_dir",      default="./activation_patching_panel_reverse_out")
args = parser.parse_args()
os.makedirs(args.out_dir, exist_ok=True)

# ── Palette ───────────────────────────────────────────────────────────────────
PALETTE = {
    'Base': '#87CEEB',   # sky blue  — Arabic baseline (right anchor)
    'CoT':  '#2A9D8F',   # teal      — English baseline (left anchor)
    'IR':   '#0D3349',   # navy      — early layers (low degradation)
    'AP':   '#F4C430',   # mango
    'FS':   '#F07C00',   # tangerine
    'MP':   '#C0392B',   # crimson   — late layers (high degradation)
}

AR_COLOR   = PALETTE['Base']
EN_COLOR   = PALETTE['CoT']
ANNO_COLOR = "#555555"

_PATCH_CMAP = LinearSegmentedColormap.from_list(
    "patch_grad",
    [PALETTE['IR'], PALETTE['AP'], PALETTE['FS'], PALETTE['MP']],
)

def patch_colors(n):
    if n == 1:
        return [PALETTE['MP']]
    return [_PATCH_CMAP(i / (n - 1)) for i in range(n)]

# ── Model registry ─────────────────────────────────────────────────────────────
MODELS = [
    ("Llama-3.3-70B",          args.llama_dir),
    ("Mistral-Small-3.2-24B",  args.mistral_dir),
    ("ALLaM-7B",               args.allam_dir),
    ("MedGemma-27B",           args.medgemma_dir),
]

# ── Load helper ────────────────────────────────────────────────────────────────
def load_npz(out_dir):
    path = os.path.join(out_dir, "patching_results_reverse.npz")
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

# ── Degradation helper ─────────────────────────────────────────────────────────
def degradation(val, ar_mean, en_mean):
    gap = en_mean - ar_mean
    return 100 * (en_mean - val) / gap if gap > 0 else 0.0

# ── Draw one subplot ───────────────────────────────────────────────────────────
def draw_panel(ax, data, model_name, show_ylabel, show_xlabel):
    if data is None:
        ax.set_facecolor("#f5f5f5")
        ax.text(0.5, 0.5, "Pending", ha="center", va="center",
                fontsize=15, color="#aaaaaa", transform=ax.transAxes)
        ax.set_title(model_name, fontsize=15, fontweight="bold", pad=8)
        ax.axis("off")
        return

    en_base    = data["en_base"]
    ar_base    = data["ar_base"]
    patch_data = data["patch_data"]
    N          = data["N"]

    ar_mean = ar_base.mean()
    en_mean = en_base.mean()

    patch_keys  = list(patch_data.keys())
    patch_means = [patch_data[k].mean() for k in patch_keys]
    patch_sems  = [patch_data[k].std() / np.sqrt(N) for k in patch_keys]

    # Order: English (left) → patches → Arabic (right)
    all_means = [en_mean]  + patch_means  + [ar_mean]
    all_sems  = [en_base.std()/np.sqrt(N)] + patch_sems + [ar_base.std()/np.sqrt(N)]

    n_patches = len(patch_keys)
    colors    = [EN_COLOR] + list(patch_colors(n_patches)) + [AR_COLOR]

    x_labels = (["En\n(base)"] +
                [k.replace("patch_", "") for k in patch_keys] +
                ["Ar\n(base)"])

    xs = np.arange(len(all_means))
    bars = ax.bar(xs, all_means, color=colors,
                  yerr=all_sems, capsize=3, alpha=0.88,
                  edgecolor="white", linewidth=0.4)

    # Value labels above bars
    for bar, m in zip(bars, all_means):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + max(all_means)*0.02,
                f"{m:.2f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold")

    # Degradation % below patch bars
    y_ann = -max(all_means) * 0.07
    ax.text(xs[0], y_ann, "Deg.:", ha="center", va="top",
            fontsize=9, color=ANNO_COLOR)
    for ci, m in enumerate(patch_means):
        ax.text(xs[ci + 1], y_ann,
                f"{degradation(m, ar_mean, en_mean):.0f}%",
                ha="center", va="top", fontsize=9, color=ANNO_COLOR)

    ax.axhline(en_mean, color=EN_COLOR, linewidth=1.2, linestyle="--", alpha=0.7)
    ax.axhline(ar_mean, color=AR_COLOR, linewidth=1.2, linestyle="--", alpha=0.7)

    ax.set_xticks(xs)
    ax.set_xticklabels(x_labels, fontsize=11, rotation=0)
    ax.tick_params(axis="y", labelsize=11)
    ax.set_ylim(y_ann * 1.6, max(all_means) * 1.3)
    ax.set_title(model_name, fontsize=15, fontweight="bold", pad=8)

    if show_ylabel:
        ax.set_ylabel("Mean P(correct letter)", fontsize=14)
    if show_xlabel:
        ax.set_xlabel("Patch config", fontsize=14)

    ax.grid(axis="y", alpha=0.18)
    ax.spines[["top", "right"]].set_visible(False)

# ── Build 2×2 panel ────────────────────────────────────────────────────────────
loaded = [(name, load_npz(d)) for name, d in MODELS]

fig = plt.figure(figsize=(22, 13))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.22, wspace=0.28)

positions = [(0,0), (0,1), (1,0), (1,1)]
for idx, ((model_name, data), (row, col)) in enumerate(zip(loaded, positions)):
    ax = fig.add_subplot(gs[row, col])
    draw_panel(ax, data, model_name,
               show_ylabel=(col == 0),
               show_xlabel=(row == 1))

# ── Shared legend ──────────────────────────────────────────────────────────────
from matplotlib.patches import Patch
_grad_handles = [
    Patch(facecolor=_PATCH_CMAP(v), edgecolor="none",
          label="" if i > 0 else "Patch layers (early → late)")
    for i, v in enumerate(np.linspace(0, 1, 8))
]
legend_handles = [
    Patch(facecolor=EN_COLOR, label="English baseline"),
    Patch(facecolor=AR_COLOR, label="Arabic baseline"),
] + _grad_handles

fig.legend(handles=legend_handles, loc="lower center",
           ncol=len(legend_handles), fontsize=13, frameon=False,
           handlelength=1.2, handletextpad=0.5, columnspacing=0.8,
           bbox_to_anchor=(0.5, 0.02))

fig.suptitle(
    "Reverse Activation Patching: Arabic → English Hidden-State Injection\n",
    fontsize=16, fontweight="bold", y=0.95
)

for ext in [".pdf", ".png"]:
    out_path = os.path.join(args.out_dir, f"fig_patching_panel_reverse{ext}")
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"Saved → {out_path}")

plt.close()
print("Done.")