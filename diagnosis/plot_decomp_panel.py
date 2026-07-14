"""
plot_decomp_panel.py
---------------------
1×4 panel figure: Attention vs MLP decomposition for all four models.
Reads decomp_results.npz from each model's output directory.

Usage:
  python plot_decomp_panel.py \
      --llama_dir    ./attn_mlp_decomp_llama_out \
      --mistral_dir  ./attn_mlp_decomp_mistral_out \
      --allam_dir    ./attn_mlp_decomp_allam_out \
      --medgemma_dir ./attn_mlp_decomp_medgemma_out \
      --out_dir      ./attn_mlp_decomp_panel_out
"""

import os, argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

parser = argparse.ArgumentParser()
parser.add_argument("--llama_dir",    default="./attn_mlp_decomp_llama_out")
parser.add_argument("--mistral_dir",  default="./attn_mlp_decomp_mistral_out")
parser.add_argument("--allam_dir",    default="./attn_mlp_decomp_allam_out")
parser.add_argument("--medgemma_dir", default="./attn_mlp_decomp_medgemma_out")
parser.add_argument("--out_dir",      default="./attn_mlp_decomp_panel_out")
args = parser.parse_args()
os.makedirs(args.out_dir, exist_ok=True)

# ── Palette (consistent with all other panels) ─────────────────────────────────
PALETTE = {
    'Base': '#0D3349',   # navy     — Arabic baseline
    'CoT':  '#2A9D8F',   # teal     — English baseline
    'attn': '#C0392B',   # crimson  — attention
    'mlp':  '#F4C430',   # mango    — MLP
    'full': '#555555',   # grey     — full layer reference
}

# ── Shared y-axis ──────────────────────────────────────────────────────────────
SHARED_Y_MIN = -30
SHARED_Y_MAX = 130

# ── Model registry ─────────────────────────────────────────────────────────────
MODELS = [
    ("Llama-3.3-70B",         args.llama_dir),
    ("Mistral-Small-3.2-24B", args.mistral_dir),
    ("ALLaM-7B",              args.allam_dir),
    ("MedGemma-27B",          args.medgemma_dir),
]

# ── Load helper ────────────────────────────────────────────────────────────────
def load_npz(out_dir):
    path = os.path.join(out_dir, "decomp_results.npz")
    if not os.path.exists(path):
        return None
    d = np.load(path, allow_pickle=True)
    en_base = d["en_base_prob"]
    ar_base = d["ar_base_prob"]
    layers  = d["critical_layers"].tolist()
    N       = len(en_base)
    results = {}
    for L in layers:
        for t in ("attn", "mlp", "full"):
            key = f"L{L}_{t}"
            if key in d.files:
                results[(L, t)] = d[key]
    return dict(en_base=en_base, ar_base=ar_base, N=N, layers=layers, results=results)

# ── Recovery helper ────────────────────────────────────────────────────────────
def recovery(val, ar_mean, en_mean):
    gap = en_mean - ar_mean
    return 100 * (val - ar_mean) / gap if gap > 0 else 0.0

# ── Draw one subplot ───────────────────────────────────────────────────────────
def draw_panel(ax, data, model_name, show_ylabel, show_xlabel):
    if data is None:
        ax.set_facecolor("#f5f5f5")
        ax.text(0.5, 0.5, "Pending", ha="center", va="center",
                fontsize=15, color="#aaaaaa", transform=ax.transAxes)
        ax.set_title(model_name, fontsize=17, fontweight="bold", pad=8)
        ax.axis("off")
        return

    ar_mean = data["ar_base"].mean()
    en_mean = data["en_base"].mean()
    N       = data["N"]
    layers  = data["layers"]
    results = data["results"]
    lx      = list(range(len(layers)))

    def rec(L, t):
        return recovery(results[(L, t)].mean(), ar_mean, en_mean)

    def sem(L, t):
        gap = en_mean - ar_mean
        return 100 * results[(L, t)].std() / np.sqrt(N) / gap if gap > 0 else 0.0

    attn_r = [rec(L, 'attn') for L in layers]
    mlp_r  = [rec(L, 'mlp')  for L in layers]
    full_r = [rec(L, 'full') for L in layers]
    attn_e = [sem(L, 'attn') for L in layers]
    mlp_e  = [sem(L, 'mlp')  for L in layers]
    full_e = [sem(L, 'full') for L in layers]

    # ── Lines ──
    ax.plot(lx, attn_r, 'o-', color=PALETTE['attn'], linewidth=2.2,
            markersize=6, zorder=3, label='Attention-only')
    ax.plot(lx, mlp_r,  's-', color=PALETTE['mlp'],  linewidth=2.2,
            markersize=6, zorder=3, label='MLP-only')
    ax.plot(lx, full_r, '^--', color=PALETTE['full'], linewidth=1.8,
            markersize=6, zorder=2, alpha=0.7, label='Full layer (reference)')

    # ── SE bands ──
    band = lambda r, e: (
        [max(a - b, SHARED_Y_MIN) for a, b in zip(r, e)],
        [min(a + b, SHARED_Y_MAX) for a, b in zip(r, e)]
    )
    lo, hi = band(attn_r, attn_e)
    ax.fill_between(lx, lo, hi, color=PALETTE['attn'], alpha=0.10)
    lo, hi = band(mlp_r, mlp_e)
    ax.fill_between(lx, lo, hi, color=PALETTE['mlp'],  alpha=0.10)

    # ── Reference lines ──
    ax.axhline(0,   color=PALETTE['Base'], linewidth=1.8, linestyle='--', alpha=0.85)
    ax.axhline(100, color=PALETTE['CoT'],  linewidth=1.8, linestyle='--', alpha=0.85)

    # ── Axes ──
    ax.set_xticks(lx)
    ax.set_xticklabels([f"L{L}" for L in layers], fontsize=11,
                       rotation=45, ha="right")
    ax.tick_params(axis="y", labelsize=11)
    ax.set_title(model_name, fontsize=17, fontweight="bold", pad=8)
    ax.set_ylim(SHARED_Y_MIN, SHARED_Y_MAX)

    if show_ylabel:
        ax.set_ylabel("Recovery (% of En–Ar gap)", fontsize=15)
    if show_xlabel:
        ax.set_xlabel("Layer", fontsize=15)

    ax.grid(axis="y", alpha=0.18)
    ax.spines[["top", "right"]].set_visible(False)

# ── Build 1×4 panel ────────────────────────────────────────────────────────────
loaded = [(name, load_npz(d)) for name, d in MODELS]

fig = plt.figure(figsize=(28, 7))
gs  = gridspec.GridSpec(1, 4, figure=fig, wspace=0.32)

positions = [(0,0), (0,1), (0,2), (0,3)]
for idx, ((model_name, data), (row, col)) in enumerate(zip(loaded, positions)):
    ax = fig.add_subplot(gs[row, col])
    draw_panel(ax, data, model_name,
               show_ylabel=(col == 0),
               show_xlabel=True)

# ── Shared legend ──────────────────────────────────────────────────────────────
legend_handles = [
    Line2D([0], [0], color=PALETTE['attn'], linewidth=2.2, marker='o',
           markersize=7, label="Attention-only patch"),
    Line2D([0], [0], color=PALETTE['mlp'],  linewidth=2.2, marker='s',
           markersize=7, label="MLP-only patch"),
    Line2D([0], [0], color=PALETTE['full'], linewidth=1.8, marker='^',
           markersize=7, linestyle='--', alpha=0.7, label="Full layer (reference)"),
    Line2D([0], [0], color=PALETTE['CoT'],  linewidth=1.8, linestyle='--',
           alpha=0.85, label="English baseline (100%)"),
    Line2D([0], [0], color=PALETTE['Base'], linewidth=1.8, linestyle='--',
           alpha=0.85, label="Arabic baseline (0%)"),
]

fig.legend(handles=legend_handles, loc="lower center", ncol=5,
           fontsize=17, frameon=False,
           handlelength=1.6, handletextpad=0.6, columnspacing=1.2,
           bbox_to_anchor=(0.5, -0.08))

fig.suptitle(
    "Attention vs MLP Sublayer Decomposition of Activation Patching\n",
    fontsize=18, fontweight="bold", y=1.02
)

for ext in [".pdf", ".png"]:
    out_path = os.path.join(args.out_dir, f"fig_decomp_panel{ext}")
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"Saved → {out_path}")

plt.close()
print("Done.")