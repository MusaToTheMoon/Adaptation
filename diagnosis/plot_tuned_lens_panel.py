"""
plot_tuned_lens_panel.py
------------------------
Four-panel tuned-lens figure (1 row × 4 columns) for MedAraBench paper.
Reads pre-computed .npz results from each model's output directory.

Layout
------
  [ALLaM 7B] [Mistral 24B] [MedGemma 27B] [Llama 3.3 70B (pending)]

Each panel shows Mean P(correct answer letter) vs layer depth,
broken out by quadrant (color) × language (solid=English, dashed=Arabic).
Shaded bands show ±1 SE.  Late-layer region is highlighted.

Usage
-----
  python plot_tuned_lens_panel.py \
      --allam_npz    ./tuned_lens_allam_out/tuned_lens_results.npz \
      --mistral_npz  ./tuned_lens_mistral_out/tuned_lens_results.npz \
      --medgemma_npz ./tuned_lens_medgemma_out/tuned_lens_results.npz \
      --out_dir      ./tuned_lens_panel
"""

import os, argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--allam_npz",    required=True)
parser.add_argument("--mistral_npz",  required=True)
parser.add_argument("--medgemma_npz", required=True)
parser.add_argument("--llama_npz",    default=None,
                    help="Optional; if not provided a pending placeholder is shown")
parser.add_argument("--out_dir",      default="./tuned_lens_panel")
args = parser.parse_args()
os.makedirs(args.out_dir, exist_ok=True)

# ── Style constants ───────────────────────────────────────────────────────────
QUAD_ORDER  = ["both_correct", "access_gap", "arabic_only", "both_wrong"]
QUAD_COLORS = {
    "both_correct": "#2A9D8F",   # teal
    "access_gap":   "#C0392B",   # crimson
    "arabic_only":  "#F4C430",   # mango
    "both_wrong":   "#0D3349",   # navy
}
QUAD_LABELS = {
    "both_correct": "En✓ Ar✓  (shared knowledge)",
    "access_gap":   "En✓ Ar✗  (access gap)",
    "arabic_only":  "En✗ Ar✓  (Arabic-only)",
    "both_wrong":   "En✗ Ar✗  (both wrong)",
}

# Section-shading: fraction of model depth that counts as "late"
# Shaded to mirror the highlighted zones in the paper's other figures
LATE_FRAC = 0.75   # shade rightmost 25% of layers

# ── Helper ────────────────────────────────────────────────────────────────────
def load_npz(path):
    d = np.load(path, allow_pickle=True)
    return (d["en_probs"], d["ar_probs"],
            d["quadrants"].astype(str), d["layers"])

def plot_model(ax, en_probs, ar_probs, quadrants, layers,
               title, n_layers_total, show_ylabel=False, show_xlabel=True):
    """Draw one panel.

    X-axis uses probe-index positions (0, 1, 2, …) so every probe point
    gets equal horizontal space — this is the standard in mechanistic
    interpretability work (Belrose et al. 2023).  Tick labels still show
    the actual layer numbers so readers can see the true depth.
    """
    n_probes = len(layers)
    xs       = np.arange(n_probes)          # evenly-spaced probe positions

    # Index of the first "late" probe
    late_probe_idx = int(n_probes * LATE_FRAC)

    # ── Background shading ──────────────────────────────────────────────────
    ax.axvspan(-0.5,                    late_probe_idx - 0.5,
               color="#EAF4FF", alpha=0.55, zorder=0)
    ax.axvspan(late_probe_idx - 0.5,    n_probes - 0.5,
               color="#FFF3E8", alpha=0.65, zorder=0)

    # ── Quadrant lines ───────────────────────────────────────────────────────
    for q in QUAD_ORDER:
        mask = quadrants == q
        if not mask.any():
            continue
        ev = en_probs[mask].mean(axis=0)
        av = ar_probs[mask].mean(axis=0)
        ee = en_probs[mask].std(axis=0) / np.sqrt(mask.sum())
        ae = ar_probs[mask].std(axis=0) / np.sqrt(mask.sum())
        c  = QUAD_COLORS[q]

        ax.plot(xs, ev, color=c, lw=2.6, ls="-",  marker="o",
                ms=6.5, zorder=3)
        ax.fill_between(xs, ev - ee, ev + ee,
                        color=c, alpha=0.13, zorder=2)
        ax.plot(xs, av, color=c, lw=2.6, ls="--", marker="^",
                ms=6.5, zorder=3)
        ax.fill_between(xs, av - ae, av + ae,
                        color=c, alpha=0.08, zorder=2)

    # ── Zone labels ──────────────────────────────────────────────────────────
    ax.text((0 + late_probe_idx - 0.5) / 2, 0.97,
            "Early / Mid",
            transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=9,
            color="#5B8DB8", style="italic")
    ax.text((late_probe_idx - 0.5 + n_probes - 0.5) / 2, 0.97,
            "Late",
            transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=9,
            color="#C27035", style="italic", fontweight="bold")

    # ── Axes decoration ───────────────────────────────────────────────────────
    ax.set_title(title, fontsize=17, fontweight="bold", pad=8)
    if show_ylabel:
        ax.set_ylabel("Mean P(correct answer)", fontsize=15)
    else:
        ax.set_ylabel("")
    if show_xlabel:
        ax.set_xlabel("Layer (probe index = equal spacing)", fontsize=15)

    # Thin out tick labels so they don't crowd each other.
    # Always show first and last; for the rest show every `step`-th probe.
    step = max(1, n_probes // 8)   # aim for ~8 visible labels per panel
    ax.set_xticks(xs)
    tick_labels = []
    for idx, l in enumerate(layers):
        if idx == 0:
            tick_labels.append(f"L0\n(emb)")
        elif idx == n_probes - 1:
            tick_labels.append(f"L{l}\n(final)")
        elif idx % step == 0:
            tick_labels.append(f"L{l}")
        else:
            tick_labels.append("")          # blank — tick mark present, no label
    ax.set_xticklabels(tick_labels, fontsize=11)
    ax.set_xlim(-0.5, n_probes - 0.5)

    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.22, lw=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=11)


def pending_panel(ax, title):
    """Narrow placeholder panel for a model whose results aren't ready yet."""
    ax.set_facecolor("#F0F0F0")
    # Rotate the text 90° so it reads vertically in the narrow stub
    ax.text(0.5, 0.5, "Pending",
            ha="center", va="center", fontsize=8.5,
            color="#999999", style="italic",
            rotation=90,
            transform=ax.transAxes)
    ax.set_title(title, fontsize=9, fontweight="bold", pad=5,
                 color="#777777")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading results ...")

# Width ratios: ready models get 3 units, pending models get 1 unit (narrow)
MODEL_DEFS = [
    ("ALLaM 7B",          args.allam_npz),
    ("Mistral-Small 24B", args.mistral_npz),
    ("MedGemma 27B",      args.medgemma_npz),
    ("Llama 3.3 70B",     args.llama_npz),
]

n_layers_map = {
    "ALLaM 7B":          32,
    "Mistral-Small 24B": 40,
    "MedGemma 27B":      62,
    "Llama 3.3 70B":     80,
}

loaded = []
for name, path in MODEL_DEFS:
    if path and os.path.exists(path):
        en_p, ar_p, quads, layers = load_npz(path)
        loaded.append((name, en_p, ar_p, quads, layers, True))
        print(f"  ✓ {name}  ({len(layers)} probe layers, {len(quads)} questions)")
    else:
        loaded.append((name, None, None, None, None, False))
        print(f"  ⚠ {name}  — no npz found, will show narrow placeholder")

n_ready   = sum(1 for *_, ok in loaded if ok)
n_pending = len(loaded) - n_ready

# ── Figure ────────────────────────────────────────────────────────────────────
print(f"\nBuilding figure ({n_ready} ready, {n_pending} pending) ...")

# Assign column widths: ready panels = 3, pending panels = 0.8
width_ratios = [3 if ok else 0.8 for *_, ok in loaded]

# Total figure width scales with ready panels (each ~5.5 in) + narrow stubs
fig_w = n_ready * 7.0 + n_pending * 1.1 + 1.0
fig   = plt.figure(figsize=(fig_w, 7.0))
gs    = GridSpec(1, 4, figure=fig, wspace=0.22,
                 width_ratios=width_ratios)

axes = [fig.add_subplot(gs[0, i]) for i in range(4)]

# Track which column index the first ready panel is (for ylabel)
first_ready_idx = next((i for i, (*_, ok) in enumerate(loaded) if ok), 0)

for i, (name, en_p, ar_p, quads, layers, ok) in enumerate(loaded):
    ax = axes[i]
    if ok:
        plot_model(ax, en_p, ar_p, quads, layers,
                   title=name,
                   n_layers_total=n_layers_map.get(name, layers[-1]),
                   show_ylabel=(i == first_ready_idx),
                   show_xlabel=True)
    else:
        pending_panel(ax, name)

# ── Shared legend ─────────────────────────────────────────────────────────────
quad_handles = [
    mlines.Line2D([], [], color=QUAD_COLORS[q], lw=3.0,
                  label=QUAD_LABELS[q])
    for q in QUAD_ORDER
]
style_handles = [
    mlines.Line2D([], [], color="dimgrey", lw=3.0, ls="-",
                  marker="o", ms=8, label="English"),
    mlines.Line2D([], [], color="dimgrey", lw=3.0, ls="--",
                  marker="^", ms=8, label="Arabic"),
]
zone_handles = [
    mpatches.Patch(facecolor="#EAF4FF", alpha=0.8, label="Early / Mid layers"),
    mpatches.Patch(facecolor="#FFF3E8", alpha=0.8, label="Late layers"),
]

# Anchor legend only under the ready-panel region (skip the pending stubs)
# Compute the x-centre of just the ready columns in figure coordinates
ready_cols   = [i for i, (*_, ok) in enumerate(loaded) if ok]
total_w      = sum(width_ratios)
# cumulative left edges of each column (fraction of total width, ignoring wspace)
col_left = []
acc = 0
for w in width_ratios:
    col_left.append(acc / total_w)
    acc += w
col_right = [(col_left[i] + width_ratios[i] / total_w) for i in range(4)]
legend_x = (col_left[ready_cols[0]] + col_right[ready_cols[-1]]) / 2

fig.legend(
    handles=quad_handles + style_handles + zone_handles,
    loc="lower center",
    ncol=5,
    fontsize=17,
    frameon=True,
    fancybox=False,
    edgecolor="#cccccc",
    bbox_to_anchor=(legend_x, -0.21),
    handlelength=2.6,
    handleheight=1.4,
    columnspacing=2.0,
    borderpad=0.8,
)

fig.suptitle(
    "Tuned Lens: Correct Answer Probability Across Layers ",
    fontsize=18, fontweight="bold", y=1.0
)

plt.tight_layout(rect=[0, 0.16, 1, 1])

# ── Save ──────────────────────────────────────────────────────────────────────
for ext in [".pdf", ".png"]:
    out = os.path.join(args.out_dir, f"fig_tuned_lens_panel{ext}")
    plt.savefig(out, bbox_inches="tight", dpi=180)
    print(f"Saved → {out}")
plt.close()
print("Done.")