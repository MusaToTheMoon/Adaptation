"""
plot_patching_panel_reverse_lines.py
-------------------------------------
1×4 panel figure from REVERSE activation patching results — LINE PLOT VERSION.
Metric: Degradation % — how much Arabic injection pulls English performance
toward the Arabic floor.
(0% = no effect, 100% = English drops to Arabic level)

Usage:
  python plot_patching_panel_reverse_lines.py \
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
from matplotlib.lines import Line2D

parser = argparse.ArgumentParser()
parser.add_argument("--model", action="append", default=[],
                    help="Repeatable 'Label=directory' entry. Overrides built-in model args when provided.")
parser.add_argument("--gemma_dir",   default="./diagnosis/activation_patching_gemma3_27b_it_reverse_out")
parser.add_argument("--mistral7b_dir", default="./diagnosis/activation_patching_mistral_7b_reverse_out")
parser.add_argument("--llama8b_dir", default="./diagnosis/activation_patching_llama31_8b_inst_reverse_out")
parser.add_argument("--falcon_dir",  default="./diagnosis/activation_patching_falcon_reverse_out")
parser.add_argument("--fanar_dir",   default="./diagnosis/activation_patching_fanar_reverse_out")
parser.add_argument("--llama_dir",    default="./activation_patching_llama_reverse_out")
parser.add_argument("--mistral_dir",  default="./activation_patching_mistral_reverse_out")
parser.add_argument("--allam_dir",    default="./activation_patching_allam_reverse_out")
parser.add_argument("--medgemma_dir", default="./activation_patching_medgemma_reverse_out")
parser.add_argument("--out_dir",      default="./activation_patching_panel_reverse_out")
args = parser.parse_args()
os.makedirs(args.out_dir, exist_ok=True)

# ── Palette (consistent with forward patching panel) ───────────────────────────
PALETTE = {
    'Base': '#0D3349',   # navy     — Arabic baseline
    'CoT':  '#2A9D8F',   # teal     — English baseline
    'line': '#C0392B',   # crimson  — degradation line
}

AR_COLOR   = PALETTE['Base']
EN_COLOR   = PALETTE['CoT']
LINE_COLOR = PALETTE['line']

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

# ── Shared y-axis range ────────────────────────────────────────────────────────
SHARED_Y_MIN = -30
SHARED_Y_MAX = 130

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

    def get_deg_sem(keys):
        means = [patch_data[k].mean() for k in keys]
        sems  = [patch_data[k].std() / np.sqrt(N) for k in keys]
        degs  = [degradation(m, ar_mean, en_mean) for m in means]
        return means, sems, degs

    single_means, single_sems, single_degs = get_deg_sem(single_keys)
    x_labels = [k.replace("patch_", "") for k in single_keys]
    xs = np.arange(len(single_keys))

    # ── Clip SE band ──
    gap = abs(en_mean - ar_mean)
    se_deg = [100 * s / gap for s in single_sems]
    band_lo = [max(d - s, SHARED_Y_MIN) for d, s in zip(single_degs, se_deg)]
    band_hi = [min(d + s, SHARED_Y_MAX) for d, s in zip(single_degs, se_deg)]

    # ── Main degradation line ──
    ax.plot(xs, single_degs, color=LINE_COLOR, linewidth=2.2,
            marker="o", markersize=6, zorder=3)
    ax.fill_between(xs, band_lo, band_hi,
                    color=LINE_COLOR, alpha=0.10, zorder=2)

    # ── Reference lines ──
    ax.axhline(100, color=AR_COLOR, linewidth=1.8, linestyle="--",
               alpha=0.85, label="Arabic baseline (100% degradation)")
    ax.axhline(0,   color=EN_COLOR, linewidth=1.8, linestyle="--",
               alpha=0.85, label="English baseline (0% degradation)")

    # ── Span patch annotations: only non-trivial ones ──
    if span_keys:
        _, _, span_degs = get_deg_sem(span_keys)
        for k, d in zip(span_keys, span_degs):
            if abs(d) < 5:
                continue
            label = k.replace("patch_", "")
            y_text = max(d + 12, SHARED_Y_MIN + 15)
            y_text = min(y_text, SHARED_Y_MAX - 10)
            ax.annotate(
                f"{label}: {d:.0f}%",
                xy=(xs[-1], max(d, SHARED_Y_MIN + 5)),
                xytext=(max(xs[-1] - 2, 0), y_text),
                fontsize=10, color="#555555",
                arrowprops=dict(arrowstyle="->", color="#aaaaaa", lw=0.8),
            )

    # ── Threshold annotation: first layer ≥ 80% degradation ──
    thresh_idx = next((i for i, d in enumerate(single_degs) if d >= 80), None)
    if thresh_idx is not None:
        ax.axvline(thresh_idx, color="#999999", linewidth=1.0,
                   linestyle=":", alpha=0.7)
        ax.text(thresh_idx + 0.15, SHARED_Y_MAX - 12,
                f"{x_labels[thresh_idx]}\n≥80%",
                fontsize=9, color="#555555", va="top")

    # ── X-axis ticks: thin out if more than 10 labels ──
    if len(xs) > 10:
        tick_positions = xs[::2]
        tick_labels    = x_labels[::2]
    else:
        tick_positions = xs
        tick_labels    = x_labels

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=11, rotation=45, ha="right")
    ax.tick_params(axis="y", labelsize=11)
    ax.set_title(model_name, fontsize=17, fontweight="bold", pad=8)
    ax.set_ylim(SHARED_Y_MIN, SHARED_Y_MAX)

    if show_ylabel:
        ax.set_ylabel("Degradation (% of En–Ar gap)", fontsize=15)
    if show_xlabel:
        ax.set_xlabel("Patch layer", fontsize=15)

    ax.grid(axis="y", alpha=0.18)
    ax.spines[["top", "right"]].set_visible(False)

# ── Build 1×4 panel ────────────────────────────────────────────────────────────
loaded = [(name, load_npz(d)) for name, d in MODELS]

fig = plt.figure(figsize=(max(7 * len(MODELS), 10), 7))
gs  = gridspec.GridSpec(1, len(MODELS), figure=fig, wspace=0.32)

for idx, (model_name, data) in enumerate(loaded):
    ax = fig.add_subplot(gs[0, idx])
    draw_panel(ax, data, model_name,
               show_ylabel=(idx == 0),
               show_xlabel=True)

# ── Shared legend ──────────────────────────────────────────────────────────────
legend_handles = [
    Line2D([0], [0], color=LINE_COLOR, linewidth=2.2, marker="o",
           markersize=7, label="Single-layer degradation %"),
    Line2D([0], [0], color=EN_COLOR, linewidth=1.8, linestyle="--",
           label="English baseline (0%)"),
    Line2D([0], [0], color=AR_COLOR, linewidth=1.8, linestyle="--",
           label="Arabic baseline (100%)"),
]

fig.legend(handles=legend_handles, loc="lower center",
           ncol=3, fontsize=17, frameon=False,
           handlelength=1.6, handletextpad=0.6, columnspacing=1.2,
           bbox_to_anchor=(0.5, -0.08))

fig.suptitle(
    "Reverse Activation Patching: Arabic → English Hidden-State Injection",
    fontsize=18, fontweight="bold", y=0.98
)

for ext in [".pdf", ".png"]:
    out_path = os.path.join(args.out_dir, f"fig_patching_panel_reverse_lines{ext}")
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"Saved → {out_path}")

plt.close()
print("Done.")
