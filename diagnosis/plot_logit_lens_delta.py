"""
plot_logit_lens_delta.py
------------------------
Replot logit lens results from saved .npz — no model reloading needed.

Produces THREE figures:
  1. fig_logit_lens_quadrant_zoom  — the main 4-condition zoomed plot (L15-L39)
  2. fig_logit_lens_delta          — lift over base AR (Full LoRA - Base AR, Targeted - Base AR)
                                     this directly answers "where does each adapter help and by how much"
  3. fig_logit_lens_full           — full L0-L39 view

Usage:
  python plot_logit_lens_delta.py --npz ./logit_lens_lora_out/logit_lens_results.npz \
                                  --out_dir ./logit_lens_lora_out
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

parser = argparse.ArgumentParser()
parser.add_argument("--npz",      required=True,
                    help="Path to logit_lens_results_<quadrant>.npz")
parser.add_argument("--out_dir",  default="./logit_lens_lora_out")
parser.add_argument("--quadrant", default="access_gap",
                    help="Label for plot titles and output filenames")
args = parser.parse_args()
os.makedirs(args.out_dir, exist_ok=True)

d = np.load(args.npz)
base_en      = d["base_en"]       # (N, 41)
base_ar      = d["base_ar"]
targeted_ar  = d["targeted_ar"]
full_ar      = d["full_ar"]

N, n_layers = base_en.shape
PLOT_END = n_layers - 1   # drop L40 (format artifact)

PALETTE = {
    "base_en":  "#2A9D8F",
    "base_ar":  "#87CEEB",
    "targeted": "#C0392B",
    "full":     "#F07C00",
    "window":   "#F4C430",
    "delta_t":  "#C0392B",
    "delta_f":  "#F07C00",
    "zero":     "#888888",
}

def sem(arr):
    return arr.std(axis=0) / np.sqrt(arr.shape[0])

def _s(arr):
    """Slice to drop final layer."""
    return arr[:, :PLOT_END]

lx = np.arange(PLOT_END)

# ── Figure 1: Zoomed main plot L15-L39 ───────────────────────────────────────────
zoom_s, zoom_e = 15, PLOT_END
lx_z = lx[zoom_s:zoom_e]

fig, ax = plt.subplots(figsize=(10, 5))
for arr, color, label, ls, lw in [
    (base_en,     PALETTE["base_en"],  "Base EN (upper bound)",      "--", 2.0),
    (base_ar,     PALETTE["base_ar"],  "Base AR (lower bound)",      "--", 2.0),
    (targeted_ar, PALETTE["targeted"], "Targeted LoRA L24-40 + AR",  "-",  2.5),
    (full_ar,     PALETTE["full"],     "Full LoRA + AR",             "-",  2.5),
]:
    mu = _s(arr).mean(0)[zoom_s:zoom_e]
    se = sem(_s(arr))[zoom_s:zoom_e]
    ax.plot(lx_z, mu, lw=lw, color=color, ls=ls, label=label)
    ax.fill_between(lx_z, mu - se, mu + se, color=color, alpha=0.15)

ax.axvspan(24, 39, color=PALETTE["window"], alpha=0.12)
ax.axvline(24, color=PALETTE["window"], lw=1.2, ls=":", alpha=0.8)
ax.axvline(39, color=PALETTE["window"], lw=1.2, ls=":", alpha=0.8,
           label="Critical window L24-L39")
ax.set_xlabel("Layer", fontsize=11)
ax.set_ylabel("Mean P(correct answer letter)", fontsize=11)
ax.set_title(
    "Direct Logit Lens: LoRA vs Base Model (L15–L39)\n"
    f"Mistral-Small-3.2-24B · MedAraBench · {args.quadrant}",
    fontsize=12, fontweight="bold"
)
ax.set_xticks(range(zoom_s, zoom_e, 2))
ax.set_xticklabels([f"L{l}" for l in range(zoom_s, zoom_e, 2)], fontsize=9)
ax.legend(fontsize=10, frameon=False)
ax.grid(axis="y", alpha=0.18)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
for ext in [".pdf", ".png"]:
    plt.savefig(os.path.join(args.out_dir, f"fig_logit_lens_zoom_{args.quadrant}{ext}"),
                bbox_inches="tight", dpi=150)
plt.close()
print("Saved fig_logit_lens_zoom")

# ── Figure 2: Delta (lift over base AR) ──────────────────────────────────────────
# This is the key plot: how much does each adapter lift the Arabic trajectory
# at each layer, and WHERE does the lift begin?
delta_targeted = _s(targeted_ar).mean(0) - _s(base_ar).mean(0)
delta_full     = _s(full_ar).mean(0)     - _s(base_ar).mean(0)

# SEM of the difference (assuming independence between conditions)
se_targeted = np.sqrt(sem(_s(targeted_ar))**2 + sem(_s(base_ar))**2)
se_full     = np.sqrt(sem(_s(full_ar))**2     + sem(_s(base_ar))**2)

# Also compute English "ceiling lift" (how much EN is above AR at each layer)
delta_en = _s(base_en).mean(0) - _s(base_ar).mean(0)

fig, ax = plt.subplots(figsize=(12, 5))

# English ceiling
ax.fill_between(lx, np.zeros(PLOT_END), delta_en,
                color=PALETTE["base_en"], alpha=0.08, label="EN ceiling (Base EN − Base AR)")
ax.plot(lx, delta_en, lw=1.5, color=PALETTE["base_en"], ls="--", alpha=0.6)

# LoRA deltas
ax.plot(lx, delta_full,     lw=2.5, color=PALETTE["full"],     label="Full LoRA lift (Full AR − Base AR)")
ax.plot(lx, delta_targeted, lw=2.5, color=PALETTE["targeted"], label="Targeted LoRA lift (Targeted AR − Base AR)")

ax.fill_between(lx,
                delta_full     - se_full,
                delta_full     + se_full,
                color=PALETTE["full"],     alpha=0.15)
ax.fill_between(lx,
                delta_targeted - se_targeted,
                delta_targeted + se_targeted,
                color=PALETTE["targeted"], alpha=0.15)

ax.axhline(0, color=PALETTE["zero"], lw=1.0, ls="-", alpha=0.4)
ax.axvspan(24, 39, color=PALETTE["window"], alpha=0.10)
ax.axvline(24, color=PALETTE["window"], lw=1.2, ls=":", alpha=0.8)
ax.axvline(39, color=PALETTE["window"], lw=1.2, ls=":", alpha=0.8)
ax.text(24.3, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 0 else 0.01,
        "← critical window →", fontsize=8, color="#999900", va="top")

ax.set_xlabel("Layer", fontsize=11)
ax.set_ylabel("ΔP(correct answer letter) over Base AR", fontsize=11)
ax.set_title(
    f"LoRA Representation Lift over Arabic Baseline per Layer\n"
    f"Mistral-Small-3.2-24B · {args.quadrant} · shaded = English ceiling",
    fontsize=12, fontweight="bold"
)
tick_locs = list(range(0, PLOT_END, 4))
ax.set_xticks(tick_locs)
ax.set_xticklabels([f"L{l}" for l in tick_locs], fontsize=8)
ax.legend(fontsize=10, frameon=False)
ax.grid(axis="y", alpha=0.18)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
for ext in [".pdf", ".png"]:
    plt.savefig(os.path.join(args.out_dir, f"fig_logit_lens_delta_{args.quadrant}{ext}"),
                bbox_inches="tight", dpi=150)
plt.close()
print("Saved fig_logit_lens_delta")

# ── Figure 3: Full overview L0-L39 ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5))
for arr, color, label, ls, lw in [
    (base_en,     PALETTE["base_en"],  "Base EN",                    "--", 2.0),
    (base_ar,     PALETTE["base_ar"],  "Base AR",                    "--", 2.0),
    (targeted_ar, PALETTE["targeted"], "Targeted LoRA L24-40 + AR",  "-",  2.5),
    (full_ar,     PALETTE["full"],     "Full LoRA + AR",             "-",  2.5),
]:
    mu = _s(arr).mean(0)
    se_ = sem(_s(arr))
    ax.plot(lx, mu, lw=lw, color=color, ls=ls, label=label)
    ax.fill_between(lx, mu - se_, mu + se_, color=color, alpha=0.12)

ax.axvspan(24, 39, color=PALETTE["window"], alpha=0.10, label="Critical window L24-L39")
ax.axvline(24, color=PALETTE["window"], lw=1.2, ls=":", alpha=0.7)
ax.axvline(39, color=PALETTE["window"], lw=1.2, ls=":", alpha=0.7)
ax.set_xlabel("Layer (0 = embedding output, 1-39 = transformer blocks)", fontsize=11)
ax.set_ylabel("Mean P(correct answer letter)", fontsize=11)
ax.set_title(
    "Direct Logit Lens: Full Layer View (L0–L39)\n"
    f"Mistral-Small-3.2-24B · MedAraBench · {args.quadrant}",
    fontsize=12, fontweight="bold"
)
ax.set_xlim(0, PLOT_END - 1)
tick_locs = list(range(0, PLOT_END, 4))
ax.set_xticks(tick_locs)
ax.set_xticklabels([f"L{l}" for l in tick_locs], fontsize=8)
ax.legend(fontsize=10, frameon=False, loc="upper left")
ax.grid(axis="y", alpha=0.18)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
for ext in [".pdf", ".png"]:
    plt.savefig(os.path.join(args.out_dir, f"fig_logit_lens_full_{args.quadrant}{ext}"),
                bbox_inches="tight", dpi=150)
plt.close()
print("Saved fig_logit_lens_full")

# ── Summary numbers ───────────────────────────────────────────────────────────────
print("\n══ Layer-wise summary (peak region L33-L39) ═══════════════")
print(f"{'Layer':>6}  {'Base EN':>8}  {'Base AR':>8}  {'Tgt LoRA':>9}  {'Full LoRA':>9}  "
      f"{'Δ Tgt/AR':>9}  {'Δ Full/AR':>9}  {'Tgt % ceiling':>14}  {'Full % ceiling':>14}")
for l in range(33, PLOT_END):
    be = base_en[:, l].mean()
    ba = base_ar[:, l].mean()
    tg = targeted_ar[:, l].mean()
    fl = full_ar[:, l].mean()
    ceil = be - ba
    pct_t = 100 * (tg - ba) / ceil if ceil > 0 else 0
    pct_f = 100 * (fl - ba) / ceil if ceil > 0 else 0
    print(f"  L{l:<4}  {be:>8.4f}  {ba:>8.4f}  {tg:>9.4f}  {fl:>9.4f}  "
          f"{tg-ba:>+9.4f}  {fl-ba:>+9.4f}  {pct_t:>13.1f}%  {pct_f:>13.1f}%")

print("\nDone.")