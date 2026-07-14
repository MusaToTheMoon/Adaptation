import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


AR_COLOR = "#0D3349"
EN_COLOR = "#2A9D8F"
LINE_COLOR = "#C0392B"
SHARED_Y_MIN = -30
SHARED_Y_MAX = 130


def patch_sort_key(key):
    vals = []
    for part in key.replace("patch_", "").split("_"):
        if part.startswith("L"):
            vals.append(int(part[1:]))
    return vals


def split_patch_keys(files):
    patch_keys = [k for k in files if k not in {"en_base_prob", "ar_base_prob", "gt_letters"}]
    patch_keys = sorted(patch_keys, key=patch_sort_key)
    single = [k for k in patch_keys if "_" not in k.replace("patch_", "")]
    span = [k for k in patch_keys if k not in single]
    return single, span


def degradation(val, ar_mean, en_mean):
    gap = en_mean - ar_mean
    return 100 * (en_mean - val) / gap if gap > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--model_label", default="Model")
    parser.add_argument("--npz_name", default="patching_results_reverse.npz")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    npz_path = os.path.join(args.patch_dir, args.npz_name)
    d = np.load(npz_path, allow_pickle=True)
    en_base = d["en_base_prob"]
    ar_base = d["ar_base_prob"]
    n = len(en_base)
    patch_data = {k: d[k] for k in d.files if k not in {"en_base_prob", "ar_base_prob", "gt_letters"}}
    single_keys, span_keys = split_patch_keys(d.files)
    if not single_keys:
        raise RuntimeError(f"No single-layer patch keys found in {npz_path}")

    ar_mean = float(ar_base.mean())
    en_mean = float(en_base.mean())
    gap = abs(en_mean - ar_mean)

    single_means = [float(patch_data[k].mean()) for k in single_keys]
    single_sems = [float(patch_data[k].std() / np.sqrt(n)) for k in single_keys]
    single_degs = [degradation(m, ar_mean, en_mean) for m in single_means]
    se_deg = [100 * s / gap if gap > 0 else 0.0 for s in single_sems]
    band_lo = [max(dg - s, SHARED_Y_MIN) for dg, s in zip(single_degs, se_deg)]
    band_hi = [min(dg + s, SHARED_Y_MAX) for dg, s in zip(single_degs, se_deg)]

    x_labels = [k.replace("patch_", "") for k in single_keys]
    xs = np.arange(len(single_keys))

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(xs, single_degs, color=LINE_COLOR, linewidth=2.2, marker="o", markersize=6, zorder=3)
    ax.fill_between(xs, band_lo, band_hi, color=LINE_COLOR, alpha=0.10, zorder=2)
    ax.axhline(100, color=AR_COLOR, linewidth=1.8, linestyle="--", alpha=0.85, label="Arabic baseline (100% degradation)")
    ax.axhline(0, color=EN_COLOR, linewidth=1.8, linestyle="--", alpha=0.85, label="English baseline (0% degradation)")

    if span_keys:
        span_means = [float(patch_data[k].mean()) for k in span_keys]
        span_degs = [degradation(m, ar_mean, en_mean) for m in span_means]
        for key, deg in zip(span_keys, span_degs):
            if abs(deg) < 5:
                continue
            label = key.replace("patch_", "")
            y_text = max(deg + 12, SHARED_Y_MIN + 15)
            y_text = min(y_text, SHARED_Y_MAX - 10)
            ax.annotate(
                f"{label}: {deg:.0f}%",
                xy=(xs[-1], max(deg, SHARED_Y_MIN + 5)),
                xytext=(max(xs[-1] - 2, 0), y_text),
                fontsize=10,
                color="#555555",
                arrowprops=dict(arrowstyle="->", color="#aaaaaa", lw=0.8),
            )

    thresh_idx = next((i for i, deg in enumerate(single_degs) if deg >= 80), None)
    if thresh_idx is not None:
        ax.axvline(thresh_idx, color="#999999", linewidth=1.0, linestyle=":", alpha=0.7)
        ax.text(thresh_idx + 0.15, SHARED_Y_MAX - 12, f"{x_labels[thresh_idx]}\n>=80%", fontsize=9, color="#555555", va="top")

    tick_positions = xs[::2] if len(xs) > 10 else xs
    tick_labels = x_labels[::2] if len(xs) > 10 else x_labels
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=11, rotation=45, ha="right")
    ax.set_ylim(SHARED_Y_MIN, SHARED_Y_MAX)
    ax.set_ylabel("Degradation (% of En-Ar gap)", fontsize=13)
    ax.set_xlabel("Patch layer", fontsize=13)
    ax.set_title(
        f"Reverse Activation Patching: Arabic -> English Hidden-State Injection\n{args.model_label}",
        fontsize=13,
        fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=10, frameon=False, loc="upper right")

    for ext in [".pdf", ".png"]:
        out_path = os.path.join(args.out_dir, f"fig_patching_reverse_lines{ext}")
        plt.savefig(out_path, bbox_inches="tight", dpi=150)
        print(f"Saved -> {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
