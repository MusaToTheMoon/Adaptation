import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


AR_COLOR = "#0D3349"
EN_COLOR = "#2A9D8F"
LINE_COLOR = "#C0392B"
SHARED_Y_MIN = -60
SHARED_Y_MAX = 130


def patch_sort_key(key):
    parts = key.replace("patch_", "").split("_")
    vals = []
    for part in parts:
        if part.startswith("L"):
            vals.append(int(part[1:]))
    return vals


def split_patch_keys(files):
    patch_keys = [k for k in files if k not in {"en_base_prob", "ar_base_prob", "gt_letters"}]
    patch_keys = sorted(patch_keys, key=patch_sort_key)
    single = [k for k in patch_keys if "_" not in k.replace("patch_", "")]
    span = [k for k in patch_keys if k not in single]
    return single, span


def recovery(val, ar_mean, en_mean):
    gap = en_mean - ar_mean
    return 100 * (val - ar_mean) / gap if gap > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--model_label", default="Model")
    parser.add_argument("--npz_name", default="patching_results.npz")
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
    single_recs = [recovery(m, ar_mean, en_mean) for m in single_means]
    se_rec = [100 * s / gap if gap > 0 else 0.0 for s in single_sems]
    band_lo = [max(r - s, SHARED_Y_MIN) for r, s in zip(single_recs, se_rec)]
    band_hi = [min(r + s, SHARED_Y_MAX) for r, s in zip(single_recs, se_rec)]

    x_labels = [k.replace("patch_", "") for k in single_keys]
    xs = np.arange(len(single_keys))

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(xs, single_recs, color=LINE_COLOR, linewidth=2.2, marker="o", markersize=6, zorder=3)
    ax.fill_between(xs, band_lo, band_hi, color=LINE_COLOR, alpha=0.10, zorder=2)
    ax.axhline(100, color=EN_COLOR, linewidth=1.8, linestyle="--", alpha=0.85, label="English baseline (100%)")
    ax.axhline(0, color=AR_COLOR, linewidth=1.8, linestyle="--", alpha=0.85, label="Arabic baseline (0%)")

    if span_keys:
        span_means = [float(patch_data[k].mean()) for k in span_keys]
        span_recs = [recovery(m, ar_mean, en_mean) for m in span_means]
        for span_idx, (key, rec) in enumerate(zip(span_keys, span_recs)):
            label = key.replace("patch_", "")
            x_text = max(xs[-1] - 2 - span_idx * 0.75, 0)
            y_text = max(rec + 12 + span_idx * 14, SHARED_Y_MIN + 15)
            y_text = min(y_text, SHARED_Y_MAX - 10)
            ax.annotate(
                f"{label}: {rec:.0f}%",
                xy=(xs[-1], max(rec, SHARED_Y_MIN + 5)),
                xytext=(x_text, y_text),
                fontsize=10,
                color="#555555",
                arrowprops=dict(arrowstyle="->", color="#aaaaaa", lw=0.8),
            )

    peak_rec = max(single_recs)
    if peak_rec > SHARED_Y_MAX:
        peak_idx = single_recs.index(peak_rec)
        ax.annotate(
            f"peak: {peak_rec:.0f}%",
            xy=(peak_idx, SHARED_Y_MAX),
            xytext=(peak_idx + 0.4, SHARED_Y_MAX - 12),
            fontsize=10,
            color=LINE_COLOR,
            fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=LINE_COLOR, lw=1.0),
        )

    thresh_idx = next((i for i, rec in enumerate(single_recs) if rec >= 80), None)
    if thresh_idx is not None:
        ax.axvline(thresh_idx, color="#999999", linewidth=1.0, linestyle=":", alpha=0.7)
        ax.text(thresh_idx + 0.15, SHARED_Y_MAX - 12, f"{x_labels[thresh_idx]}\n>=80%", fontsize=9, color="#555555", va="top")

    tick_positions = xs[::2] if len(xs) > 10 else xs
    tick_labels = x_labels[::2] if len(xs) > 10 else x_labels
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=11, rotation=45, ha="right")
    ax.set_ylim(SHARED_Y_MIN, SHARED_Y_MAX)
    ax.set_ylabel("Recovery (% of En-Ar gap)", fontsize=13)
    ax.set_xlabel("Patch layer", fontsize=13)
    ax.set_title(
        f"Causal Activation Patching: English -> Arabic Hidden-State Injection\n{args.model_label}",
        fontsize=13,
        fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=10, frameon=False, loc="lower right")

    for ext in [".pdf", ".png"]:
        out_path = os.path.join(args.out_dir, f"fig_patching_lines{ext}")
        plt.savefig(out_path, bbox_inches="tight", dpi=150)
        print(f"Saved -> {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
