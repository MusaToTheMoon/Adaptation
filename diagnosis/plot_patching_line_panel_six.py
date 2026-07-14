import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_MODELS = [
    "gemma3_27b_it",
    "mistral_7b",
    "llama31_8b_inst",
    "falcon",
    "fanar",
    "medgemma",
]

MODEL_LABELS = {
    "gemma3_27b_it": "Gemma-3-27B-IT",
    "mistral_7b": "Mistral-7B-Instruct-v0.3",
    "llama31_8b_inst": "Llama-3.1-8B-Instruct",
    "falcon": "Falcon-H1-7B-Instruct",
    "fanar": "Fanar-1-9B",
    "medgemma": "MedGemma-27B-Text-IT",
}

AR_COLOR = "#0D3349"
EN_COLOR = "#2A9D8F"
LINE_COLOR = "#C0392B"
SHARED_Y_MIN = -60
SHARED_Y_MAX = 130


def patch_sort_key(key):
    vals = []
    for part in key.replace("patch_", "").split("_"):
        if part.startswith("L"):
            vals.append(int(part[1:]))
    return vals


def split_patch_keys(keys):
    patch_keys = [k for k in keys if k not in {"en_base_prob", "ar_base_prob", "gt_letters"}]
    patch_keys = sorted(patch_keys, key=patch_sort_key)
    single = [k for k in patch_keys if "_" not in k.replace("patch_", "")]
    span = [k for k in patch_keys if k not in single]
    return single, span


def ensure_finite(path, arrays):
    bad = []
    for name, arr in arrays.items():
        if not np.isfinite(arr).all():
            bad.append(name)
    if bad:
        raise ValueError(f"Non-finite values in {path}: {', '.join(bad)}")


def recovery(val, ar_mean, en_mean):
    gap = en_mean - ar_mean
    return 100 * (val - ar_mean) / gap if gap > 0 else 0.0


def load_profile(project_root, model_key):
    path = project_root / "diagnosis" / f"activation_patching_{model_key}_out" / "patching_results.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing forward patching npz for {model_key}: {path}")

    d = np.load(path, allow_pickle=True)
    single_keys, span_keys = split_patch_keys(d.files)
    if not single_keys:
        raise RuntimeError(f"No single-layer patch keys found in {path}")

    arrays = {k: d[k] for k in ["en_base_prob", "ar_base_prob", *single_keys, *span_keys]}
    ensure_finite(path, arrays)

    en_base = d["en_base_prob"]
    ar_base = d["ar_base_prob"]
    n = len(en_base)
    ar_mean = float(ar_base.mean())
    en_mean = float(en_base.mean())
    gap = abs(en_mean - ar_mean)

    single_means = [float(d[k].mean()) for k in single_keys]
    single_sems = [float(d[k].std() / np.sqrt(n)) for k in single_keys]
    single_recs = [recovery(m, ar_mean, en_mean) for m in single_means]
    se_rec = [100 * s / gap if gap > 0 else 0.0 for s in single_sems]

    span_recs = []
    for key in span_keys:
        span_recs.append((key.replace("patch_", ""), recovery(float(d[key].mean()), ar_mean, en_mean)))

    return {
        "model_key": model_key,
        "label": MODEL_LABELS.get(model_key, model_key),
        "x_labels": [k.replace("patch_", "") for k in single_keys],
        "values": np.asarray(single_recs, dtype=float),
        "se": np.asarray(se_rec, dtype=float),
        "span_values": span_recs,
    }


def draw_panel(ax, profile, annotate_spans=False):
    xs = np.arange(len(profile["values"]))
    values = profile["values"]
    se = profile["se"]
    band_lo = np.maximum(values - se, SHARED_Y_MIN)
    band_hi = np.minimum(values + se, SHARED_Y_MAX)

    ax.plot(xs, values, color=LINE_COLOR, linewidth=1.8, marker="o", markersize=4.5, zorder=3)
    ax.fill_between(xs, band_lo, band_hi, color=LINE_COLOR, alpha=0.10, zorder=2)
    ax.axhline(100, color=EN_COLOR, linewidth=1.5, linestyle="--", alpha=0.85)
    ax.axhline(0, color=AR_COLOR, linewidth=1.5, linestyle="--", alpha=0.85)

    if annotate_spans and profile["span_values"]:
        for span_idx, (label, rec) in enumerate(profile["span_values"]):
            x_text = max(xs[-1] - 2 - span_idx * 0.75, 0)
            y_text = min(max(rec + 12 + span_idx * 14, SHARED_Y_MIN + 15), SHARED_Y_MAX - 10)
            ax.annotate(
                f"{label}: {rec:.0f}%",
                xy=(xs[-1], max(rec, SHARED_Y_MIN + 5)),
                xytext=(x_text, y_text),
                fontsize=8,
                color="#555555",
                arrowprops=dict(arrowstyle="->", color="#aaaaaa", lw=0.8),
            )

    peak = float(values.max())
    if peak > SHARED_Y_MAX:
        peak_idx = int(values.argmax())
        ax.annotate(
            f"peak: {peak:.0f}%",
            xy=(peak_idx, SHARED_Y_MAX),
            xytext=(peak_idx + 0.4, SHARED_Y_MAX - 12),
            fontsize=8,
            color=LINE_COLOR,
            arrowprops=dict(arrowstyle="->", color=LINE_COLOR, lw=0.9),
        )

    thresh_idx = next((i for i, rec in enumerate(values) if rec >= 80), None)
    subtitle = "no >=80%"
    if thresh_idx is not None:
        ax.axvline(thresh_idx, color="#999999", linewidth=1.0, linestyle=":", alpha=0.7)
        subtitle = f"{profile['x_labels'][thresh_idx]} >=80%"

    ticks = xs[::2] if len(xs) > 10 else xs
    labels = profile["x_labels"][::2] if len(xs) > 10 else profile["x_labels"]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=8, rotation=45, ha="right")
    ax.set_ylim(SHARED_Y_MIN, SHARED_Y_MAX)
    ax.set_title(f"{profile['label']}\n{subtitle}", fontsize=10)
    ax.grid(axis="y", alpha=0.18)
    ax.tick_params(axis="y", labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)


def make_panel(profiles, output_path, annotate_spans=False):
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.8), squeeze=False)
    for ax, profile in zip(axes.ravel(), profiles):
        draw_panel(ax, profile, annotate_spans=annotate_spans)

    for ax in axes[-1, :]:
        ax.set_xlabel("Patch layer", fontsize=10)
    for ax in axes[:, 0]:
        ax.set_ylabel("Recovery (% of En-Ar gap)", fontsize=10)

    handles = [
        mlines.Line2D([], [], color=LINE_COLOR, linewidth=1.8, marker="o", markersize=5, label="Single-layer recovery"),
        mlines.Line2D([], [], color=EN_COLOR, linewidth=1.5, linestyle="--", label="English baseline (100%)"),
        mlines.Line2D([], [], color=AR_COLOR, linewidth=1.5, linestyle="--", label="Arabic baseline (0%)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle("Activation Patching: English -> Arabic Hidden-State Injection", fontsize=13, y=0.98)
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    pdf_path = output_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")
    print(f"Saved {pdf_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot six-model forward activation patching line panel.")
    parser.add_argument("--project_root", default="/scratch/mk8737/farah/Adaptation")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--filename", default="activation_patching_forward_panel.png")
    parser.add_argument("--annotate_spans", action="store_true", help="Annotate multi-layer span patches.")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    out_dir = Path(args.out_dir) if args.out_dir else project_root / "diagnosis" / "activation_patching_forward_panel_out"
    profiles = [load_profile(project_root, model_key) for model_key in args.models]
    make_panel(profiles, out_dir / args.filename, annotate_spans=args.annotate_spans)


if __name__ == "__main__":
    main()
