import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
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

QUAD_ORDER = ["both_correct", "access_gap", "arabic_only", "both_wrong"]
QUAD_COLORS = {
    "both_correct": "#2A9D8F",
    "access_gap": "#C0392B",
    "arabic_only": "#F4C430",
    "both_wrong": "#0D3349",
}
QUAD_LABELS = {
    "both_correct": "En correct / Ar correct",
    "access_gap": "En correct / Ar wrong",
    "arabic_only": "En wrong / Ar correct",
    "both_wrong": "En wrong / Ar wrong",
}

LATE_FRAC = 0.75


def load_profile(project_root, model_key):
    path = project_root / "diagnosis" / f"tuned_lens_{model_key}_out" / "tuned_lens_results.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing tuned-lens npz for {model_key}: {path}")

    d = np.load(path, allow_pickle=True)
    required = ["en_probs", "ar_probs", "quadrants", "layers"]
    missing = [k for k in required if k not in d.files]
    if missing:
        raise ValueError(f"Missing fields in {path}: {missing}")

    en_probs = np.asarray(d["en_probs"], dtype=float)
    ar_probs = np.asarray(d["ar_probs"], dtype=float)
    quadrants = d["quadrants"].astype(str)
    layers = np.asarray(d["layers"], dtype=int)

    if en_probs.shape != ar_probs.shape:
        raise ValueError(f"EN/AR probability shapes differ in {path}: {en_probs.shape} vs {ar_probs.shape}")
    if en_probs.shape[0] != quadrants.shape[0]:
        raise ValueError(f"Probability rows and quadrants differ in {path}")
    if en_probs.shape[1] != layers.shape[0]:
        raise ValueError(f"Probability columns and layers differ in {path}")
    if not np.isfinite(en_probs).all() or not np.isfinite(ar_probs).all():
        raise ValueError(f"Non-finite tuned-lens probabilities in {path}")

    return {
        "model_key": model_key,
        "label": MODEL_LABELS.get(model_key, model_key),
        "path": path,
        "en_probs": en_probs,
        "ar_probs": ar_probs,
        "quadrants": quadrants,
        "layers": layers,
    }


def draw_panel(ax, profile):
    en_probs = profile["en_probs"]
    ar_probs = profile["ar_probs"]
    quadrants = profile["quadrants"]
    layers = profile["layers"]

    xs = np.arange(len(layers))
    late_probe_idx = int(len(layers) * LATE_FRAC)

    ax.axvspan(-0.5, late_probe_idx - 0.5, color="#EAF4FF", alpha=0.45, zorder=0)
    ax.axvspan(late_probe_idx - 0.5, len(layers) - 0.5, color="#FFF3E8", alpha=0.55, zorder=0)

    for quadrant in QUAD_ORDER:
        mask = quadrants == quadrant
        if not mask.any():
            continue
        en_mean = en_probs[mask].mean(axis=0)
        ar_mean = ar_probs[mask].mean(axis=0)
        en_se = en_probs[mask].std(axis=0) / np.sqrt(mask.sum())
        ar_se = ar_probs[mask].std(axis=0) / np.sqrt(mask.sum())
        color = QUAD_COLORS[quadrant]

        ax.plot(xs, en_mean, color=color, linewidth=1.8, linestyle="-", marker="o", markersize=4.2, zorder=3)
        ax.fill_between(xs, en_mean - en_se, en_mean + en_se, color=color, alpha=0.11, zorder=2)
        ax.plot(xs, ar_mean, color=color, linewidth=1.8, linestyle="--", marker="^", markersize=4.2, zorder=3)
        ax.fill_between(xs, ar_mean - ar_se, ar_mean + ar_se, color=color, alpha=0.07, zorder=2)

    step = max(1, len(layers) // 6)
    tick_labels = []
    for idx, layer in enumerate(layers):
        if idx == 0:
            tick_labels.append("L0\nemb")
        elif idx == len(layers) - 1:
            tick_labels.append(f"L{layer}\nfinal")
        elif idx % step == 0:
            tick_labels.append(f"L{layer}")
        else:
            tick_labels.append("")

    ax.set_xticks(xs)
    ax.set_xticklabels(tick_labels, fontsize=8)
    ax.set_xlim(-0.5, len(layers) - 0.5)
    ax.set_ylim(0, 1.02)
    ax.set_title(profile["label"], fontsize=10)
    ax.grid(axis="y", alpha=0.22, linewidth=0.7)
    ax.tick_params(axis="y", labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)


def make_panel(profiles, output_path):
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.8), squeeze=False)
    for ax, profile in zip(axes.ravel(), profiles):
        draw_panel(ax, profile)

    for ax in axes[-1, :]:
        ax.set_xlabel("Layer", fontsize=10)
    for ax in axes[:, 0]:
        ax.set_ylabel("Mean P(correct answer)", fontsize=10)

    quadrant_handles = [
        mlines.Line2D([], [], color=QUAD_COLORS[q], linewidth=2.2, label=QUAD_LABELS[q])
        for q in QUAD_ORDER
    ]
    style_handles = [
        mlines.Line2D([], [], color="dimgrey", linewidth=2.2, linestyle="-", marker="o", markersize=5, label="English"),
        mlines.Line2D([], [], color="dimgrey", linewidth=2.2, linestyle="--", marker="^", markersize=5, label="Arabic"),
    ]
    zone_handles = [
        mpatches.Patch(facecolor="#EAF4FF", alpha=0.7, label="Early / mid layers"),
        mpatches.Patch(facecolor="#FFF3E8", alpha=0.7, label="Late layers"),
    ]

    fig.legend(
        handles=quadrant_handles + style_handles + zone_handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=8.5,
        handlelength=2.2,
        columnspacing=1.2,
    )
    fig.suptitle("Tuned Lens: Correct Answer Probability Across Layers", fontsize=13, y=0.98)
    fig.tight_layout(rect=[0, 0.10, 1, 0.95])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    pdf_path = output_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")
    print(f"Saved {pdf_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot six-model tuned-lens panel from saved npz files.")
    parser.add_argument("--project_root", default="/scratch/mk8737/farah/Adaptation")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--filename", default="tuned_lens_panel.png")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    out_dir = Path(args.out_dir) if args.out_dir else project_root / "diagnosis" / "tuned_lens_panel_six_out"
    profiles = [load_profile(project_root, model_key) for model_key in args.models]
    make_panel(profiles, out_dir / args.filename)


if __name__ == "__main__":
    main()
