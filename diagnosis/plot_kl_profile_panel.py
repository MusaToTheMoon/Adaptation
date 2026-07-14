import argparse
import json
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


def load_profile(project_root, model_key):
    path = project_root / "diagnosis" / f"kl_profile_{model_key}_out" / "kl_profile.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing KL profile JSON for {model_key}: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    kl_profile = np.asarray(data["kl_profile"], dtype=float)
    if kl_profile.ndim != 1 or kl_profile.size == 0:
        raise ValueError(f"Invalid kl_profile in {path}")
    if not np.isfinite(kl_profile).all():
        raise ValueError(f"Non-finite KL values in {path}")

    window = data.get("window") or {}
    required = ["tau", "L_kl", "L_patch", "lora_window"]
    missing = [k for k in required if k not in window]
    if missing:
        raise ValueError(f"Missing window fields in {path}: {missing}")

    label = data.get("model_label") or MODEL_LABELS.get(model_key, model_key)
    return {
        "model_key": model_key,
        "path": path,
        "label": label,
        "kl_profile": kl_profile,
        "window": window,
    }


def make_panel(profiles, output_path, title):
    n = len(profiles)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 7.8), squeeze=False)

    for ax in axes.ravel()[n:]:
        ax.axis("off")

    for ax, profile in zip(axes.ravel(), profiles):
        kl_profile = profile["kl_profile"]
        window = profile["window"]
        layers = np.arange(1, len(kl_profile) + 1)

        l_kl = int(window["L_kl"])
        l_patch = int(window["L_patch"])
        win_start, win_end = [int(x) for x in window["lora_window"]]

        ax.plot(layers, kl_profile, color="#2c7bb6", linewidth=1.7)
        ax.axhline(float(window["tau"]), color="#d7191c", linestyle="--", linewidth=1.0)
        ax.axvline(l_kl, color="#fdae61", linestyle=":", linewidth=1.4)
        ax.axvline(l_patch, color="#1a9641", linestyle=":", linewidth=1.4)
        ax.axvspan(win_start, win_end, alpha=0.12, color="#fdae61")

        ax.set_title(
            f"{profile['label']}\nL_kl=L{l_kl}, L_patch=L{l_patch}",
            fontsize=10,
        )
        ax.set_xlim(1, len(kl_profile))
        ax.set_xticks(np.arange(0, len(kl_profile) + 1, 10))
        ax.grid(axis="y", alpha=0.28)
        ax.tick_params(labelsize=8)

    for ax in axes[-1, :]:
        if ax.has_data():
            ax.set_xlabel("Transformer Layer", fontsize=10)
    for ax in axes[:, 0]:
        if ax.has_data():
            ax.set_ylabel("KL Divergence (nats)", fontsize=10)

    handles = [
        mlines.Line2D([], [], color="#2c7bb6", linewidth=1.7, label="KL divergence"),
        mlines.Line2D([], [], color="#d7191c", linestyle="--", linewidth=1.0, label="tau = mean + std"),
        mlines.Line2D([], [], color="#fdae61", linestyle=":", linewidth=1.4, label="L_kl"),
        mlines.Line2D([], [], color="#1a9641", linestyle=":", linewidth=1.4, label="L_patch"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.suptitle(title, fontsize=13, y=0.98)
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    pdf_path = output_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")
    print(f"Saved {pdf_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot KL profile JSONs for multiple models in a single panel."
    )
    parser.add_argument(
        "--project_root",
        default="/scratch/mk8737/farah/Adaptation",
        help="Repository/project root. Defaults to this HPC project path.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Model keys to include, in panel order.",
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Output directory. Defaults to <project_root>/diagnosis/kl_profile_panel_out.",
    )
    parser.add_argument(
        "--filename",
        default="kl_profile_panel.png",
        help="Output PNG filename.",
    )
    parser.add_argument(
        "--title",
        default="Cross-Lingual KL Divergence Profiles",
    )
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    out_dir = Path(args.out_dir) if args.out_dir else project_root / "diagnosis" / "kl_profile_panel_out"
    profiles = [load_profile(project_root, model_key) for model_key in args.models]
    make_panel(profiles, out_dir / args.filename, args.title)


if __name__ == "__main__":
    main()
