import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MODEL_LABELS = {
    "gemma3_27b_it": "Gemma-3-27B-IT",
    "mistral_7b": "Mistral-7B-Instruct-v0.3",
    "llama31_8b_inst": "Llama-3.1-8B-Instruct",
    "falcon": "Falcon-H1-7B-Instruct",
    "fanar": "Fanar-1-9B",
    "medgemma": "MedGemma-27B-Text-IT",
}


def plot_kl_profile_json(json_path, output_path=None, model_label=None):
    json_path = Path(json_path)
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    kl_profile = np.asarray(data["kl_profile"], dtype=float)
    if kl_profile.ndim != 1 or kl_profile.size == 0:
        raise ValueError(f"Invalid kl_profile in {json_path}")
    if not np.isfinite(kl_profile).all():
        raise ValueError(f"Non-finite KL values in {json_path}")

    window = data.get("window") or {}
    required = ["tau", "L_kl", "L_patch", "lora_window"]
    missing = [key for key in required if key not in window]
    if missing:
        raise ValueError(f"Missing window fields in {json_path}: {missing}")

    label = model_label or data.get("model_label") or data.get("model") or json_path.parent.name
    layers = np.arange(1, len(kl_profile) + 1)
    l_kl = int(window["L_kl"])
    l_patch = int(window["L_patch"])
    window_start, window_end = [int(x) for x in window["lora_window"]]
    threshold_layers = window.get("threshold_layers")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(layers, kl_profile, color="#2c7bb6", linewidth=1.8, label="KL divergence")
    ax.axhline(
        float(window["tau"]),
        color="#d7191c",
        linestyle="--",
        linewidth=1.2,
        label=f"post-L_patch tau = mu + sigma = {float(window['tau']):.3f}",
    )
    ax.axvline(l_kl, color="#fdae61", linestyle=":", linewidth=1.5, label=f"L_kl = L{l_kl}")
    ax.axvline(l_patch, color="#1a9641", linestyle=":", linewidth=1.5, label=f"L_patch = L{l_patch}")
    ax.axvspan(window_start, window_end, alpha=0.12, color="#fdae61", label=f"Window L{window_start}-L{window_end}")

    if threshold_layers:
        scope_label = f"tau stats: L{threshold_layers[0]}-L{threshold_layers[1]}"
        ax.text(0.98, 0.04, scope_label, transform=ax.transAxes, ha="right", va="bottom", fontsize=9, color="#555555")

    ax.set_xlabel("Transformer Layer", fontsize=12)
    ax.set_ylabel("KL Divergence (nats)", fontsize=12)
    ax.set_title(
        "Cross-Lingual KL Divergence Profile (Arabic vs. English, Logit Lens)\n"
        f"{label} · Base Model · MedAraBench val",
        fontsize=11,
    )
    ax.set_xlim(1, len(kl_profile))
    ax.set_xticks(np.arange(0, len(kl_profile) + 1, 5))
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    output_path = Path(output_path) if output_path else json_path.with_name("kl_profile.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot one saved KL profile JSON.")
    parser.add_argument("--json", required=True, help="Path to kl_profile.json")
    parser.add_argument("--out", default=None, help="Output PNG path. Defaults to kl_profile.png next to JSON.")
    parser.add_argument("--model_label", default=None)
    args = parser.parse_args()
    plot_kl_profile_json(args.json, args.out, args.model_label)


if __name__ == "__main__":
    main()
