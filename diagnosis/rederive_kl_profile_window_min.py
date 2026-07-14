import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


MODEL_LABELS = {
    "gemma3_27b_it": "Gemma-3-27B-IT",
    "mistral_7b": "Mistral-7B-Instruct-v0.3",
    "llama31_8b_inst": "Llama-3.1-8B-Instruct",
    "falcon": "Falcon-H1-7B-Instruct",
    "fanar": "Fanar-1-9B",
    "medgemma": "MedGemma-27B-Text-IT",
}


def derive_window_from_minimum(kl_profile, l_patch):
    n_layers = len(kl_profile)
    if n_layers == 0:
        raise ValueError("Empty KL profile")

    l_patch = int(l_patch)
    if l_patch > n_layers:
        print(f"[WARN] L_patch={l_patch} exceeds n_layers={n_layers}; clamping to L{n_layers}.")
        l_patch = n_layers
    if l_patch < 1:
        print(f"[WARN] L_patch={l_patch} is < 1; clamping to L1.")
        l_patch = 1

    l_min_idx = int(np.argmin(kl_profile))
    l_min = l_min_idx + 1
    lmax = n_layers

    neutral_and_active = kl_profile[l_min_idx:]
    mu = float(np.mean(neutral_and_active))
    std = float(np.std(neutral_and_active))
    tau = mu + std

    l_kl = None
    for idx in range(l_min_idx, n_layers):
        if kl_profile[idx] >= tau:
            l_kl = idx + 1
            break

    if l_kl is None:
        l_kl = lmax
        print("[WARN] No layer from L_min onward reached tau; using L_max.")

    windows = {
        "W1": [1, int(l_patch)],
        "W2": [int(l_patch), int(lmax)],
        "W3": [1, int(l_kl)],
        "W4": [int(l_kl), int(lmax)],
        "W5": [1, int(lmax)],
    }

    return {
        "tau": tau,
        "mu": mu,
        "std": std,
        "L_min": int(l_min),
        "L_kl": int(l_kl),
        "L_patch": int(l_patch),
        "L_max": int(lmax),
        "lora_window": [min(int(l_patch), int(l_kl)), max(int(l_patch), int(l_kl))],
        "kl_probe": int(l_kl),
        "windows": windows,
        "threshold_scope": "min_to_final",
        "threshold_layers": [int(l_min), int(lmax)],
        "search_scope": "min_to_final",
        "search_layers": [int(l_min), int(lmax)],
    }


def resolve_json_path(project_root, model_id, json_path):
    if json_path:
        return Path(json_path)
    return Path(project_root) / "diagnosis" / f"kl_profile_{model_id}_out" / "kl_profile.json"


def main():
    parser = argparse.ArgumentParser(
        description="Re-derive KL window using the L_min-anchored MedGemma method and regenerate one model's KL plot."
    )
    parser.add_argument("model_id", help="Model id, e.g. mistral_7b, falcon, medgemma")
    parser.add_argument("--project_root", default="/scratch/mk8737/farah/Adaptation")
    parser.add_argument("--json", default=None, help="Optional explicit path to kl_profile.json")
    parser.add_argument("--l_patch", type=int, default=None, help="Override L_patch instead of reading JSON window.L_patch")
    parser.add_argument("--dry_run", action="store_true", help="Print the derived window without writing or plotting.")
    parser.add_argument("--no_plot", action="store_true", help="Update JSON but do not regenerate kl_profile.png.")
    args = parser.parse_args()

    json_path = resolve_json_path(args.project_root, args.model_id, args.json)
    if not json_path.exists():
        raise FileNotFoundError(f"KL profile JSON not found: {json_path}")

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    kl_profile = np.asarray(data["kl_profile"], dtype=float)
    if kl_profile.ndim != 1 or kl_profile.size == 0:
        raise ValueError(f"Invalid kl_profile in {json_path}")
    if not np.isfinite(kl_profile).all():
        raise ValueError(f"Non-finite KL values in {json_path}")

    old_window = data.get("window") or {}
    l_patch = args.l_patch if args.l_patch is not None else old_window.get("L_patch")
    if l_patch is None:
        raise ValueError(f"No L_patch found in {json_path}; pass --l_patch.")

    new_window = derive_window_from_minimum(kl_profile, l_patch)
    print(f"{args.model_id}:")
    if old_window:
        print(
            f"  old L_patch=L{old_window.get('L_patch')} "
            f"L_kl=L{old_window.get('L_kl')} tau={float(old_window.get('tau', float('nan'))):.6g}"
        )
    print(
        f"  new L_patch=L{new_window['L_patch']} "
        f"L_min=L{new_window['L_min']} "
        f"L_kl=L{new_window['L_kl']} tau={new_window['tau']:.6g} "
        f"stats=L{new_window['threshold_layers'][0]}-L{new_window['threshold_layers'][1]} "
        f"search=L{new_window['search_layers'][0]}-L{new_window['search_layers'][1]} "
        f"lora_window=L{new_window['lora_window'][0]}-L{new_window['lora_window'][1]}"
    )

    if args.dry_run:
        return

    data["window"] = new_window
    data.setdefault("model_label", MODEL_LABELS.get(args.model_id, args.model_id))
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    print(f"Updated {json_path}")

    if not args.no_plot:
        plot_script = Path(__file__).with_name("plot_kl_profile_single.py")
        cmd = [
            sys.executable,
            str(plot_script),
            "--json",
            str(json_path),
            "--model_label",
            data.get("model_label", MODEL_LABELS.get(args.model_id, args.model_id)),
        ]
        print("CMD:", " ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
