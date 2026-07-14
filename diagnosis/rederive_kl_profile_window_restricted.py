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


def derive_window_post_l_patch(kl_profile, l_patch):
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

    search_start = l_patch + 1
    if search_start > n_layers:
        print(f"[WARN] No layers after L_patch=L{l_patch}; using L{n_layers} for threshold/search.")
        search_start = n_layers

    candidates = kl_profile[search_start - 1:]
    mu = float(np.mean(candidates))
    std = float(np.std(candidates))
    tau = mu + std

    l_kl = None
    for idx in range(search_start - 1, n_layers):
        if kl_profile[idx] >= tau:
            l_kl = idx + 1
            break

    if l_kl is None:
        peak_offset = int(np.argmax(candidates))
        l_kl = search_start + peak_offset
        print(f"[WARN] No post-L_patch layer reached tau; using post-L_patch peak L{l_kl}.")

    return {
        "tau": tau,
        "mu": mu,
        "std": std,
        "L_kl": int(l_kl),
        "L_patch": int(l_patch),
        "lora_window": [min(int(l_patch), int(l_kl)), max(int(l_patch), int(l_kl))],
        "kl_probe": int(l_kl),
        "threshold_scope": "post_l_patch",
        "threshold_layers": [int(search_start), int(n_layers)],
    }


def resolve_json_path(project_root, model_id, json_path):
    if json_path:
        return Path(json_path)
    return Path(project_root) / "diagnosis" / f"kl_profile_{model_id}_out" / "kl_profile.json"


def main():
    parser = argparse.ArgumentParser(
        description="Re-derive post-L_patch KL window for one model and regenerate its single KL plot."
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

    new_window = derive_window_post_l_patch(kl_profile, l_patch)
    print(f"{args.model_id}:")
    if old_window:
        print(
            f"  old L_patch=L{old_window.get('L_patch')} "
            f"L_kl=L{old_window.get('L_kl')} tau={float(old_window.get('tau', float('nan'))):.6g}"
        )
    print(
        f"  new L_patch=L{new_window['L_patch']} "
        f"L_kl=L{new_window['L_kl']} tau={new_window['tau']:.6g} "
        f"stats=L{new_window['threshold_layers'][0]}-L{new_window['threshold_layers'][1]}"
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
