import argparse
import csv
import os
import re
from collections import OrderedDict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from generic_model_utils import (
    answer_token_ids,
    batch_tokenize,
    correct_probs,
    get_layers,
    load_causal_lm,
    load_tokenizer,
)


def extract_letter(val):
    if not val or not isinstance(val, str):
        return ""
    s = val.strip().upper()
    m = re.search(r"\bANSWER\s*:\s*([A-F])\b", s)
    if m:
        return m.group(1)
    m = re.search(r"\b([A-F])\b", s)
    return m.group(1) if m else ""


def load_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def default_probe_layers(n_layers):
    step = max(1, round(n_layers * 0.10))
    layers = list(range(step, n_layers + 1, step))
    layers += [n_layers - 6, n_layers - 4, n_layers - 2, n_layers]
    return sorted(set(l for l in layers if 1 <= l <= n_layers))


def parse_probe_layers(value, n_layers):
    if not value:
        return default_probe_layers(n_layers)
    layers = sorted(set(int(x.strip()) for x in value.split(",") if x.strip()))
    bad = [x for x in layers if x < 1 or x > n_layers]
    if bad:
        raise ValueError(f"Probe layers out of range 1..{n_layers}: {bad}")
    return layers


def build_patch_configs(probe_layers):
    configs = OrderedDict()
    for layer_idx in probe_layers:
        configs[f"patch_L{layer_idx}"] = [layer_idx]

    if len(probe_layers) >= 3:
        configs[f"patch_L{probe_layers[-3]}_{probe_layers[-1]}"] = probe_layers[-3:]
    if len(probe_layers) >= 4:
        configs[f"patch_L{probe_layers[-4]}_{probe_layers[-1]}"] = probe_layers[-4:]
    return configs


def select_last(hidden, seq_lens):
    bs = hidden.shape[0]
    rows = torch.arange(bs, device=hidden.device)
    cols = seq_lens.to(hidden.device)
    return hidden[rows, cols]


def run_baseline(model, tokenizer, texts, gt_letters, ans_ids, args, label):
    probs = []
    n = len(texts)
    for i in range(0, n, args.batch_size):
        batch = texts[i:i + args.batch_size]
        gt = gt_letters[i:i + args.batch_size]
        input_ids, attn_mask, seq_lens = batch_tokenize(tokenizer, batch, args.max_len)
        with torch.no_grad():
            out = model(
                input_ids=input_ids.to("cuda:0"),
                attention_mask=attn_mask.to("cuda:0"),
            )
        probs.extend(correct_probs(out.logits.cpu(), seq_lens, gt, ans_ids))
        if i % 20 == 0:
            print(f"  [{label}] {min(i + len(batch), n)}/{n}")
        del out
        torch.cuda.empty_cache()
    return np.asarray(probs, dtype=np.float32)


def cache_source_hidden_states(model, tokenizer, texts, gt_letters, ans_ids, probe_layers, args, label):
    n = len(texts)
    source_hs = {layer_idx: [] for layer_idx in probe_layers}
    source_base_prob = []

    for i in range(0, n, args.batch_size):
        batch = texts[i:i + args.batch_size]
        gt = gt_letters[i:i + args.batch_size]
        input_ids, attn_mask, seq_lens = batch_tokenize(tokenizer, batch, args.max_len)
        with torch.no_grad():
            out = model(
                input_ids=input_ids.to("cuda:0"),
                attention_mask=attn_mask.to("cuda:0"),
                output_hidden_states=True,
            )

        for layer_idx in probe_layers:
            last = select_last(out.hidden_states[layer_idx], seq_lens).float().cpu()
            source_hs[layer_idx].append(last.numpy())

        source_base_prob.extend(correct_probs(out.logits.cpu(), seq_lens, gt, ans_ids))
        if i % 20 == 0:
            print(f"  [{label}] {min(i + len(batch), n)}/{n}")
        del out
        torch.cuda.empty_cache()

    for layer_idx in probe_layers:
        source_hs[layer_idx] = np.concatenate(source_hs[layer_idx], axis=0)
    return source_hs, np.asarray(source_base_prob, dtype=np.float32)


def patch_target(model, tokenizer, target_texts, gt_letters, ans_ids, layers, source_hs, configs, args):
    n = len(target_texts)
    patch_results = OrderedDict()

    for config_name, patch_layers in configs.items():
        print(f"\nPhase 3 [{config_name}]: patching at layers {patch_layers} ...")
        config_probs = []

        for i in range(0, n, args.batch_size):
            batch = target_texts[i:i + args.batch_size]
            gt = gt_letters[i:i + args.batch_size]
            bs = len(batch)
            input_ids, attn_mask, seq_lens = batch_tokenize(tokenizer, batch, args.max_len)
            batch_source_hs = {
                layer_idx: torch.tensor(source_hs[layer_idx][i:i + bs])
                for layer_idx in patch_layers
            }

            handles = []
            for layer_idx in patch_layers:
                cached = batch_source_hs[layer_idx]
                target_lens = seq_lens

                def make_hook(cached_=cached, target_lens_=target_lens, bs_=bs):
                    def hook_fn(_module, _inputs, output):
                        if isinstance(output, tuple):
                            hidden = output[0].clone()
                            rest = output[1:]
                            tuple_output = True
                        else:
                            hidden = output.clone()
                            rest = ()
                            tuple_output = False

                        for j in range(bs_):
                            hidden[j, target_lens_[j]] = cached_[j].to(hidden.device).to(hidden.dtype)
                        return (hidden, *rest) if tuple_output else hidden
                    return hook_fn

                handles.append(layers[layer_idx - 1].register_forward_hook(make_hook()))

            try:
                with torch.no_grad():
                    out = model(
                        input_ids=input_ids.to("cuda:0"),
                        attention_mask=attn_mask.to("cuda:0"),
                    )
                config_probs.extend(correct_probs(out.logits.cpu(), seq_lens, gt, ans_ids))
            finally:
                for handle in handles:
                    handle.remove()

            if i % 20 == 0:
                print(f"  [{config_name}] {min(i + bs, n)}/{n}")
            del out, input_ids, attn_mask
            torch.cuda.empty_cache()

        patch_results[config_name] = np.asarray(config_probs, dtype=np.float32)
        print(f"  Mean P(correct): {patch_results[config_name].mean():.3f}")

    return patch_results


def recovery(val, target_mean, source_mean):
    gap = source_mean - target_mean
    return 100 * (val - target_mean) / gap if gap > 0 else 0.0


def degradation(val, target_mean, source_mean):
    gap = target_mean - source_mean
    return 100 * (target_mean - val) / gap if gap > 0 else 0.0


def save_outputs(args, source_name, target_name, source_base, target_base, patch_results, gt_letters, out_npz):
    save = {"gt_letters": np.asarray(gt_letters)}
    if args.direction == "en_to_ar":
        save["en_base_prob"] = source_base
        save["ar_base_prob"] = target_base
    else:
        save["en_base_prob"] = target_base
        save["ar_base_prob"] = source_base
    for key, vals in patch_results.items():
        save[key] = vals

    out_path = os.path.join(args.out_dir, out_npz)
    np.savez(out_path, **save)
    print(f"\nSaved -> {out_path}")

    source_mean = source_base.mean()
    target_mean = target_base.mean()
    reverse = args.direction == "ar_to_en"
    metric_name = "Degradation" if reverse else "Recovery"
    metric_fn = degradation if reverse else recovery

    if reverse:
        print("\n== Reverse Patching Results (Arabic -> English injection) ==")
    else:
        print("\n== Activation Patching Results ==")
    print(f"  {'Config':<22} {'Mean P':>10}  {metric_name:>12}")
    print(f"  {target_name.lower() + '_base':<22} {target_mean:>10.3f}  {'(0%)':>12}")
    for config_name, vals in patch_results.items():
        mean_val = vals.mean()
        print(f"  {config_name:<22} {mean_val:>10.3f}  {metric_fn(mean_val, target_mean, source_mean):>11.1f}%")
    print(f"  {source_name.lower() + '_base':<22} {source_mean:>10.3f}  {'(100%)':>12}")

    all_configs = [f"{target_name.lower()}_base"] + list(patch_results.keys()) + [f"{source_name.lower()}_base"]
    all_means = [target_mean] + [patch_results[c].mean() for c in patch_results] + [source_mean]
    n = len(gt_letters)
    all_sems = (
        [target_base.std() / np.sqrt(n)]
        + [patch_results[c].std() / np.sqrt(n) for c in patch_results]
        + [source_base.std() / np.sqrt(n)]
    )

    n_patches = len(patch_results)
    cmap = plt.cm.RdYlGn(np.linspace(0.15, 0.85, n_patches))
    target_color = "#2A9D8F" if reverse else "#87CEEB"
    source_color = "#87CEEB" if reverse else "#2A9D8F"
    colors = [target_color] + list(cmap) + [source_color]
    x_labels = (
        [f"{target_name}\n(base)"]
        + [c.replace("patch_", "") for c in patch_results]
        + [f"{source_name}\n(base)"]
    )

    fig, ax = plt.subplots(figsize=(max(13, len(all_configs) * 0.75), 6))
    bars = ax.bar(
        range(len(all_configs)),
        all_means,
        color=colors,
        yerr=all_sems,
        capsize=4,
        alpha=0.88,
        edgecolor="white",
        linewidth=0.5,
    )
    for bar, mean_val in zip(bars, all_means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{mean_val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    for ci, config_name in enumerate(patch_results.keys()):
        ax.text(
            ci + 1,
            -0.018,
            f"{metric_fn(patch_results[config_name].mean(), target_mean, source_mean):.0f}%",
            ha="center",
            va="top",
            fontsize=8,
            color="#555555",
        )
    ax.text(-0.5, -0.018, f"{'Degrad.' if reverse else 'Recovery'}:", ha="left", va="top", fontsize=8, color="#555555")
    ax.axhline(
        target_mean,
        color=target_color,
        linewidth=1.5,
        linestyle="--",
        alpha=0.8,
        label=f"{target_name} baseline ({target_mean:.3f})",
    )
    ax.axhline(
        source_mean,
        color=source_color,
        linewidth=1.5,
        linestyle="--",
        alpha=0.8,
        label=f"{source_name} baseline ({source_mean:.3f})",
    )
    ax.set_xticks(range(len(all_configs)))
    ax.set_xticklabels(x_labels, fontsize=9.5)
    ax.set_ylabel("Mean P(correct answer letter) at model output", fontsize=11)
    title_prefix = "Reverse Activation Patching" if reverse else "Activation Patching"
    ax.set_title(
        f"{title_prefix}: {source_name} -> {target_name} Hidden State Injection\n"
        f"{args.model_label} - MedAraBench - access_gap",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(fontsize=10, frameon=False, loc="upper right" if reverse else "upper left")
    ax.set_ylim(-0.03, max(all_means) * 1.25 if max(all_means) > 0 else 1.0)
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    stem = "fig_patching_generic_reverse" if args.direction == "ar_to_en" else "fig_patching_generic"
    for ext in [".pdf", ".png"]:
        out_fig = os.path.join(args.out_dir, f"{stem}{ext}")
        plt.savefig(out_fig, bbox_inches="tight", dpi=150)
        print(f"Saved -> {out_fig}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--model_label", default="Model")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--probe_layers", default="")
    parser.add_argument("--direction", choices=["en_to_ar", "ar_to_en", "both"], default="en_to_ar")
    parser.add_argument("--reverse_out_dir", default="")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    if args.direction == "both":
        if not args.reverse_out_dir:
            raise ValueError("--reverse_out_dir is required when --direction both")
        os.makedirs(args.reverse_out_dir, exist_ok=True)

    rows = load_rows(args.csv)
    ag_rows = [r for r in rows if r.get("quadrant") == "access_gap"]
    if not ag_rows:
        raise RuntimeError("No access_gap rows found.")

    en_texts = [r["input_english"] for r in ag_rows]
    ar_texts = [r["input_arabic"] for r in ag_rows]
    gt_letters = [extract_letter(r["ground_truth"]) for r in ag_rows]
    n = len(ag_rows)
    print(f"Loading {args.csv}...")
    print(f"  Total rows: {len(rows)} | access_gap: {n}")

    tokenizer = load_tokenizer(args.model_path)
    model = load_causal_lm(args.model_path)
    layers = get_layers(model)
    n_layers = len(layers)
    probe_layers = parse_probe_layers(args.probe_layers, n_layers)
    configs = build_patch_configs(probe_layers)
    ans_ids = answer_token_ids(tokenizer)

    print(f"Model label: {args.model_label}")
    print(f"n_layers: {n_layers}")
    print(f"Probe layers: {probe_layers}")
    print(f"Patch configs: {dict(configs)}")

    def run_direction(direction, out_dir):
        run_args = argparse.Namespace(**vars(args))
        run_args.direction = direction
        run_args.out_dir = out_dir

        if direction == "en_to_ar":
            source_texts, target_texts = en_texts, ar_texts
            source_name, target_name = "English", "Arabic"
            out_npz = "patching_results.npz"
        else:
            source_texts, target_texts = ar_texts, en_texts
            source_name, target_name = "Arabic", "English"
            out_npz = "patching_results_reverse.npz"

        print(f"\n===== Direction: {source_name} -> {target_name} =====")
        print(f"\nPhase 1: {source_name} forward pass - caching layers {probe_layers} ...")
        source_hs, source_base = cache_source_hidden_states(
            model, tokenizer, source_texts, gt_letters, ans_ids, probe_layers, run_args, source_name)
        print(f"  {source_name} baseline mean P(correct): {source_base.mean():.3f}")

        print(f"\nPhase 2: {target_name} baseline (no patching) ...")
        target_base = run_baseline(model, tokenizer, target_texts, gt_letters, ans_ids, run_args, f"{target_name} base")
        print(f"  {target_name} baseline mean P(correct): {target_base.mean():.3f}")

        patch_results = patch_target(
            model, tokenizer, target_texts, gt_letters, ans_ids, layers, source_hs, configs, run_args)

        save_outputs(
            run_args, source_name, target_name, source_base, target_base,
            patch_results, gt_letters, out_npz)
        del source_hs, source_base, target_base, patch_results
        torch.cuda.empty_cache()

    if args.direction == "both":
        run_direction("en_to_ar", args.out_dir)
        run_direction("ar_to_en", args.reverse_out_dir)
    else:
        run_direction(args.direction, args.out_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
