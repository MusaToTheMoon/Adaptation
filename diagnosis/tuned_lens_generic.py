import argparse
import csv
import math
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from generic_model_utils import (
    answer_token_ids,
    batch_tokenize,
    get_layers,
    get_norm_and_lm_head,
    load_causal_lm,
    load_tokenizer,
)


QUAD_ORDER = ["both_correct", "access_gap", "arabic_only", "both_wrong"]
QUAD_COLORS = {
    "both_correct": "#2DC653",
    "access_gap": "#E63946",
    "arabic_only": "#F4A261",
    "both_wrong": "#ADB5BD",
}
QUAD_LABELS = {
    "both_correct": "En correct / Ar correct (shared knowledge)",
    "access_gap": "En correct / Ar wrong (access gap)",
    "arabic_only": "En wrong / Ar correct (Arabic-only)",
    "both_wrong": "En wrong / Ar wrong (both wrong)",
}
PROJECTION_VERSION = 2


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
    layers = list(range(0, n_layers + 1, step))
    layers += [n_layers - 6, n_layers - 4, n_layers - 2, n_layers - 1, n_layers]
    return sorted(set(l for l in layers if 0 <= l <= n_layers))


def parse_probe_layers(value, n_layers):
    if not value:
        return default_probe_layers(n_layers)
    layers = sorted(set(int(x.strip()) for x in value.split(",") if x.strip()))
    bad = [x for x in layers if x < 0 or x > n_layers]
    if bad:
        raise ValueError(f"Probe layers out of range 0..{n_layers}: {bad}")
    return layers


def module_execution_device(*modules):
    for module in modules:
        hook = getattr(module, "_hf_hook", None)
        device = getattr(hook, "execution_device", None)
        if device is not None and str(device) != "meta":
            return torch.device(device)
    for module in modules:
        for param in module.parameters(recurse=True):
            if param.device.type != "meta":
                return param.device
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def module_param_dtype(*modules):
    for module in modules:
        for param in module.parameters(recurse=True):
            return param.dtype
    return torch.bfloat16


def lm_head_multiplier(model):
    for obj in (getattr(model, "model", None), model, getattr(model, "config", None)):
        if obj is None:
            continue
        value = getattr(obj, "lm_head_multiplier", None)
        if value is not None:
            return float(value)
    return 1.0


def hidden_state_already_final_normed(model, layer_idx, n_layers):
    model_type = getattr(getattr(model, "config", None), "model_type", "")
    return model_type == "falcon_h1" and layer_idx == n_layers


def project_hidden(model, norm, lm_head, h, proj_dtype, layer_idx=None, n_layers=None):
    if hidden_state_already_final_normed(model, layer_idx, n_layers):
        projected_h = h.to(proj_dtype)
    else:
        projected_h = norm(h.to(proj_dtype))
    logits = lm_head(projected_h).float()
    scale = lm_head_multiplier(model)
    if scale != 1.0:
        logits = logits * scale
    return logits


def select_last(hidden, seq_lens):
    bs = hidden.shape[0]
    rows = torch.arange(bs, device=hidden.device)
    cols = seq_lens.to(hidden.device)
    return hidden[rows, cols]


class AffineTranslator(nn.Module):
    """
    Mistral-style residual affine translator:
      T_l(h) = h + W_l h + b_l
    with W_l initialized to zero, so the translator starts as identity.
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.W = nn.Parameter(torch.zeros(hidden_size, hidden_size))
        self.b = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, h):
        return h + h @ self.W.T + self.b


def checkpoint_compatible(path, hidden_size, model=None):
    try:
        state = torch.load(path, map_location="cpu")
    except Exception:
        return False
    ok_shape = (
        "W" in state
        and "b" in state
        and tuple(state["W"].shape) == (hidden_size, hidden_size)
        and tuple(state["b"].shape) == (hidden_size,)
    )
    if not ok_shape:
        return False
    if getattr(getattr(model, "config", None), "model_type", "") == "falcon_h1":
        return (
            state.get("_projection_version") == PROJECTION_VERSION
            and abs(float(state.get("_lm_head_multiplier", -1.0)) - lm_head_multiplier(model)) < 1e-12
        )
    return True


def checkpoint_state_dict(path):
    state = torch.load(path, map_location="cpu")
    return {"W": state["W"], "b": state["b"]}


def translator_checkpoint_payload(translator, model):
    payload = {k: v.cpu() for k, v in translator.state_dict().items()}
    payload["_projection_version"] = PROJECTION_VERSION
    payload["_lm_head_multiplier"] = lm_head_multiplier(model)
    return payload


def answer_prob_and_rank(logits, gt, ans_ids):
    ids = ans_ids.get(gt, [])
    if not ids:
        return 0.0, -1
    valid = [tid for tid in ids if 0 <= tid < logits.shape[-1]]
    if not valid:
        return 0.0, -1
    idx = torch.tensor(valid, dtype=torch.long, device=logits.device)
    probs = torch.softmax(logits.float(), dim=-1)
    prob = probs[idx].sum().item()
    best_logit = logits[idx].max()
    rank = int((logits > best_logit).sum().item())
    return float(prob), rank


def train_translators(args, model, tokenizer, norm, lm_head, n_layers, probe_layers, ans_ids):
    if not args.train_csv:
        raise ValueError("--train_csv is required for train")

    input_device = torch.device("cuda:0")
    proj_device = module_execution_device(norm, lm_head)
    proj_dtype = module_param_dtype(norm, lm_head)
    print(f"\nTraining - input_device={input_device}, proj_device={proj_device}, proj_dtype={proj_dtype}")

    rows = load_rows(args.train_csv)
    texts = [r["input_english"] for r in rows[:args.train_n]]
    print(f"Loading training CSV: {args.train_csv}")
    print(f"  Using {len(texts)} training examples")

    print("\nCollecting hidden states ...")
    hs_by_layer = defaultdict(list)
    final_logits_list = []
    layers_to_save = set(probe_layers)

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), args.train_batch), desc="forward"):
            batch = texts[i:i + args.train_batch]
            input_ids, attn_mask, seq_lens = batch_tokenize(tokenizer, batch, args.max_len)
            bs = len(batch)

            out = model(
                input_ids=input_ids.to(input_device),
                attention_mask=attn_mask.to(input_device),
                output_hidden_states=True,
            )

            logits = select_last(out.logits, seq_lens).float().cpu()
            final_logits_list.append(logits)

            for layer_idx in layers_to_save:
                hs = out.hidden_states[layer_idx]
                last = select_last(hs, seq_lens).float().cpu()
                hs_by_layer[layer_idx].append(last)

            del out
            torch.cuda.empty_cache()

    final_logits_all = torch.cat(final_logits_list, dim=0)
    final_probs_all = F.softmax(final_logits_all, dim=-1)
    for layer_idx in probe_layers:
        hs_by_layer[layer_idx] = torch.cat(hs_by_layer[layer_idx], dim=0)

    n_train = final_probs_all.shape[0]
    hidden_size = hs_by_layer[probe_layers[0]].shape[-1]
    layers_to_train = [l for l in probe_layers if 0 < l < n_layers]
    print(f"  N_train={n_train}, d={hidden_size}")
    print(f"\nTraining translators for layers: {layers_to_train}")

    for layer_idx in layers_to_train:
        ckpt = os.path.join(args.lens_dir, f"translator_layer_{layer_idx:02d}.pt")
        if os.path.exists(ckpt) and checkpoint_compatible(ckpt, hidden_size, model):
            print(f"  [L{layer_idx:2d}] already exists, skipping")
            continue
        if os.path.exists(ckpt):
            print(f"  [L{layer_idx:2d}] existing checkpoint is incompatible; overwriting")

        print(f"\n  [L{layer_idx:2d}] training ...")
        translator = AffineTranslator(hidden_size).to(proj_device)
        opt = torch.optim.Adam(translator.parameters(), lr=args.lr)
        hs = hs_by_layer[layer_idx]

        best_loss = math.inf
        for epoch in range(args.epochs):
            perm = torch.randperm(n_train)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n_train, args.train_batch):
                idx = perm[start:start + args.train_batch]
                h_b = hs[idx].to(proj_device)
                p_target = final_probs_all[idx]

                opt.zero_grad()
                h_trans = translator(h_b)
                logits = project_hidden(model, norm, lm_head, h_trans, proj_dtype, layer_idx, n_layers)
                log_p = F.log_softmax(logits, dim=-1)

                kl = F.kl_div(log_p, p_target.to(log_p.device), reduction="batchmean")
                reg = args.reg * translator.W.pow(2).sum()
                loss = kl + reg.to(kl.device)
                loss.backward()
                opt.step()

                epoch_loss += float(loss.item())
                n_batches += 1

            avg = epoch_loss / max(n_batches, 1)
            if epoch == 0 or (epoch + 1) % 2 == 0:
                print(f"    epoch {epoch + 1}/{args.epochs} loss={avg:.5f}")
            if avg < best_loss:
                best_loss = avg
                torch.save(translator_checkpoint_payload(translator, model), ckpt)

        print(f"  [L{layer_idx:2d}] best_loss={best_loss:.5f} -> {ckpt}")

    print("\nTraining complete.")


def run_tuned_lens(args, texts, gt_letters, model, tokenizer, norm, lm_head, translators, probe_layers, ans_ids, label, n_layers):
    input_device = torch.device("cuda:0")
    proj_device = module_execution_device(norm, lm_head)
    proj_dtype = module_param_dtype(norm, lm_head)

    n = len(texts)
    all_ranks = np.zeros((n, len(probe_layers)), dtype=np.int32)
    all_probs = np.zeros((n, len(probe_layers)), dtype=np.float32)

    for i in tqdm(range(0, n, args.batch_size), desc=f"tuned-lens [{label}]"):
        batch = texts[i:i + args.batch_size]
        gt_batch = gt_letters[i:i + args.batch_size]
        bs = len(batch)
        input_ids, attn_mask, seq_lens = batch_tokenize(tokenizer, batch, args.max_len)

        with torch.no_grad():
            out = model(
                input_ids=input_ids.to(input_device),
                attention_mask=attn_mask.to(input_device),
                output_hidden_states=True,
            )

        for li, layer_idx in enumerate(probe_layers):
            hs = out.hidden_states[layer_idx]
            last = select_last(hs, seq_lens).float().cpu()

            if layer_idx in translators:
                with torch.no_grad():
                    last = translators[layer_idx](last)

            h = last.to(proj_device)
            logits = project_hidden(model, norm, lm_head, h, proj_dtype, layer_idx, n_layers).cpu()

            for j, gt in enumerate(gt_batch):
                prob, rank = answer_prob_and_rank(logits[j], gt, ans_ids)
                all_probs[i + j, li] = prob
                all_ranks[i + j, li] = rank

        del out
        torch.cuda.empty_cache()

    return all_ranks, all_probs


def eval_mode(args, model, tokenizer, norm, lm_head, n_layers, probe_layers, ans_ids):
    if not args.csv:
        raise ValueError("--csv is required for eval")

    rows = load_rows(args.csv)
    en_texts = [r["input_english"] for r in rows]
    ar_texts = [r["input_arabic"] for r in rows]
    quadrants = np.asarray([r["quadrant"] for r in rows])
    gt_letters = [extract_letter(r["ground_truth"]) for r in rows]
    print(f"\nLoading eval CSV: {args.csv}")
    print(f"  {len(rows)} rows")

    quad_counts = defaultdict(int)
    for q in quadrants:
        quad_counts[q] += 1
    print("  Quadrant counts:")
    for q in QUAD_ORDER:
        print(f"    {q}: {quad_counts[q]}")

    hidden_size = None
    for layer_idx in probe_layers:
        ckpt = os.path.join(args.lens_dir, f"translator_layer_{layer_idx:02d}.pt")
        if os.path.exists(ckpt):
            state = torch.load(ckpt, map_location="cpu")
            if "W" in state:
                hidden_size = int(state["W"].shape[0])
                break
    if hidden_size is None:
        hidden_size = int(getattr(model.config, "hidden_size", 0) or getattr(getattr(model.config, "text_config", None), "hidden_size", 0))
    if hidden_size <= 0:
        raise RuntimeError("Could not infer translator hidden size.")

    translators = {}
    for layer_idx in probe_layers:
        if layer_idx == 0 or layer_idx >= n_layers:
            continue
        ckpt = os.path.join(args.lens_dir, f"translator_layer_{layer_idx:02d}.pt")
        if not os.path.exists(ckpt):
            print(f"  [WARN] No checkpoint for L{layer_idx}; falling back to logit lens")
            continue
        if not checkpoint_compatible(ckpt, hidden_size, model):
            print(f"  [WARN] Incompatible checkpoint for L{layer_idx}; falling back to logit lens")
            continue
        translator = AffineTranslator(hidden_size)
        translator.load_state_dict(checkpoint_state_dict(ckpt))
        translator.eval()
        translators[layer_idx] = translator
    print(f"Loaded {len(translators)} translators: {sorted(translators.keys())}")

    print(f"\nRunning tuned-lens - English ({len(rows)} questions) ...")
    en_ranks, en_probs = run_tuned_lens(
        args, en_texts, gt_letters, model, tokenizer, norm, lm_head,
        translators, probe_layers, ans_ids, label="EN", n_layers=n_layers)

    print(f"\nRunning tuned-lens - Arabic ({len(rows)} questions) ...")
    ar_ranks, ar_probs = run_tuned_lens(
        args, ar_texts, gt_letters, model, tokenizer, norm, lm_head,
        translators, probe_layers, ans_ids, label="AR", n_layers=n_layers)

    out_npz = os.path.join(args.out_dir, "tuned_lens_results.npz")
    np.savez(
        out_npz,
        en_ranks=en_ranks,
        en_probs=en_probs,
        ar_ranks=ar_ranks,
        ar_probs=ar_probs,
        quadrants=quadrants,
        layers=np.asarray(probe_layers),
    )
    print(f"\nSaved -> {out_npz}")

    print("\n-- Mean P(correct) at each probe layer (tuned lens) --")
    header = "  ".join([f"L{l:2d}" for l in probe_layers])
    print(f"{'Quadrant / Lang':<25} {header}")
    for q in QUAD_ORDER:
        mask = quadrants == q
        if not mask.any():
            continue
        for lang_name, probs in [("English", en_probs), ("Arabic", ar_probs)]:
            vals = "  ".join(f"{probs[mask, li].mean():.3f}" for li in range(len(probe_layers)))
            print(f"  {q[:15]:<15} {lang_name:<8} {vals}")
        print()

    fig, ax = plt.subplots(figsize=(10, 6))
    for q in QUAD_ORDER:
        mask = quadrants == q
        if not mask.any():
            continue
        ev = en_probs[mask].mean(axis=0)
        av = ar_probs[mask].mean(axis=0)
        ee = en_probs[mask].std(axis=0) / np.sqrt(mask.sum())
        ae = ar_probs[mask].std(axis=0) / np.sqrt(mask.sum())

        ax.plot(probe_layers, ev, color=QUAD_COLORS[q], lw=2.5, ls="-", marker="o", ms=6)
        ax.fill_between(probe_layers, ev - ee, ev + ee, color=QUAD_COLORS[q], alpha=0.12)
        ax.plot(probe_layers, av, color=QUAD_COLORS[q], lw=2.5, ls="--", marker="^", ms=6)
        ax.fill_between(probe_layers, av - ae, av + ae, color=QUAD_COLORS[q], alpha=0.08)

    ax.set_xlabel("Layer depth", fontsize=12)
    ax.set_ylabel("Mean P(correct answer letter)", fontsize=11)
    ax.set_title(
        f"Tuned Lens: Correct Answer Emergence Across Layers\n{args.model_label} - MedAraBench",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xticks(probe_layers)
    ax.set_xticklabels(
        ["L0\n(emb)" if l == 0 else (f"L{n_layers}\n(final)" if l == n_layers else f"L{l}") for l in probe_layers],
        fontsize=9,
    )
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    quad_handles = [mlines.Line2D([], [], color=QUAD_COLORS[q], lw=2, label=QUAD_LABELS[q]) for q in QUAD_ORDER]
    style_handles = [
        mlines.Line2D([], [], color="grey", lw=2, ls="-", marker="o", ms=6, label="English"),
        mlines.Line2D([], [], color="grey", lw=2, ls="--", marker="^", ms=6, label="Arabic"),
    ]
    ax.legend(handles=quad_handles + style_handles, loc="upper left", fontsize=9, frameon=False)
    plt.tight_layout()
    for ext in [".pdf", ".png"]:
        out_fig = os.path.join(args.out_dir, f"fig_tuned_lens_generic{ext}")
        plt.savefig(out_fig, bbox_inches="tight", dpi=150)
        print(f"Saved -> {out_fig}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["train", "eval"])
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--model_label", default="Model")
    parser.add_argument("--lens_dir", required=True)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--probe_layers", default="")
    parser.add_argument("--train_csv")
    parser.add_argument("--train_n", type=int, default=400)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--reg", type=float, default=1e-4)
    parser.add_argument("--train_batch", type=int, default=4)
    parser.add_argument("--csv")
    parser.add_argument("--out_dir")
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    os.makedirs(args.lens_dir, exist_ok=True)
    if args.mode == "eval":
        if not args.out_dir:
            raise ValueError("--out_dir is required for eval")
        os.makedirs(args.out_dir, exist_ok=True)

    tokenizer = load_tokenizer(args.model_path)
    model = load_causal_lm(args.model_path)
    layers = get_layers(model)
    norm, lm_head = get_norm_and_lm_head(model)
    n_layers = len(layers)
    probe_layers = parse_probe_layers(args.probe_layers, n_layers)
    ans_ids = answer_token_ids(tokenizer)

    print(f"Model label: {args.model_label}")
    print(f"n_layers: {n_layers}")
    print(f"Probe layers: {probe_layers}")
    print(f"Projection device={module_execution_device(norm, lm_head)}, dtype={module_param_dtype(norm, lm_head)}")
    print(f"LM head multiplier={lm_head_multiplier(model):.8g}")

    if args.mode == "train":
        train_translators(args, model, tokenizer, norm, lm_head, n_layers, probe_layers, ans_ids)
    else:
        eval_mode(args, model, tokenizer, norm, lm_head, n_layers, probe_layers, ans_ids)


if __name__ == "__main__":
    main()
