"""
probe_kl_profile_medgemma.py
----------------------------
Diagnostic: per-layer cross-lingual KL divergence profile on MedGemma
(base model, no LoRA, no training).

For each transformer layer l = 0..L-1, computes:
    KL(l) = mean over D_calib of KL( p^En_l || p^Ar_l )

where p^._l = softmax( W_U · RMSNorm(h_l) ) is the logit-lens projection
of the final-token hidden state at layer l.

This is a MedGemma adaptation of probe_kl_profile.py:
  - AutoTokenizer instead of mistral_common tokenizer
  - Gemma3ForCausalLM-first loading for the 27B text checkpoint
  - MedGemma/Gemma3 architecture path resolution for final norm + lm_head

Outputs
-------
  <output_dir>/kl_profile.json   — per-layer KL values + derived thresholds
  <output_dir>/kl_profile.png    — publication-quality plot

Usage:
    python probe_kl_profile_medgemma.py \
        --data_file  datasets/train/splits/val_task1_translated.json \
        --output_dir outputs/kl_profile_medgemma \
        --model_name /scratch/mk8737/huggingface/models--google--medgemma-27b-text-it/snapshots/<hash> \
        --n_examples 300 \
        --l_patch    40
"""

import os
import json
import argparse
import random
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModelForCausalLM


HF_CACHE = "/scratch/mk8737/huggingface"
os.environ.setdefault("HF_HOME", HF_CACHE)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

def load_tokenizer(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def tokenize_plain(tokenizer, text: str, max_len: int = 512) -> List[int]:
    """
    Plain-text tokenization for KL profiling.
    Mirrors the original script's intent: BOS, no chat template, no EOS.
    """
    text = (text or "").strip()
    ids = tokenizer.encode(text, add_special_tokens=False)
    bos_id = getattr(tokenizer, "bos_token_id", None)
    if bos_id is not None:
        ids = [bos_id] + ids
    return ids[:max_len]


def load_model_and_projection(model_path: str):
    print(f"[INFO] Loading tokenizer from {model_path} ...")
    tokenizer = load_tokenizer(model_path)

    print(f"[INFO] Loading MedGemma model from {model_path} ...")
    model = None
    for loader_name, loader in [
        (
            "Gemma3ForCausalLM",
            lambda: __import__("transformers", fromlist=["Gemma3ForCausalLM"])
            .Gemma3ForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                local_files_only=True,
            ),
        ),
        (
            "Gemma3ForConditionalGeneration",
            lambda: __import__("transformers", fromlist=["Gemma3ForConditionalGeneration"])
            .Gemma3ForConditionalGeneration.from_pretrained(
                model_path,
                dtype=torch.bfloat16,
                device_map="auto",
                local_files_only=True,
            ),
        ),
        (
            "AutoModelForCausalLM",
            lambda: AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                local_files_only=True,
                trust_remote_code=True,
            ),
        ),
    ]:
        try:
            print(f"  Trying {loader_name} ...")
            model = loader()
            print(f"  Loaded via {loader_name}")
            break
        except Exception as exc:
            print(f"  {loader_name} failed: {exc}")

    if model is None:
        raise RuntimeError("Could not load MedGemma.")

    model.eval()
    print(f"[INFO] Model type: {type(model).__name__}")

    def _get_norm(m):
        for path, fn in [
            ("model.language_model.model.norm", lambda x: x.language_model.model.norm),
            ("model.language_model.norm", lambda x: x.language_model.norm),
            ("model.model.norm", lambda x: x.model.norm),
            ("model.model.language_model.model.norm", lambda x: x.model.language_model.model.norm),
        ]:
            try:
                obj = fn(m)
                if obj is not None:
                    print(f"[INFO] final_norm at: {path}")
                    return obj
            except AttributeError:
                pass
        raise AttributeError("Cannot locate final RMSNorm for MedGemma.")

    def _get_lm_head(m):
        for path, fn in [
            ("model.language_model.lm_head", lambda x: x.language_model.lm_head),
            ("model.lm_head", lambda x: x.lm_head),
            ("model.model.lm_head", lambda x: x.model.lm_head),
            ("model.model.language_model.lm_head", lambda x: x.model.language_model.lm_head),
        ]:
            try:
                obj = fn(m)
                if obj is not None:
                    print(f"[INFO] lm_head at: {path}  shape={obj.weight.shape}")
                    return obj
            except AttributeError:
                pass
        raise AttributeError("Cannot locate lm_head for MedGemma.")

    norm = _get_norm(model)
    lm_head = _get_lm_head(model)
    return model, tokenizer, norm, lm_head


def load_pairs(data_file: str, n_examples: int, seed: int = 42) -> List[dict]:
    """
    Load up to n_examples parallel (full_text_ar, full_text_en) pairs.
    Skips rows missing either field.
    """
    with open(data_file, "r", encoding="utf-8") as f:
        rows = json.load(f)

    valid = []
    for row in rows:
        ar = (row.get("full_text_ar") or "").strip()
        en = (row.get("full_text_en") or "").strip()
        if ar and en:
            valid.append({"ar": ar, "en": en, "id": row.get("id", "?")})

    print(f"[Data] {len(valid)} valid parallel pairs found in {data_file}")

    rng = random.Random(seed)
    rng.shuffle(valid)
    selected = valid[:n_examples]
    print(f"[Data] Using {len(selected)} examples (seed={seed})")
    return selected


@torch.no_grad()
def compute_kl_profile(
    model,
    norm,
    lm_head,
    pairs: List[dict],
    tokenizer,
    device: torch.device,
    max_len: int = 512,
    dtype=torch.bfloat16,
) -> np.ndarray:
    """
    Returns kl_profile: np.ndarray of shape (n_layers,)
    kl_profile[l] = mean KL( p^En_l || p^Ar_l ) over all pairs.

    Uses final-token hidden state at each layer, projected via logit lens.
    """
    model.eval()

    n_pairs = len(pairs)
    n_layers = None
    kl_accum = None
    proj_device = lm_head.weight.device

    for i, pair in enumerate(pairs):
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{n_pairs}] processing...")

        ar_ids = tokenize_plain(tokenizer, pair["ar"], max_len)
        en_ids = tokenize_plain(tokenizer, pair["en"], max_len)

        ar_tensor = torch.tensor([ar_ids], dtype=torch.long, device=device)
        en_tensor = torch.tensor([en_ids], dtype=torch.long, device=device)

        ar_mask = torch.ones_like(ar_tensor)
        en_mask = torch.ones_like(en_tensor)

        ar_out = model(input_ids=ar_tensor, attention_mask=ar_mask, output_hidden_states=True)
        en_out = model(input_ids=en_tensor, attention_mask=en_mask, output_hidden_states=True)

        ar_hidden = ar_out.hidden_states
        en_hidden = en_out.hidden_states

        if n_layers is None:
            n_layers = len(ar_hidden) - 1
            kl_accum = np.zeros(n_layers, dtype=np.float64)
            print(f"[INFO] Profiling {n_layers} transformer layers")

        ar_pos = ar_tensor.shape[1] - 1
        en_pos = en_tensor.shape[1] - 1

        for layer_idx in range(n_layers):
            hs_idx = layer_idx + 1

            h_ar = ar_hidden[hs_idx][0, ar_pos, :].unsqueeze(0).to(
                device=proj_device, dtype=dtype
            )
            h_en = en_hidden[hs_idx][0, en_pos, :].unsqueeze(0).to(
                device=proj_device, dtype=dtype
            )

            logits_ar = lm_head(norm(h_ar))
            logits_en = lm_head(norm(h_en))

            ar_log_probs = F.log_softmax(logits_ar.float(), dim=-1)
            en_probs = F.softmax(logits_en.float(), dim=-1)

            kl = F.kl_div(ar_log_probs, en_probs, reduction="batchmean", log_target=False)
            kl_accum[layer_idx] += kl.item()

        del ar_out, en_out, ar_hidden, en_hidden
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return kl_accum / n_pairs


def derive_window(kl_profile: np.ndarray, l_patch: int) -> dict:
    n_layers = len(kl_profile)
    if l_patch > n_layers:
        print(f"[WARN] Requested L_patch={l_patch} exceeds n_layers={n_layers}; clamping to L{n_layers}.")
        l_patch = n_layers
    if l_patch < 1:
        print(f"[WARN] Requested L_patch={l_patch} is < 1; clamping to L1.")
        l_patch = 1

    search_start = l_patch + 1
    if search_start > n_layers:
        print(f"[WARN] No layers after L_patch=L{l_patch}; using L{n_layers} for KL threshold/search.")
        search_start = n_layers
    candidate_values = kl_profile[search_start - 1:]
    mu = float(np.mean(candidate_values))
    std = float(np.std(candidate_values))
    tau = mu + std

    L_kl = None
    for i in range(search_start - 1, n_layers):
        kl = kl_profile[i]
        if kl >= tau:
            L_kl = i + 1
            break

    if L_kl is None:
        peak_offset = int(np.argmax(candidate_values))
        L_kl = search_start + peak_offset
        print(f"[WARN] No post-L_patch layer reached tau; using post-L_patch peak L{L_kl}.")

    window_start = min(L_kl, l_patch)
    window_end = max(L_kl, l_patch)

    return {
        "tau": tau,
        "mu": mu,
        "std": std,
        "L_kl": L_kl,
        "L_patch": l_patch,
        "lora_window": [window_start, window_end],
        "kl_probe": L_kl,
        "threshold_scope": "post_l_patch",
        "threshold_layers": [search_start, n_layers],
    }


def make_plot(kl_profile: np.ndarray, window: dict, output_path: str):
    n = len(kl_profile)
    layers = np.arange(1, n + 1)

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(layers, kl_profile, color="#2c7bb6", linewidth=1.8, label="KL divergence")
    ax.axhline(
        window["tau"],
        color="#d7191c",
        linestyle="--",
        linewidth=1.2,
        label=f"post-L_patch tau = mu + sigma = {window['tau']:.3f}",
    )

    L_kl = window["L_kl"]
    L_patch = window["L_patch"]

    ax.axvline(
        L_kl,
        color="#fdae61",
        linestyle=":",
        linewidth=1.5,
        label=f"L_kl = L{L_kl}  (active-zone onset)",
    )
    ax.axvline(
        L_patch,
        color="#1a9641",
        linestyle=":",
        linewidth=1.5,
        label=f"L_patch = L{L_patch}  (causal boundary)",
    )
    window_start, window_end = window["lora_window"]
    ax.axvspan(window_start, window_end, alpha=0.12, color="#fdae61", label=f"LoRA window L{window_start}-L{window_end}")

    ax.set_xlabel("Transformer Layer", fontsize=12)
    ax.set_ylabel("KL Divergence (nats)", fontsize=12)
    ax.set_title(
        "Cross-Lingual KL Divergence Profile (Arabic vs. English, Logit Lens)\n"
        "MedGemma-27B-Text-It · Base Model · MedAraBench val",
        fontsize=11,
    )
    ax.set_xlim(1, n)
    ax.set_xticks(np.arange(0, n + 1, 5))
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved -> {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute per-layer cross-lingual KL divergence profile (no training)."
    )
    parser.add_argument("--data_file",   required=True,
                        help="JSON file with full_text_ar + full_text_en fields.")
    parser.add_argument("--output_dir",  required=True)
    parser.add_argument("--model_name",  default="google/medgemma-27b-text-it")
    parser.add_argument("--n_examples",  type=int, default=300,
                        help="Number of parallel pairs to average over.")
    parser.add_argument("--max_len",     type=int, default=512,
                        help="Max token length per text (Arabic or English).")
    parser.add_argument("--l_patch",     type=int, default=40,
                        help="L_patch from activation patching (1-indexed paper notation). "
                             "Default 40 = MedGemma patching onset.")
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using {device}")

    pairs = load_pairs(args.data_file, args.n_examples, args.seed)
    if not pairs:
        raise ValueError("No valid parallel pairs found. Check full_text_ar / full_text_en fields.")

    model, tokenizer, norm, lm_head = load_model_and_projection(args.model_name)
    print(f"[INFO] Model loaded. Profiling {len(pairs)} examples across all layers...")

    kl_profile = compute_kl_profile(
        model=model,
        norm=norm,
        lm_head=lm_head,
        pairs=pairs,
        tokenizer=tokenizer,
        device=device,
        max_len=args.max_len,
        dtype=torch.bfloat16,
    )

    print("\n[Results] Per-layer KL divergence:")
    for i, kl in enumerate(kl_profile):
        bar = "#" * int(kl * 20)
        print(f"  L{i+1:>2}  {kl:.4f}  {bar}")

    window = derive_window(kl_profile, args.l_patch)
    print(f"\n[Window] tau = {window['tau']:.4f}  (mu={window['mu']:.4f}, sigma={window['std']:.4f}; layers L{window['threshold_layers'][0]}-L{window['threshold_layers'][1]})")
    print(f"[Window] L_kl    = L{window['L_kl']}  (first post-L_patch layer with KL >= tau)")
    print(f"[Window] L_patch = L{window['L_patch']}  (from activation patching)")
    print(f"[Window] -> LoRA window : L{window['lora_window'][0]}-L{window['lora_window'][1]}")
    print(f"[Window] -> KL probe    : L{window['kl_probe']}")

    result = {
        "model": args.model_name,
        "n_examples": len(pairs),
        "seed": args.seed,
        "kl_profile": kl_profile.tolist(),
        "window": window,
    }
    json_path = os.path.join(args.output_dir, "kl_profile.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"[Saved] {json_path}")

    plot_path = os.path.join(args.output_dir, "kl_profile.png")
    make_plot(kl_profile, window, plot_path)


if __name__ == "__main__":
    main()
