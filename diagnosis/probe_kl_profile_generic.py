"""
probe_kl_profile1.py
--------------------
Diagnostic: per-layer cross-lingual KL divergence profile on the BASE model
(no LoRA, no training).

For each transformer layer ℓ = 0..L-1, computes:
    KL(ℓ) = mean over D_calib of  KL( p^En_ℓ ∥ p^Ar_ℓ )

where p^·_ℓ = softmax( W_U · RMSNorm(h_ℓ) ) is the logit-lens projection
of the final-token hidden state at layer ℓ.

Outputs
-------
  <output_dir>/kl_profile.json   — per-layer KL values + derived thresholds
  <output_dir>/kl_profile.png    — publication-quality plot

Derived quantities (saved in JSON and printed):
  tau         — threshold = mean(KL) + 1·std(KL), computed after L_patch
  L_kl        — first post-L_patch layer where KL(ℓ) >= tau  (active-zone onset)
  L_patch     — provided via --l_patch (from activation patching, default 24)
  lora_window — [L_kl, L_patch]
  kl_probe    — L_kl (recommended KL probe layer)

Usage (single GPU — no torchrun needed):
    python probe_kl_profile1.py \\
        --data_file  datasets/train/splits/val_task1_translated.json \\
        --output_dir outputs/kl_profile_medgemma \\
        --n_examples 300 \\
        --l_patch    40

    # Larger calibration set for publication:
    python probe_kl_profile1.py \\
        --data_file  datasets/train/splits/val_task1_translated.json \\
        --output_dir outputs/kl_profile_medgemma \\
        --n_examples 400 \\
        --l_patch    40 \\
        --model_name google/medgemma-27b-text-it
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
import matplotlib.patches as mpatches

from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

from generic_model_utils import disable_config_cache, disable_model_cache

HF_CACHE = "/scratch/mk8737/huggingface"
os.environ.setdefault("HF_HOME", HF_CACHE)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


# ---------------------------------------------------------------------------
# Tokenizer helpers (reused from train_lora_align.py)
# ---------------------------------------------------------------------------

def load_tokenizer(model_name: str):
    tok = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=HF_CACHE,
        local_files_only=os.path.isdir(model_name),
        trust_remote_code=True,
    )
    if tok.pad_token_id is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    return tok


def tokenize_plain(tok, text: str, max_len: int = 512) -> List[int]:
    ids = tok.encode(text.strip(), add_special_tokens=False)
    bos_id = getattr(tok, "bos_token_id", None)
    if bos_id is not None:
        ids = [bos_id] + ids
    return ids[:max_len]


# ---------------------------------------------------------------------------
# Logit-lens helper (same logic as train_lora_align.py but for bare model)
# ---------------------------------------------------------------------------

def get_norm_and_lmhead(model):
    """Resolve (RMSNorm, lm_head) from the raw (non-PEFT) model."""
    base = model

    # MedGemma 27B text checkpoint loaded as Gemma3ForCausalLM:
    #   base.model.norm
    #   base.lm_head
    if hasattr(base, "model") and hasattr(base.model, "norm") and hasattr(base, "lm_head"):
        return base.model.norm, base.lm_head

    # MedGemma multimodal wrapper / alternate Gemma3 layouts.
    if hasattr(base, "language_model"):
        lm = base.language_model
        if hasattr(lm, "model") and hasattr(lm.model, "norm") and hasattr(lm, "lm_head"):
            return lm.model.norm, lm.lm_head
        if hasattr(lm, "norm") and hasattr(lm, "lm_head"):
            return lm.norm, lm.lm_head

    # Falcon-family layout:
    #   base.transformer.ln_f
    #   base.lm_head
    if hasattr(base, "transformer") and hasattr(base.transformer, "ln_f") and hasattr(base, "lm_head"):
        return base.transformer.ln_f, base.lm_head

    # Falcon-H1 layout:
    #   base.model.final_layernorm
    #   base.lm_head
    if hasattr(base, "model") and hasattr(base.model, "final_layernorm") and hasattr(base, "lm_head"):
        return base.model.final_layernorm, base.lm_head

    if hasattr(base, "final_layernorm") and hasattr(base, "lm_head"):
        return base.final_layernorm, base.lm_head

    # Some GPT-style wrappers expose the backbone under .model/.transformer.
    if (hasattr(base, "model") and hasattr(base.model, "transformer")
            and hasattr(base.model.transformer, "ln_f") and hasattr(base, "lm_head")):
        return base.model.transformer.ln_f, base.lm_head

    # Mistral3ForConditionalGeneration:
    #   base.model (Mistral3Model) → .language_model (MistralModel) → .norm
    #   base.lm_head
    if (hasattr(base, "model") and hasattr(base.model, "language_model")
            and hasattr(base.model.language_model, "norm") and hasattr(base, "lm_head")):
        return base.model.language_model.norm, base.lm_head

    # fallback: walk the tree
    def _tree(mod, prefix="", depth=3):
        if depth == 0:
            return
        for name, child in mod._modules.items():
            print(f"  {prefix}{name}: {type(child).__name__}")
            _tree(child, prefix + "  ", depth - 1)

    print("[LogitLens] ERROR — could not resolve norm/lm_head. Model tree:")
    _tree(base)
    raise AttributeError("Cannot locate final norm/lm_head.")


def load_model(model_name: str):
    cfg = AutoConfig.from_pretrained(
        model_name,
        cache_dir=HF_CACHE,
        local_files_only=os.path.isdir(model_name),
        trust_remote_code=True,
    )
    disable_config_cache(cfg)
    architectures = set(getattr(cfg, "architectures", []) or [])
    loaders = []

    if "Gemma3ForCausalLM" in architectures:
        try:
            from transformers import Gemma3ForCausalLM
            loaders.append(
                (
                    "Gemma3ForCausalLM",
                    lambda: Gemma3ForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=torch.bfloat16,
                        device_map="auto",
                        cache_dir=HF_CACHE,
                        local_files_only=os.path.isdir(model_name),
                        config=cfg,
                    ),
                )
            )
        except Exception:
            pass

    if "Gemma3ForConditionalGeneration" in architectures:
        try:
            from transformers import Gemma3ForConditionalGeneration
            loaders.append(
                (
                    "Gemma3ForConditionalGeneration",
                    lambda: Gemma3ForConditionalGeneration.from_pretrained(
                        model_name,
                        torch_dtype=torch.bfloat16,
                        device_map="auto",
                        cache_dir=HF_CACHE,
                        local_files_only=os.path.isdir(model_name),
                        config=cfg,
                    ),
                )
            )
        except Exception:
            pass

    if "Mistral3ForConditionalGeneration" in architectures:
        try:
            from transformers import Mistral3ForConditionalGeneration
            loaders.append(
                (
                    "Mistral3ForConditionalGeneration",
                    lambda: Mistral3ForConditionalGeneration.from_pretrained(
                        model_name,
                        torch_dtype=torch.bfloat16,
                        device_map="auto",
                        cache_dir=HF_CACHE,
                        local_files_only=os.path.isdir(model_name),
                        config=cfg,
                    ),
                )
            )
        except Exception:
            pass

    loaders.append(
        (
            "AutoModelForCausalLM",
            lambda: AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                cache_dir=HF_CACHE,
                local_files_only=os.path.isdir(model_name),
                trust_remote_code=True,
                config=cfg,
            ),
        )
    )

    for loader_name, loader in loaders:
        try:
            print(f"  Trying {loader_name} ...")
            model = loader()
            disable_model_cache(model)
            return model
        except Exception as exc:
            print(f"  {loader_name} failed: {exc}")
    raise RuntimeError(f"Could not load model: {model_name}")


def module_execution_device(*modules) -> torch.device:
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


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_pairs(data_file: str, n_examples: int, seed: int = 42) -> List[dict]:
    """
    Load up to n_examples parallel (full_text_ar, full_text_en) pairs.
    Skips rows missing either field.
    """
    with open(data_file, "r", encoding="utf-8") as f:
        rows = json.load(f)

    valid = []
    for r in rows:
        ar = (r.get("full_text_ar") or "").strip()
        en = (r.get("full_text_en") or "").strip()
        if ar and en:
            valid.append({"ar": ar, "en": en, "id": r.get("id", "?")})

    print(f"[Data] {len(valid)} valid parallel pairs found in {data_file}")

    rng = random.Random(seed)
    rng.shuffle(valid)
    selected = valid[:n_examples]
    print(f"[Data] Using {len(selected)} examples (seed={seed})")
    return selected


# ---------------------------------------------------------------------------
# Core: per-layer KL profile
# ---------------------------------------------------------------------------

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
    kl_profile[ℓ] = mean KL( p^En_ℓ ∥ p^Ar_ℓ ) over all pairs.

    Uses final-token hidden state at each layer, projected via logit lens.
    """
    model.eval()

    n_pairs = len(pairs)
    n_layers = None
    kl_accum = None
    proj_device = module_execution_device(norm, lm_head)
    print(f"[KL] Projection device={proj_device}, dtype={dtype}")

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

        ar_hidden = ar_out.hidden_states   # tuple: (embedding, L1, ..., L40)
        en_hidden = en_out.hidden_states   # same

        if n_layers is None:
            # hidden_states[0] = embedding output (before any transformer layer)
            # hidden_states[k] = output of transformer block k  (k = 1..L)
            # We profile transformer blocks only: indices 1..L
            n_layers = len(ar_hidden) - 1   # = 40 for Mistral-Small-24B
            kl_accum = np.zeros(n_layers, dtype=np.float64)

        # final token position
        ar_pos = ar_tensor.shape[1] - 1
        en_pos = en_tensor.shape[1] - 1

        for ℓ in range(n_layers):
            hs_idx = ℓ + 1   # skip embedding (index 0)

            h_ar = ar_hidden[hs_idx][0, ar_pos, :].unsqueeze(0).to(
                device=proj_device, dtype=dtype
            )
            h_en = en_hidden[hs_idx][0, en_pos, :].unsqueeze(0).to(
                device=proj_device, dtype=dtype
            )

            logits_ar = lm_head(norm(h_ar))           # (1, vocab)
            logits_en = lm_head(norm(h_en))           # (1, vocab)

            ar_log_probs = F.log_softmax(logits_ar, dim=-1)
            en_probs     = F.softmax(logits_en,     dim=-1)

            kl = F.kl_div(ar_log_probs, en_probs, reduction="batchmean", log_target=False)
            kl_accum[ℓ] += kl.item()

        # free memory each step
        del ar_out, en_out, ar_hidden, en_hidden
        torch.cuda.empty_cache()

    kl_profile = kl_accum / n_pairs
    return kl_profile


# ---------------------------------------------------------------------------
# Threshold + window derivation
# ---------------------------------------------------------------------------

def derive_window(kl_profile: np.ndarray, l_patch: int) -> dict:
    """
    From the KL profile, derive:
      tau       = mean + 1*std over layers after L_patch
      L_kl      = first post-L_patch layer (1-indexed) where KL >= tau
      lora_window = [L_patch, L_kl]
      kl_probe  = L_kl
    """
    n_layers = len(kl_profile)
    if l_patch > n_layers:
        print(f"[WARN] Requested L_patch={l_patch} exceeds n_layers={n_layers}; clamping to L{n_layers}.")
        l_patch = n_layers
    if l_patch < 1:
        print(f"[WARN] Requested L_patch={l_patch} is < 1; clamping to L1.")
        l_patch = 1

    # Candidate layers are strictly after L_patch: L_patch+1..L_final.
    # If L_patch is already final, keep the function defined by using L_final.
    search_start = l_patch + 1
    if search_start > n_layers:
        print(f"[WARN] No layers after L_patch=L{l_patch}; using L{n_layers} for KL threshold/search.")
        search_start = n_layers
    candidate_values = kl_profile[search_start - 1:]
    mu = float(np.mean(candidate_values))
    std = float(np.std(candidate_values))
    tau = mu + std

    # Layers are 1-indexed (L1..L40); kl_profile[0] = L1.
    L_kl = None
    for i in range(search_start - 1, n_layers):
        kl = kl_profile[i]
        if kl >= tau:
            L_kl = i + 1   # convert to 1-indexed paper notation
            break

    if L_kl is None:
        peak_offset = int(np.argmax(candidate_values))
        L_kl = search_start + peak_offset
        print(f"[WARN] No post-L_patch layer reached tau; using post-L_patch peak L{L_kl}.")

    window_start = min(L_kl, l_patch)
    window_end = max(L_kl, l_patch)

    return {
        "tau":         tau,
        "mu":          mu,
        "std":         std,
        "L_kl":        L_kl,
        "L_patch":     l_patch,
        "lora_window": [window_start, window_end],
        "kl_probe":    L_kl,
        "threshold_scope": "post_l_patch",
        "threshold_layers": [search_start, n_layers],
    }


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def make_plot(kl_profile: np.ndarray, window: dict, output_path: str, model_label: str):
    n = len(kl_profile)
    layers = np.arange(1, n + 1)   # L1..L40

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(layers, kl_profile, color="#2c7bb6", linewidth=1.8, label="KL divergence")

    # threshold line
    ax.axhline(window["tau"], color="#d7191c", linestyle="--", linewidth=1.2,
               label=f"post-L_patch τ = μ + σ = {window['tau']:.3f}")

    # L_kl marker
    L_kl   = window["L_kl"]
    L_patch = window["L_patch"]

    ax.axvline(L_kl, color="#fdae61", linestyle=":", linewidth=1.5,
               label=f"L_kl = L{L_kl}  (active-zone onset)")
    ax.axvline(L_patch, color="#1a9641", linestyle=":", linewidth=1.5,
               label=f"L_patch = L{L_patch}  (causal boundary)")

    # shaded LoRA window
    window_start, window_end = window["lora_window"]
    ax.axvspan(window_start, window_end, alpha=0.12, color="#fdae61",
               label=f"LoRA window L{window_start}–L{window_end}")

    ax.set_xlabel("Transformer Layer", fontsize=12)
    ax.set_ylabel("KL Divergence (nats)", fontsize=12)
    ax.set_title(
        "Cross-Lingual KL Divergence Profile (Arabic vs. English, Logit Lens)\n"
        f"{model_label} · Base Model · MedAraBench val",
        fontsize=11,
    )
    ax.set_xlim(1, n)
    ax.set_xticks(np.arange(0, n + 1, 5))
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved → {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute per-layer cross-lingual KL divergence profile (no training)."
    )
    parser.add_argument("--data_file",   required=True,
                        help="JSON file with full_text_ar + full_text_en fields.")
    parser.add_argument("--output_dir",  required=True)
    parser.add_argument("--model_name",  default="google/medgemma-27b-text-it")
    parser.add_argument("--model_label", default=None,
                        help="Display name used in plots. Defaults to basename/model_name.")
    parser.add_argument("--n_examples",  type=int, default=300,
                        help="Number of parallel pairs to average over.")
    parser.add_argument("--max_len",     type=int, default=512,
                        help="Max token length per text (Arabic or English).")
    parser.add_argument("--l_patch",     type=int, default=40,
                        help="L_patch from activation patching (1-indexed paper notation). "
                             "Default 40 = MedGemma patching onset.")
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()
    model_label = args.model_label or os.path.basename(os.path.normpath(args.model_name))

    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.seed)

    # ── Device ──────────────────────────────────────────────────────────────
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using {device}")

    # ── Tokenizer ────────────────────────────────────────────────────────────
    print("[INFO] Loading tokenizer...")
    tokenizer = load_tokenizer(args.model_name)

    # ── Data ─────────────────────────────────────────────────────────────────
    pairs = load_pairs(args.data_file, args.n_examples, args.seed)
    if not pairs:
        raise ValueError("No valid parallel pairs found. Check full_text_ar / full_text_en fields.")

    # ── Model ────────────────────────────────────────────────────────────────
    print("[INFO] Loading model (base, no LoRA)...")
    model = load_model(args.model_name)
    model.eval()

    norm, lm_head = get_norm_and_lmhead(model)
    print(f"[INFO] Model loaded. Profiling {len(pairs)} examples across all layers...")

    # ── KL profile ───────────────────────────────────────────────────────────
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
        bar = "█" * int(kl * 20)
        print(f"  L{i+1:>2}  {kl:.4f}  {bar}")

    # ── Derive window ────────────────────────────────────────────────────────
    window = derive_window(kl_profile, args.l_patch)
    print(f"\n[Window] τ = {window['tau']:.4f}  (μ={window['mu']:.4f}, σ={window['std']:.4f}; layers L{window['threshold_layers'][0]}-L{window['threshold_layers'][1]})")
    print(f"[Window] L_kl    = L{window['L_kl']}  (first post-L_patch layer with KL >= τ)")
    print(f"[Window] L_patch = L{window['L_patch']}  (from activation patching)")
    print(f"[Window] → LoRA window : L{window['lora_window'][0]}–L{window['lora_window'][1]}")
    print(f"[Window] → KL probe    : L{window['kl_probe']}")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    result = {
        "model":      args.model_name,
        "model_label": model_label,
        "n_examples": len(pairs),
        "seed":       args.seed,
        "kl_profile": kl_profile.tolist(),
        "window":     window,
    }
    json_path = os.path.join(args.output_dir, "kl_profile.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"[Saved] {json_path}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_path = os.path.join(args.output_dir, "kl_profile.png")
    make_plot(kl_profile, window, plot_path, model_label)


if __name__ == "__main__":
    main()
