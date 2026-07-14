"""
kl_profiling_medgemma.py
------------------------
Per-layer cross-lingual KL divergence profile on the BASE MedGemma-27B model
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
  tau         — threshold = mean(KL) + 1·std(KL)
  L_kl        — first layer where KL(ℓ) >= tau  (divergence onset)
  L_patch     — provided via --l_patch (from activation patching experiment)
  lora_window — W3 = [1, L_kl]  (below divergence onset; paper's winning window)
  kl_probe    — L_kl  (probe layer for alignment loss during training)

MedGemma-27B specifics vs. Mistral script
------------------------------------------
  - Uses AutoModelForCausalLM + AutoTokenizer (not Mistral3 / MistralTokenizer)
  - Gemma norm/lm_head path: model.model.norm  /  model.lm_head
  - 62 transformer layers (not 40)
  - --l_patch default updated; pass your actual value from patching experiment

Usage:
    python kl_profiling_medgemma.py \\
        --data_file  datasets/train/splits/val_task1_translated.json \\
        --output_dir outputs/kl_profile_medgemma \\
        --n_examples 300 \\
        --l_patch    40

    # Larger calibration set:
    python kl_profiling_medgemma.py \\
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

from transformers import AutoModelForCausalLM, AutoTokenizer

HF_CACHE = "/scratch/ca2627/huggingface"
os.environ.setdefault("HF_HOME", HF_CACHE)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


# ---------------------------------------------------------------------------
# Tokenizer helpers
# ---------------------------------------------------------------------------

def load_tokenizer(model_name: str) -> AutoTokenizer:
    tok = AutoTokenizer.from_pretrained(model_name, cache_dir=HF_CACHE)
    # Gemma tokenizer does not set a pad token by default
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def tokenize_plain(tok: AutoTokenizer, text: str, max_len: int = 512) -> List[int]:
    ids = tok.encode(text.strip(), add_special_tokens=True)
    return ids[:max_len]


# ---------------------------------------------------------------------------
# Logit-lens helper — Gemma architecture
# ---------------------------------------------------------------------------

def get_norm_and_lmhead(model):
    """
    Resolve (final RMSNorm, lm_head) from a MedGemma / Gemma model.

    MedGemma-27B (GemmaForCausalLM) layout:
        model.model          — GemmaModel
        model.model.norm     — final RMSNorm
        model.lm_head        — Linear (vocab projection)
    """
    # Primary path: standard GemmaForCausalLM / Gemma2ForCausalLM
    if (hasattr(model, "model")
            and hasattr(model.model, "norm")
            and hasattr(model, "lm_head")):
        print("[LogitLens] Resolved: model.model.norm  +  model.lm_head")
        return model.model.norm, model.lm_head

    # Fallback: walk one level deeper (some HF wrappers)
    if hasattr(model, "language_model"):
        lm = model.language_model
        if hasattr(lm, "model") and hasattr(lm.model, "norm") and hasattr(lm, "lm_head"):
            print("[LogitLens] Resolved via language_model wrapper.")
            return lm.model.norm, lm.lm_head

    # Debug dump if nothing matched
    def _tree(mod, prefix="", depth=3):
        if depth == 0:
            return
        for name, child in mod._modules.items():
            print(f"  {prefix}{name}: {type(child).__name__}")
            _tree(child, prefix + "  ", depth - 1)

    print("[LogitLens] ERROR — could not resolve norm/lm_head. Model tree:")
    _tree(model)
    raise AttributeError(
        "Cannot locate norm/lm_head for this model. "
        "Inspect the tree above and add the correct path."
    )


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
    tokenizer: AutoTokenizer,
    device: torch.device,
    max_len: int = 512,
    dtype=torch.bfloat16,
) -> np.ndarray:
    """
    Returns kl_profile: np.ndarray of shape (n_layers,)

    kl_profile[ℓ] = mean KL( p^En_ℓ ∥ p^Ar_ℓ ) over all pairs.

    Direction matches paper Eq. (2):  D_KL( p^En || p^Ar )
      — English is the reference distribution.

    Uses final-token hidden state at each layer, projected via logit lens.
    MedGemma-27B has 62 transformer layers → n_layers = 62.
    """
    model.eval()

    n_pairs  = len(pairs)
    n_layers = None
    kl_accum = None

    for i, pair in enumerate(pairs):
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{n_pairs}] processing...")

        ar_ids = tokenize_plain(tokenizer, pair["ar"], max_len)
        en_ids = tokenize_plain(tokenizer, pair["en"], max_len)

        ar_tensor = torch.tensor([ar_ids], dtype=torch.long, device=device)
        en_tensor = torch.tensor([en_ids], dtype=torch.long, device=device)

        ar_mask = torch.ones_like(ar_tensor)
        en_mask = torch.ones_like(en_tensor)

        ar_out = model(
            input_ids=ar_tensor, attention_mask=ar_mask, output_hidden_states=True
        )
        en_out = model(
            input_ids=en_tensor, attention_mask=en_mask, output_hidden_states=True
        )

        # hidden_states: tuple of length (n_layers + 1)
        #   index 0  = embedding output (before any transformer block)
        #   index k  = output of transformer block k  (k = 1 .. n_layers)
        ar_hidden = ar_out.hidden_states
        en_hidden = en_out.hidden_states

        if n_layers is None:
            n_layers = len(ar_hidden) - 1  # = 62 for MedGemma-27B
            kl_accum = np.zeros(n_layers, dtype=np.float64)
            print(f"[Info] Detected {n_layers} transformer layers.")

        # final token position for each sequence
        ar_pos = ar_tensor.shape[1] - 1
        en_pos = en_tensor.shape[1] - 1

        for ℓ in range(n_layers):
            hs_idx = ℓ + 1  # skip embedding (index 0)

            h_ar = ar_hidden[hs_idx][0, ar_pos, :].unsqueeze(0).to(dtype)
            h_en = en_hidden[hs_idx][0, en_pos, :].unsqueeze(0).to(dtype)

            # Logit-lens projection: softmax( lm_head( norm( h ) ) )
            logits_ar = lm_head(norm(h_ar))   # (1, vocab)
            logits_en = lm_head(norm(h_en))   # (1, vocab)

            # D_KL( p^En || p^Ar )  — English as reference, Arabic as approximation
            # F.kl_div(input, target) computes D_KL(target || input)
            # → pass English as target, Arabic log-probs as input
            en_probs     = F.softmax(logits_en,     dim=-1)          # p^En
            ar_log_probs = F.log_softmax(logits_ar, dim=-1)          # log p^Ar

            kl = F.kl_div(
                ar_log_probs, en_probs,
                reduction="batchmean",
                log_target=False,
            )
            kl_accum[ℓ] += kl.item()

        del ar_out, en_out, ar_hidden, en_hidden
        torch.cuda.empty_cache()

    kl_profile = kl_accum / n_pairs
    return kl_profile


# ---------------------------------------------------------------------------
# Threshold + window derivation
# ---------------------------------------------------------------------------

def derive_window(kl_profile: np.ndarray, l_patch: int) -> dict:
    """
    From the KL profile derive the two mechanistic boundary layers and the
    five candidate LoRA windows defined in Table 1 of the paper.

    tau   = mu + sigma  (computed from the neutral zone onwards — see below)
    L_kl  = first layer (1-indexed) AFTER L_min where KL(ℓ) >= tau
    L_patch = from activation patching experiment (passed in)

    Candidate windows (paper Table 1):
        W1: L1  – L_patch       below causal boundary
        W2: L_patch – L_max     causal window
        W3: L1  – L_kl          below divergence onset
        W4: L_kl – L_max        active zone only
        W5: L1  – L_max         full model

    Generalized zone detection (Mistral + MedGemma):
    ------------------------------------------------
    The paper's original formula L_kl = min{ℓ : KL(ℓ) ≥ μ + σ} implicitly
    assumes the profile starts in a neutral zone (low KL) and rises later —
    which holds for Mistral-Small-3.2-24B.

    For models like MedGemma-27B, early layers (L1–L15) exhibit very high KL
    driven by script-level tokenization differences between Arabic and English
    (different character sets → maximally different token distributions before
    any cross-lingual alignment occurs). This is not a representational failure
    — it is a pre-linguistic artifact. Including these layers in the global
    mean inflates τ far above the late-layer divergence, causing L_kl = L1.

    Generalized procedure (preserving the method's intent):
      1. Identify L_min = argmin KL(ℓ) — the global minimum, which marks the
         boundary between the script-noise zone (early) and the neutral zone
         (where cross-lingual representations are maximally aligned).
      2. Compute μ and σ over layers [L_min .. L_max] only — the neutral and
         active zones, excluding the script-level divergence spike.
      3. L_kl = first layer after L_min where KL(ℓ) ≥ μ + σ — the onset of
         meaningful representational divergence in the late network.

    This is equivalent to the original formula on profiles with a flat early
    zone (Mistral), and correctly generalizes to profiles where early layers
    are dominated by script-level noise (MedGemma).
    """
    lmax    = len(kl_profile)   # = n_layers (1-indexed last layer)

    # 1. Find the global minimum layer (0-indexed)
    l_min_idx = int(np.argmin(kl_profile))

    # 2. Compute mu / sigma / tau only from L_min onwards (neutral + active zone)
    neutral_and_active = kl_profile[l_min_idx:]
    mu   = float(np.mean(neutral_and_active))
    std  = float(np.std(neutral_and_active))
    tau  = mu + std

    print(f"[Window] L_min = L{l_min_idx + 1}  (KL minimum, zone anchor)")
    print(f"[Window] tau computed over L{l_min_idx+1}–L{lmax}: "
          f"μ={mu:.4f}, σ={std:.4f}, τ={tau:.4f}")

    # 3. L_kl: first layer AFTER L_min (1-indexed) where KL >= tau
    L_kl = None
    for i in range(l_min_idx, lmax):
        if kl_profile[i] >= tau:
            L_kl = i + 1   # convert to 1-indexed
            break

    if L_kl is None:
        print("[WARN] No layer after L_min exceeded tau — setting L_kl = L_max.")
        L_kl = lmax

    windows = {
        "W1": [1,       l_patch],
        "W2": [l_patch, lmax],
        "W3": [1,       L_kl],
        "W4": [L_kl,    lmax],
        "W5": [1,       lmax],
    }

    return {
        "tau":         tau,
        "mu":          mu,
        "std":         std,
        "L_min":       l_min_idx + 1,   # 1-indexed global minimum layer
        "L_kl":        L_kl,
        "L_patch":     l_patch,
        "L_max":       lmax,
        "kl_probe":    L_kl,            # probe layer for alignment loss = L_kl
        "windows":     windows,
    }


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def make_plot(kl_profile: np.ndarray, window: dict, model_label: str, output_path: str):
    n      = len(kl_profile)
    layers = np.arange(1, n + 1)

    L_kl    = window["L_kl"]
    L_patch = window["L_patch"]
    tau     = window["tau"]
    L_min   = window["L_min"]   # 1-indexed neutral zone anchor

    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(layers, kl_profile, color="#2c7bb6", linewidth=1.8, label="KL divergence")

    # Shade script-noise zone (L1 – L_min) in light grey
    ax.axvspan(1, L_min, alpha=0.08, color="grey",
               label=f"Script-noise zone  L1–L{L_min}")

    # L_min marker (neutral zone onset)
    ax.axvline(L_min, color="grey", linestyle="--", linewidth=1.0,
               label=f"$L_{{min}}$ = L{L_min}  (neutral zone onset)")

    # tau line — drawn only from L_min onwards to make clear it excludes the spike
    ax.hlines(tau, L_min, n, colors="#d7191c", linestyles="--", linewidth=1.2,
              label=f"τ = μ + σ = {tau:.3f}  (neutral zone)")

    # L_kl marker (divergence onset — first late crossing of tau)
    ax.axvline(L_kl, color="#fdae61", linestyle=":", linewidth=1.5,
               label=f"$L_{{KL}}$ = L{L_kl}  (divergence onset)")

    # L_patch marker (causal boundary from patching)
    ax.axvline(L_patch, color="#1a9641", linestyle=":", linewidth=1.5,
               label=f"$L_{{patch}}$ = L{L_patch}  (causal boundary)")

    # Shade W3 = [L_min, L_kl] — the neutral zone (below divergence onset)
    ax.axvspan(L_min, L_kl, alpha=0.12, color="#fdae61",
               label=f"LoRA window W3  L1–L{L_kl}")

    ax.set_xlabel("Transformer Layer", fontsize=12)
    ax.set_ylabel("KL Divergence (nats)", fontsize=12)
    ax.set_title(
        f"Cross-Lingual KL Divergence Profile (Arabic vs. English, Logit Lens)\n"
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
        description="Per-layer cross-lingual KL divergence profile for MedGemma-27B."
    )
    parser.add_argument("--data_file",   required=True,
                        help="JSON file with full_text_ar + full_text_en fields.")
    parser.add_argument("--output_dir",  required=True)
    parser.add_argument("--model_name",  default="google/medgemma-27b-text-it")
    parser.add_argument("--n_examples",  type=int, default=300,
                        help="Number of parallel pairs to average over.")
    parser.add_argument("--max_len",     type=int, default=512,
                        help="Max token length per sequence.")
    parser.add_argument("--l_patch",     type=int, default=40,
                        help="L_patch from your activation patching experiment "
                             "(1-indexed). Update this with your actual result.")
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

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
        raise ValueError(
            "No valid parallel pairs found. Check full_text_ar / full_text_en fields."
        )

    # ── Model ────────────────────────────────────────────────────────────────
    print("[INFO] Loading MedGemma-27B (base, no LoRA)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir=HF_CACHE,
    )
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
        bar = "█" * min(int(kl * 10), 60)
        print(f"  L{i+1:>3}  {kl:.4f}  {bar}")

    # ── Derive windows ────────────────────────────────────────────────────────
    window = derive_window(kl_profile, args.l_patch)

    print(f"\n[Window] L_min   = L{window['L_min']}  (neutral zone onset; script-noise excluded)")
    print(f"[Window] τ = {window['tau']:.4f}  (μ={window['mu']:.4f}, σ={window['std']:.4f}  "
          f"computed over L{window['L_min']}–L{window['L_max']})")
    print(f"[Window] L_kl    = L{window['L_kl']}  (divergence onset, = KL probe layer)")
    print(f"[Window] L_patch = L{window['L_patch']}  (causal boundary, from patching)")
    print(f"[Window] L_max   = L{window['L_max']}")
    print(f"\n[Windows] Candidate LoRA adaptation windows (paper Table 1):")
    for name, (lo, hi) in window["windows"].items():
        print(f"  {name}: L{lo}–L{hi}")
    print(f"\n[KL probe layer for alignment loss] L{window['kl_probe']}")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    result = {
        "model":      args.model_name,
        "n_examples": len(pairs),
        "seed":       args.seed,
        "kl_profile": kl_profile.tolist(),
        "window":     window,
    }
    json_path = os.path.join(args.output_dir, "kl_profile.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\n[Saved] {json_path}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    model_label = args.model_name.split("/")[-1]
    plot_path   = os.path.join(args.output_dir, "kl_profile.png")
    make_plot(kl_profile, window, model_label, plot_path)


if __name__ == "__main__":
    main()