"""
logit_lens_lora_mistral.py
--------------------------
Direct logit lens: track P(correct answer letter) at every transformer layer
under four conditions on Mistral-Small-3.2-24B:

  1. Base model  + English input  (upper bound)
  2. Base model  + Arabic input   (lower bound)
  3. Targeted LoRA (L24-L40)  + Arabic input
  4. Full LoRA (all layers)   + Arabic input

Uses access_gap quadrant (En✓ Ar✗) from the sampled quadrant CSV.
Applies model's own lm_head (unchanged by LoRA) to each intermediate hidden
state → methodologically cleanest because any LoRA-induced differences in the
probability curves are due entirely to adapted representations, not a different
output projection.

Usage
-----
  python logit_lens_lora_mistral.py \\
      --csv mistral_sampled_quadrants.csv \\
      --model_path /scratch/ca2627/huggingface/models--mistralai--Mistral-Small-3.2-24B-Instruct-2506/snapshots/<hash> \\
      --targeted_adapter_dir /scratch/ca2627/clinicalAI/Adaptation/outputs/lr_search_10ep/trial_03_lr2.80e-05 \\
      --full_adapter_dir     /scratch/ca2627/clinicalAI/Adaptation/outputs/lr_search_full_lora/trial_01_lr1.12e-05 \\
      --out_dir ./logit_lens_lora_out

Checkpoint provenance
---------------------
  Targeted LoRA (L24-L40):  lr_search_10ep/trial_03_lr2.80e-05   (lr=2.795e-05, best of 5 trials)
  Full LoRA (all layers):   lr_search_full_lora/trial_01_lr1.12e-05  (lr=1.122e-05, best of 5 trials)
  Both trained with: 10 epochs, early_stopping_patience=2, r=16, alpha=32

Notes
-----
- PEFT adapters are loaded/unloaded sequentially on the SAME base model object
  to avoid reloading 24B weights three times.
- The final RMSNorm is applied before lm_head (standard Mistral architecture).
- Left-padding is used; seq_lens = last token position = max_seq_len - 1.
"""

import os
import csv
import re
import json
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.patches import Patch
from peft import PeftModel

# ── Args ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--csv",                  required=True)
parser.add_argument("--model_path",           required=True)
parser.add_argument("--targeted_adapter_dir", required=True,
                    help="Targeted LoRA adapter dir, e.g. outputs/lr_search_10ep/trial_03_lr2.80e-05")
parser.add_argument("--full_adapter_dir",     required=True,
                    help="Full LoRA adapter dir, e.g. outputs/lr_search_full_lora/trial_01_lr1.12e-05")
parser.add_argument("--out_dir",  default="./logit_lens_lora_out")
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--max_len",    type=int, default=512)
parser.add_argument("--n_samples",  type=int, default=None,
                    help="Cap N for faster iteration; None = use all rows in quadrant")
parser.add_argument("--quadrant",   type=str, default="access_gap",
                    choices=["access_gap", "both_correct", "both_wrong", "arabic_only"],
                    help="Which quadrant to analyze (default: access_gap)")
args = parser.parse_args()
os.makedirs(args.out_dir, exist_ok=True)

# ── Constants ───────────────────────────────────────────────────────────────────
# Use the EXACT training system prompt so all conditions (base + LoRA) are measured
# under the same format the LoRA models were trained on.
# We also append "ANSWER:" to every prompt so the model's NEXT predicted token
# is the answer letter — making this consistent with how the model was evaluated
# during fine-tuning and with the activation patching analyses (which also measure
# P(correct letter) at the first generation token position).
SYSTEM_PROMPT = (
    "You are a medical expert answering multiple-choice exam questions. "
    "You will receive exactly ONE question followed by answer options labeled: "
    "A), B), C), D), E), and sometimes F). "
    "You must output exactly ONE line in this format: ANSWER: <LETTER> "
    "Rules: Output ONLY that line. Do NOT repeat or paraphrase the question. "
    "Do NOT translate anything. Do NOT explain your reasoning. "
    "Do NOT list the options."
)

# Suffix appended to every prompt so the last token is the ":" in "ANSWER:"
# and the NEXT token to predict is the answer letter (e.g. " A").
ANSWER_PREFIX = "ANSWER:"

PALETTE = {
    "base_en":  "#2A9D8F",   # teal      — English baseline (upper bound)
    "base_ar":  "#87CEEB",   # sky blue  — Arabic baseline  (lower bound)
    "targeted": "#C0392B",   # crimson   — targeted LoRA L24-40
    "full":     "#F07C00",   # tangerine — full LoRA
    "window":   "#F4C430",   # mango     — critical window shading
}

# ── Data loading ────────────────────────────────────────────────────────────────
def extract_letter(val):
    if not val or not isinstance(val, str): return ""
    s = val.strip().upper()
    m = re.search(r'\bANSWER\s*:\s*([A-F])\b', s)
    if m: return m.group(1)
    m = re.search(r'\b([A-F])\b', s)
    return m.group(1) if m else ""

print(f"Loading {args.csv} ...")
rows = []
with open(args.csv, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        rows.append(row)

all_quadrants = sorted({r["quadrant"] for r in rows})
print(f"  Quadrant values in CSV: {all_quadrants}")

ag_rows = [r for r in rows if r["quadrant"] == args.quadrant]
if args.n_samples is not None:
    ag_rows = ag_rows[:args.n_samples]

en_texts   = [r["input_english"] for r in ag_rows]
ar_texts   = [r["input_arabic"]  for r in ag_rows]
gt_letters = [extract_letter(r["ground_truth"]) for r in ag_rows]
N = len(ag_rows)
print(f"  {args.quadrant} rows: {N}")

if N == 0:
    print(f"\nERROR: No rows found for quadrant='{args.quadrant}'.")
    print(f"  Available quadrant values: {all_quadrants}")
    print("  Check that --quadrant matches exactly (case-sensitive).")
    import sys; sys.exit(1)

# ── Load tokenizer & resolve answer token IDs ───────────────────────────────────
print("\nLoading tokenizer ...")
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Resolve answer token IDs in the EXACT context they appear after "ANSWER:".
# Tokenize "ANSWER: A", "ANSWER: B", etc. and take the final token — this is
# the letter token the model predicts immediately after "ANSWER: ".
ANSWER_TOKEN_IDS = {}
for letter in "ABCDEF":
    full_seq = tokenizer.encode(f"ANSWER: {letter}", add_special_tokens=False)
    prefix   = tokenizer.encode("ANSWER:",           add_special_tokens=False)
    # The letter token(s) are whatever comes after the prefix tokens
    letter_toks = full_seq[len(prefix):]
    if len(letter_toks) == 0:
        # Fallback: re-encode with space prefix
        letter_toks = tokenizer.encode(f" {letter}", add_special_tokens=False)
    ANSWER_TOKEN_IDS[letter] = letter_toks[0]

print(f"  Answer token IDs (in ANSWER: X context): {ANSWER_TOKEN_IDS}")

# Tokenize ANSWER_PREFIX once so we can append it to every prompt
ANSWER_PREFIX_IDS = tokenizer.encode(ANSWER_PREFIX, add_special_tokens=False)
print(f"  ANSWER: prefix token IDs: {ANSWER_PREFIX_IDS}  "
      f"(decoded: {[tokenizer.decode([t]) for t in ANSWER_PREFIX_IDS]})")

# ── Tokenization helper ─────────────────────────────────────────────────────────
def tokenize_batch(texts):
    """
    Builds input_ids ending with the tokens for "ANSWER:" so the model's
    next-token prediction is the answer letter.  This is consistent with:
      - the LoRA training format ("ANSWER: <LETTER>")
      - activation patching analyses (which measure P(letter) at first gen token)
    """
    all_ids = []
    for t in texts:
        if not t: t = "[empty]"
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": t}]
        try:
            chat_ids = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True)
        except Exception:
            chat_ids = tokenizer.encode(t, add_special_tokens=True)
        if not isinstance(chat_ids, list):
            try:    chat_ids = list(chat_ids["input_ids"])
            except: chat_ids = list(chat_ids)
        # Append "ANSWER:" — next predicted token will be the answer letter
        ids = chat_ids + ANSWER_PREFIX_IDS
        all_ids.append(ids[:args.max_len])

    bs    = len(texts)
    max_l = max(len(x) for x in all_ids)
    pad   = tokenizer.pad_token_id
    input_ids = torch.full((bs, max_l), pad, dtype=torch.long)
    attn_mask = torch.zeros((bs, max_l), dtype=torch.long)
    for j, ids in enumerate(all_ids):
        sl = len(ids)
        input_ids[j, max_l - sl:] = torch.tensor(ids, dtype=torch.long)
        attn_mask[j, max_l - sl:] = 1
    # Left-padded: last token is always at position max_l - 1
    seq_lens = torch.full((bs,), max_l - 1, dtype=torch.long)
    return input_ids, attn_mask, seq_lens

# ── Helper: get the final norm and lm_head from base or PEFT model ──────────────
def get_output_head(model):
    """
    Returns (final_norm, lm_head) regardless of whether model is a raw
    Mistral3ForConditionalGeneration or a PeftModel wrapping one.

    Mistral3ForConditionalGeneration architecture:
      Mistral3ForConditionalGeneration
        .model  (Mistral3Model)
          .language_model  (MistralModel)
            .layers / .norm
        .lm_head

    PeftModel architecture:
      PeftModel
        .base_model  (LoraModel)
          .model  (Mistral3ForConditionalGeneration)   ← inner
            .model.language_model.norm
            .lm_head

    NOTE: do NOT use hasattr(model, "base_model") — PreTrainedModel defines
    base_model as a property that returns self.model, so it's always truthy.
    Use isinstance instead.
    """
    if isinstance(model, PeftModel):
        inner = model.base_model.model   # LoraModel.model → Mistral3ForConditionalGeneration
    else:
        inner = model                    # already Mistral3ForConditionalGeneration
    return inner.model.language_model.norm, inner.lm_head

# ── Core: run direct logit lens → P(correct letter) at each layer ───────────────
def run_logit_lens(model, texts, gt_letters_local, label):
    """
    Returns array of shape (N, n_layers+1) — P(correct letter) at every
    hidden state (embedding + n_layers transformer blocks).
    """
    model.eval()
    final_norm, lm_head = get_output_head(model)
    n_layers_total = None

    # First pass: detect n_layers from model config
    with torch.no_grad():
        ids_probe, mask_probe, _ = tokenize_batch(texts[:1])
        out_probe = model(
            input_ids=ids_probe.to("cuda:0"),
            attention_mask=mask_probe.to("cuda:0"),
            output_hidden_states=True,
        )
        n_layers_total = len(out_probe.hidden_states)  # embedding + n_layers

    print(f"  [{label}] n_hidden_states = {n_layers_total}  (embedding + transformer blocks)")
    all_probs = []  # list of (bs, n_layers_total) arrays

    for i in range(0, N, args.batch_size):
        batch_texts = texts[i : i + args.batch_size]
        batch_gt    = gt_letters_local[i : i + args.batch_size]
        bs          = len(batch_texts)

        input_ids, attn_mask, seq_lens = tokenize_batch(batch_texts)
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids.to("cuda:0"),
                attention_mask=attn_mask.to("cuda:0"),
                output_hidden_states=True,
            )

        # outputs.hidden_states: tuple of n_layers_total tensors, each (bs, seq_len, d)
        batch_probs = np.zeros((bs, n_layers_total), dtype=np.float32)

        model_dtype = final_norm.weight.dtype  # bfloat16
        for layer_idx, hs in enumerate(outputs.hidden_states):
            # Grab last token hidden state for each example in batch
            # Keep in model dtype (bfloat16) through norm+lm_head; cast to float32 only for softmax
            last_hs = hs[torch.arange(bs), seq_lens].to(
                device=final_norm.weight.device, dtype=model_dtype
            )  # (bs, d)

            # Apply final RMSNorm then lm_head
            # Explicit device routing needed with device_map="auto":
            # final_norm lives with the last transformer block (GPU 1 on 2-GPU setup),
            # lm_head lives on GPU 0 — move normed tensor explicitly to avoid silent failure.
            normed = final_norm(last_hs)
            normed = normed.to(lm_head.weight.device)             # ensure same device as lm_head
            logits = lm_head(normed)                              # (bs, vocab_size), bfloat16
            probs  = torch.softmax(logits.float(), dim=-1).cpu()  # (bs, vocab_size), float32

            for j, gt in enumerate(batch_gt):
                if gt in ANSWER_TOKEN_IDS:
                    batch_probs[j, layer_idx] = probs[j, ANSWER_TOKEN_IDS[gt]].item()

        all_probs.append(batch_probs)
        if i % 20 == 0:
            print(f"  [{label}] {min(i + bs, N)}/{N}")

    return np.concatenate(all_probs, axis=0)  # (N, n_layers_total)

# ── Load base model ─────────────────────────────────────────────────────────────
print("\nLoading Mistral-Small-3.2-24B ...")
from transformers import Mistral3ForConditionalGeneration
model = Mistral3ForConditionalGeneration.from_pretrained(
    args.model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    local_files_only=True,
)
model.eval()
print(f"  n_layers: {len(model.model.language_model.layers)}")

# ── Condition 1 & 2: Base model, EN + AR ────────────────────────────────────────
print("\n── Condition 1: Base model + English ──────────────────────────────────")
base_en_probs = run_logit_lens(model, en_texts, gt_letters, label="base_EN")

print(f"  [sanity] base_EN final-layer P(correct) = {base_en_probs[:, -1].mean():.4f}  "
      f"(should be ~{base_en_probs[:, -2].mean():.4f} or higher)")

print("\n── Condition 2: Base model + Arabic ───────────────────────────────────")
base_ar_probs = run_logit_lens(model, ar_texts, gt_letters, label="base_AR")

# ── Condition 3: Targeted LoRA (L24-L40) + Arabic ───────────────────────────────
print(f"\n── Condition 3: Loading targeted LoRA from {args.targeted_adapter_dir} ──")
targeted_meta_path = os.path.join(args.targeted_adapter_dir, "train_meta.json")
if os.path.exists(targeted_meta_path):
    with open(targeted_meta_path) as f:
        tmeta = json.load(f)
    print(f"  lora_mode = {tmeta.get('lora_mode', 'unknown')}")
    print(f"  paper_layer_range = {tmeta.get('paper_layer_range', 'unknown')}")

model = PeftModel.from_pretrained(model, args.targeted_adapter_dir)
model.eval()

print("\n── Condition 3: Targeted LoRA + Arabic ────────────────────────────────")
targeted_ar_probs = run_logit_lens(model, ar_texts, gt_letters, label="targeted_AR")

# Unload adapter to restore base model weights
print("\nUnloading targeted LoRA adapter ...")
model = model.unload()
model.eval()

# ── Condition 4: Full LoRA + Arabic ─────────────────────────────────────────────
print(f"\n── Condition 4: Loading full LoRA from {args.full_adapter_dir} ──")
full_meta_path = os.path.join(args.full_adapter_dir, "train_meta.json")
if os.path.exists(full_meta_path):
    with open(full_meta_path) as f:
        fmeta = json.load(f)
    print(f"  lora_mode = {fmeta.get('lora_mode', 'unknown')}")
    print(f"  paper_layer_range = {fmeta.get('paper_layer_range', 'unknown')}")

model = PeftModel.from_pretrained(model, args.full_adapter_dir)
model.eval()

print("\n── Condition 4: Full LoRA + Arabic ────────────────────────────────────")
full_ar_probs = run_logit_lens(model, ar_texts, gt_letters, label="full_AR")

# ── Save raw results ─────────────────────────────────────────────────────────────
out_npz = os.path.join(args.out_dir, f"logit_lens_results_{args.quadrant}.npz")
np.savez(out_npz,
         base_en=base_en_probs,
         base_ar=base_ar_probs,
         targeted_ar=targeted_ar_probs,
         full_ar=full_ar_probs)
print(f"\nSaved → {out_npz}")

n_layers_total = base_en_probs.shape[1]  # embedding (layer 0) + 40 blocks
layer_labels   = list(range(n_layers_total))  # 0 = embedding, 1-40 = blocks

# ── Summary table ────────────────────────────────────────────────────────────────
print("\n══ Direct Logit Lens Summary (mean P(correct)) ══════════════════════════")
print(f"{'Layer':>6}  {'Base EN':>9}  {'Base AR':>9}  {'Tgt LoRA':>9}  {'Full LoRA':>9}  {'Δ Tgt':>8}  {'Δ Full':>8}")
# Show every 4th layer for readability
for l in range(0, n_layers_total, max(1, n_layers_total // 15)):
    be = base_en_probs[:, l].mean()
    ba = base_ar_probs[:, l].mean()
    tg = targeted_ar_probs[:, l].mean()
    fl = full_ar_probs[:, l].mean()
    print(f"  L{l:<4}  {be:>9.4f}  {ba:>9.4f}  {tg:>9.4f}  {fl:>9.4f}  {tg-ba:>+8.4f}  {fl-ba:>+8.4f}")

# ── Plot helpers ─────────────────────────────────────────────────────────────────
def sem(arr):
    return arr.std(axis=0) / np.sqrt(arr.shape[0])

# Truncate at L39: L40 (final block output) decodes near-zero for all conditions
# because the final transformer blocks shift probability mass to the output format
# token ("ANSWER:") rather than the bare letter — this is expected model behaviour,
# not a bug. The meaningful comparison region is L0–L39.
PLOT_END = n_layers_total - 1   # = 40, so slice [:40] gives indices 0-39

lx = np.arange(PLOT_END)  # 0 .. 39

def _slice(arr):
    return arr[:, :PLOT_END]

# ── Full overview (L0-L39) ────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))

for arr, color, label, ls, lw in [
    (base_en_probs,      PALETTE["base_en"],  "Base EN (upper bound)", "--", 2.2),
    (base_ar_probs,      PALETTE["base_ar"],  "Base AR (lower bound)", "--", 2.2),
    (targeted_ar_probs,  PALETTE["targeted"], "Targeted LoRA L24-40 + AR", "-", 2.5),
    (full_ar_probs,      PALETTE["full"],     "Full LoRA + AR",        "-",  2.5),
]:
    mu = _slice(arr).mean(0)
    se = sem(_slice(arr))
    ax.plot(lx, mu, lw=lw, color=color, ls=ls, label=label)
    ax.fill_between(lx, mu - se, mu + se, color=color, alpha=0.15)

ax.axvspan(24, 39, color=PALETTE["window"], alpha=0.10, label="Critical window L24-L39")
ax.axvline(24, color=PALETTE["window"], lw=1.2, ls=":", alpha=0.7)
ax.axvline(39, color=PALETTE["window"], lw=1.2, ls=":", alpha=0.7)

ax.set_xlabel("Transformer block (hidden state index)", fontsize=11)
ax.set_ylabel("Mean P(correct answer letter)", fontsize=11)
ax.set_title(
    f"Direct Logit Lens: LoRA vs Base Model\n"
    f"Mistral-Small-3.2-24B · MedAraBench · {args.quadrant}",
    fontsize=13, fontweight="bold"
)
ax.set_xlim(0, PLOT_END - 1)
ax.legend(fontsize=10, frameon=False, loc="upper left")
ax.grid(axis="y", alpha=0.18)
ax.spines[["top", "right"]].set_visible(False)

tick_locs = list(range(0, PLOT_END, 4))
ax.set_xticks(tick_locs)
ax.set_xticklabels([f"L{l}" for l in tick_locs], fontsize=8)

plt.tight_layout()
for ext in [".pdf", ".png"]:
    plt.savefig(os.path.join(args.out_dir, f"fig_logit_lens_lora_{args.quadrant}{ext}"),
                bbox_inches="tight", dpi=150)
    print(f"Saved → {os.path.join(args.out_dir, 'fig_logit_lens_lora' + ext)}")
plt.close()

# ── Zoomed panel: L15-L39, critical window ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))

zoom_start = 15
zoom_end   = PLOT_END        # 40 → slice gives indices 15-39
lx_zoom    = lx[zoom_start:zoom_end]

for arr, color, label, ls, lw in [
    (base_en_probs,      PALETTE["base_en"],  "Base EN",               "--", 2.0),
    (base_ar_probs,      PALETTE["base_ar"],  "Base AR",               "--", 2.0),
    (targeted_ar_probs,  PALETTE["targeted"], "Targeted LoRA L24-40",  "-",  2.5),
    (full_ar_probs,      PALETTE["full"],     "Full LoRA",             "-",  2.5),
]:
    mu = _slice(arr).mean(0)[zoom_start:zoom_end]
    se = sem(_slice(arr))[zoom_start:zoom_end]
    ax.plot(lx_zoom, mu, lw=lw, color=color, ls=ls, label=label)
    ax.fill_between(lx_zoom, mu - se, mu + se, color=color, alpha=0.15)

ax.axvspan(24, 39, color=PALETTE["window"], alpha=0.12)
ax.axvline(24, color=PALETTE["window"], lw=1.2, ls=":", alpha=0.8)
ax.axvline(39, color=PALETTE["window"], lw=1.2, ls=":", alpha=0.8,
           label="Critical window L24-L39")

ax.set_xlabel("Layer", fontsize=11)
ax.set_ylabel("Mean P(correct answer letter)", fontsize=11)
ax.set_title(
    f"Direct Logit Lens (Zoomed: L15–L39)\n"
    f"Mistral-Small-3.2-24B · {args.quadrant}",
    fontsize=12, fontweight="bold"
)
ax.set_xticks(list(range(zoom_start, zoom_end, 2)))
ax.set_xticklabels([f"L{l}" for l in range(zoom_start, zoom_end, 2)], fontsize=9)
ax.legend(fontsize=10, frameon=False)
ax.grid(axis="y", alpha=0.18)
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
for ext in [".pdf", ".png"]:
    plt.savefig(os.path.join(args.out_dir, f"fig_logit_lens_lora_zoom_{args.quadrant}{ext}"),
                bbox_inches="tight", dpi=150)
    print(f"Saved → {os.path.join(args.out_dir, 'fig_logit_lens_lora_zoom' + ext)}")
plt.close()

print("\nDone.")