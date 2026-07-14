"""
lora_weight_analysis.py
-----------------------
Tests whether the LoRA update directions at L24-40 align with the
English-Arabic representational gap at those layers.

For each layer l in the targeted range:
  1. delta_W_l = (alpha/r) * B_l @ A_l  for o_proj + down_proj
     (modules that write directly to the residual stream)
  2. Top left singular vector of delta_W_l  →  principal output direction
  3. English-Arabic hidden state gap at l   →  mean(h_EN) - mean(h_AR)
  4. Cosine similarity between (2) and (3)

High alignment  →  LoRA learned to bridge the EN-AR gap mechanistically
Low alignment   →  improvement is data-driven, not representational

Also reports per-layer Frobenius norm of delta_W (update magnitude).

Usage:
  python lora_weight_analysis.py \
      --csv mistral_sampled_quadrants.csv \
      --model_path /path/to/mistral \
      --targeted_adapter_dir outputs/lr_search_10ep/trial_03_lr2.80e-05 \
      --out_dir ./lora_weight_analysis_out \
      --n_samples 100
"""

import os
import csv
import re
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from peft import PeftModel
from transformers import AutoTokenizer, Mistral3ForConditionalGeneration

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--csv",                  required=True)
parser.add_argument("--model_path",           required=True)
parser.add_argument("--targeted_adapter_dir", required=True)
parser.add_argument("--out_dir",   default="./lora_weight_analysis_out")
parser.add_argument("--max_len",   type=int, default=512)
parser.add_argument("--n_samples", type=int, default=100,
                    help="Questions per condition for hidden state collection")
parser.add_argument("--quadrant",  type=str, default="access_gap")
args = parser.parse_args()
os.makedirs(args.out_dir, exist_ok=True)

TARGETED_LAYERS = list(range(24, 40))   # paper L24-L39 (0-indexed transformer blocks)
LORA_ALPHA = 32
LORA_R     = 16
SCALE      = LORA_ALPHA / LORA_R        # = 2.0

PALETTE = {
    "magnitude":  "#F07C00",
    "alignment":  "#C0392B",
    "random":     "#AAAAAA",
    "window":     "#F4C430",
}

SYSTEM_PROMPT = (
    "You are a medical expert answering multiple-choice exam questions. "
    "You will receive exactly ONE question followed by answer options labeled: "
    "A), B), C), D), E), and sometimes F). "
    "You must output exactly ONE line in this format: ANSWER: <LETTER> "
    "Rules: Output ONLY that line. Do NOT repeat or paraphrase the question. "
    "Do NOT translate anything. Do NOT explain your reasoning. "
    "Do NOT list the options."
)
ANSWER_PREFIX = "ANSWER:"

# ── Load CSV ──────────────────────────────────────────────────────────────────
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

ag_rows = [r for r in rows if r["quadrant"] == args.quadrant][:args.n_samples]
en_texts   = [r["input_english"] for r in ag_rows]
ar_texts   = [r["input_arabic"]  for r in ag_rows]
N = len(ag_rows)
print(f"  {args.quadrant}: {N} samples")

# ── Tokenizer ─────────────────────────────────────────────────────────────────
print("\nLoading tokenizer ...")
tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

ANSWER_PREFIX_IDS = tokenizer.encode(ANSWER_PREFIX, add_special_tokens=False)

def tokenize_batch(texts):
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
    seq_lens = torch.full((bs,), max_l - 1, dtype=torch.long)
    return input_ids, attn_mask, seq_lens

# ── Load base model ───────────────────────────────────────────────────────────
print("\nLoading Mistral-Small-3.2-24B ...")
model = Mistral3ForConditionalGeneration.from_pretrained(
    args.model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    local_files_only=True,
)
model.eval()

# ─────────────────────────────────────────────────────────────────────────────
#  PART 1: Collect English and Arabic hidden states (base model)
# ─────────────────────────────────────────────────────────────────────────────
def collect_hidden_states(texts, label):
    """Returns dict: layer_idx → (N, d_model) numpy array."""
    hs_store = {l: [] for l in TARGETED_LAYERS}
    for i in range(0, len(texts), 1):
        batch = texts[i:i+1]
        input_ids, attn_mask, seq_lens = tokenize_batch(batch)
        with torch.no_grad():
            out = model(
                input_ids=input_ids.to("cuda:0"),
                attention_mask=attn_mask.to("cuda:0"),
                output_hidden_states=True,
            )
        # hidden_states[0] = embedding, hidden_states[l+1] = output of block l
        for l in TARGETED_LAYERS:
            hs = out.hidden_states[l + 1]   # +1: block l output
            last = hs[0, seq_lens[0]].float().cpu().numpy()
            hs_store[l].append(last)
        if i % 20 == 0:
            print(f"  [{label}] {i+1}/{len(texts)}")
    return {l: np.stack(hs_store[l]) for l in TARGETED_LAYERS}

print("\nCollecting English hidden states (base model) ...")
en_hs = collect_hidden_states(en_texts, "EN")

print("\nCollecting Arabic hidden states (base model) ...")
ar_hs = collect_hidden_states(ar_texts, "AR")

# English-Arabic gap at each targeted layer
gap_vectors = {}
for l in TARGETED_LAYERS:
    gap = en_hs[l].mean(0) - ar_hs[l].mean(0)          # (d_model,)
    norm = np.linalg.norm(gap)
    gap_vectors[l] = gap / norm if norm > 1e-8 else gap  # unit vector

print("\nGap vector norms (before normalization):")
for l in TARGETED_LAYERS:
    raw = en_hs[l].mean(0) - ar_hs[l].mean(0)
    print(f"  L{l}: {np.linalg.norm(raw):.4f}")

# ─────────────────────────────────────────────────────────────────────────────
#  PART 2: Extract LoRA weight directions
# ─────────────────────────────────────────────────────────────────────────────
print(f"\nLoading targeted LoRA from {args.targeted_adapter_dir} ...")
model_peft = PeftModel.from_pretrained(model, args.targeted_adapter_dir)
model_peft.eval()

# Modules that write to the residual stream: o_proj (attention) + down_proj (MLP)
# These are the most direct paths from LoRA updates to the hidden state trajectory.
TARGET_MODULES = ["o_proj", "down_proj"]

layer_magnitudes  = {}   # l → mean Frobenius norm across modules
layer_alignments  = {}   # l → mean cosine similarity with gap vector

print("\nExtracting LoRA update directions ...")

for l in TARGETED_LAYERS:
    layer_idx = l   # 0-indexed block

    magnitudes = []
    alignments = []

    for module_name in TARGET_MODULES:
        # Navigate to the correct module
        try:
            if module_name == "o_proj":
                lora_module = (model_peft.base_model.model
                               .model.language_model.layers[layer_idx]
                               .self_attn.o_proj)
            elif module_name == "down_proj":
                lora_module = (model_peft.base_model.model
                               .model.language_model.layers[layer_idx]
                               .mlp.down_proj)
        except (AttributeError, IndexError) as e:
            print(f"  [WARN] L{l} {module_name}: {e}")
            continue

        # Extract lora_A and lora_B — PEFT stores them under .lora_A / .lora_B
        # which are ModuleDicts keyed by adapter name (default "default")
        try:
            A = lora_module.lora_A["default"].weight.float().cpu()  # (r, d_in)
            B = lora_module.lora_B["default"].weight.float().cpu()  # (d_out, r)
        except (AttributeError, KeyError):
            # Older PEFT versions store directly
            try:
                A = lora_module.lora_A.weight.float().cpu()
                B = lora_module.lora_B.weight.float().cpu()
            except AttributeError:
                print(f"  [WARN] L{l} {module_name}: could not extract lora_A/lora_B")
                continue

        # delta_W = scale * B @ A,  shape (d_out, d_in)
        with torch.no_grad():
            delta_W = (SCALE * B @ A).numpy()    # (d_out, d_in)

        # Frobenius norm = overall update magnitude
        frob = np.linalg.norm(delta_W, "fro")
        magnitudes.append(frob)

        # SVD — top LEFT singular vector = principal output direction (shape d_out)
        # For o_proj:   d_out = d_model  → lives in residual stream space ✓
        # For down_proj: d_out = d_model ✓
        U, S, Vt = np.linalg.svd(delta_W, full_matrices=False)
        top_u = U[:, 0]   # (d_model,) — principal direction written to residual

        # Cosine similarity with English-Arabic gap
        gap = gap_vectors[l]                     # (d_model,) unit vector
        d_out = top_u.shape[0]
        d_gap = gap.shape[0]

        if d_out != d_gap:
            # down_proj d_out should equal d_model; o_proj too — but sanity check
            print(f"  [WARN] L{l} {module_name}: d_out={d_out} != d_gap={d_gap}, skipping alignment")
            continue

        top_u_norm = top_u / (np.linalg.norm(top_u) + 1e-8)
        cos_sim = float(np.dot(top_u_norm, gap))
        alignments.append(abs(cos_sim))    # absolute value: direction or anti-direction both count

        print(f"  L{l:2d} {module_name:<12} frob={frob:.4f}  cos_sim={cos_sim:+.4f}  |cos|={abs(cos_sim):.4f}")

    if magnitudes:
        layer_magnitudes[l] = np.mean(magnitudes)
    if alignments:
        layer_alignments[l] = np.mean(alignments)

# ─────────────────────────────────────────────────────────────────────────────
#  PART 3: Random baseline for alignment
# ─────────────────────────────────────────────────────────────────────────────
# Expected |cos similarity| between a random unit vector and a fixed unit
# vector in d-dimensional space ≈ sqrt(2/(pi*d))
# For d=5120:  ≈ 0.011  (near zero — any alignment >> this is meaningful)
d_model = list(en_hs.values())[0].shape[1]
random_baseline = np.sqrt(2 / (np.pi * d_model))
print(f"\nRandom alignment baseline (d={d_model}): {random_baseline:.5f}")

# ─────────────────────────────────────────────────────────────────────────────
#  PART 4: Save results
# ─────────────────────────────────────────────────────────────────────────────
layers_arr     = np.array(TARGETED_LAYERS)
magnitudes_arr = np.array([layer_magnitudes.get(l, np.nan) for l in TARGETED_LAYERS])
alignments_arr = np.array([layer_alignments.get(l, np.nan) for l in TARGETED_LAYERS])

np.savez(
    os.path.join(args.out_dir, "lora_alignment_results.npz"),
    layers=layers_arr,
    magnitudes=magnitudes_arr,
    alignments=alignments_arr,
    random_baseline=random_baseline,
)
print(f"\nSaved → {os.path.join(args.out_dir, 'lora_alignment_results.npz')}")

# Console summary
print("\n══ Results ══════════════════════════════════════════════")
print(f"{'Layer':>6}  {'||ΔW|| (Frob)':>14}  {'|cos(ΔW, gap)|':>16}")
print("-" * 42)
for l, mag, aln in zip(layers_arr, magnitudes_arr, alignments_arr):
    print(f"  L{l:<4}  {mag:>14.4f}  {aln:>16.4f}")
print(f"\nRandom baseline alignment: {random_baseline:.5f}")
print(f"Mean alignment L24-40:     {np.nanmean(alignments_arr):.4f}")
print(f"Max alignment:             {np.nanmax(alignments_arr):.4f}  "
      f"at L{layers_arr[np.nanargmax(alignments_arr)]}")

# ─────────────────────────────────────────────────────────────────────────────
#  PART 5: Plot
# ─────────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 5))
gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

# Panel A: Update magnitude
ax1 = fig.add_subplot(gs[0, 0])
ax1.bar(layers_arr, magnitudes_arr, color=PALETTE["magnitude"],
        alpha=0.85, edgecolor="white", linewidth=0.4)
ax1.axvspan(23.5, 39.5, color=PALETTE["window"], alpha=0.12, label="Targeted window L24-39")
ax1.set_xlabel("Layer", fontsize=11)
ax1.set_ylabel("Frobenius Norm of $\\Delta W$", fontsize=11)
ax1.set_title("LoRA Update Magnitude per Layer\n(o\\_proj + down\\_proj)",
              fontsize=11, fontweight="bold")
ax1.set_xticks(layers_arr[::2])
ax1.set_xticklabels([f"L{l}" for l in layers_arr[::2]], fontsize=8)
ax1.grid(axis="y", alpha=0.18)
ax1.spines[["top", "right"]].set_visible(False)

# Panel B: Alignment with EN-AR gap
ax2 = fig.add_subplot(gs[0, 1])
ax2.bar(layers_arr, alignments_arr, color=PALETTE["alignment"],
        alpha=0.85, edgecolor="white", linewidth=0.4, label="Observed alignment")
ax2.axhline(random_baseline, color=PALETTE["random"], lw=1.5, ls="--",
            label=f"Random baseline ({random_baseline:.4f})")
ax2.axvspan(23.5, 39.5, color=PALETTE["window"], alpha=0.12, label="Targeted window L24-39")
ax2.set_xlabel("Layer", fontsize=11)
ax2.set_ylabel("|cos(top $\\Delta W$ direction, EN–AR gap)|", fontsize=10)
ax2.set_title("Alignment: LoRA Update vs EN–AR Gap\n(o\\_proj + down\\_proj)",
              fontsize=11, fontweight="bold")
ax2.set_xticks(layers_arr[::2])
ax2.set_xticklabels([f"L{l}" for l in layers_arr[::2]], fontsize=8)
ax2.legend(fontsize=9, frameon=False)
ax2.grid(axis="y", alpha=0.18)
ax2.spines[["top", "right"]].set_visible(False)

fig.suptitle(
    "LoRA Weight Direction Analysis — Targeted L24-40\n"
    "Mistral-Small-3.2-24B · MedAraBench · access\\_gap",
    fontsize=12, fontweight="bold", y=1.02
)

for ext in [".pdf", ".png"]:
    out = os.path.join(args.out_dir, f"fig_lora_weight_analysis{ext}")
    plt.savefig(out, bbox_inches="tight", dpi=150)
    print(f"Saved → {out}")
plt.close()
print("\nDone.")