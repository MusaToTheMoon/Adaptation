"""
activation_patching_allam.py
-----------------------------
Causal activation patching for ALLaM 7B Instruct on MedAraBench.
Adapted from activation_patching_llama.py.

Architecture (LlamaForCausalLM, 32 layers):
  model.model.layers[i]  — 32 × LlamaDecoderLayer (0-indexed)
  model.model.norm       — final RMSNorm
  model.lm_head          — Linear(64000 × 4096)

Answer token IDs (ALLaM 64K vocab, confirmed by diagnose.py):
  A=395, B=482, C=415, D=526, E=578, F=521

Probe window from tuned lens: ALLaM shows a flat profile with late collapse
at L28-32. Probe dense around the final layers.

Usage:
  python activation_patching_allam.py \\
      --csv        allam_sampled_quadrants.csv \\
      --model_path /scratch/ca2627/huggingface/models--humain-ai--ALLaM-7B-Instruct-preview/snapshots/<hash> \\
      --out_dir    ./activation_patching_allam_out
"""

import os, csv, re, argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--csv",        required=True)
parser.add_argument("--model_path", required=True)
parser.add_argument("--out_dir",    default="./activation_patching_allam_out")
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--max_len",    type=int, default=512)
args = parser.parse_args()
os.makedirs(args.out_dir, exist_ok=True)

# Probe layers — hidden_states[L] = output of transformer layer L-1
# ALLaM has 32 layers; probe dense in the late zone where collapse occurs
PROBE_LAYERS = [16, 20, 24, 26, 28, 29, 30, 31, 32]

PATCH_CONFIGS = {
    "patch_L16":    [16],
    "patch_L20":    [20],
    "patch_L24":    [24],
    "patch_L26":    [26],
    "patch_L28":    [28],
    "patch_L29":    [29],
    "patch_L30":    [30],
    "patch_L31":    [31],
    "patch_L28_32": [28, 29, 30, 31, 32],
    "patch_L24_32": [24, 26, 28, 29, 30, 31, 32],
}

# Confirmed by diagnose.py on ALLaM 7B (64K vocab)
ANSWER_TOKEN_IDS = {
    "A": 395, "B": 482, "C": 415,
    "D": 526, "E": 578, "F": 521,
}

SYSTEM_PROMPT = (
    "You are a medical expert. "
    "Answer the following multiple choice question "
    "by responding with only the letter of the correct option: A, B, C, or D. "
    "Do not explain your answer."
)

def extract_letter(val):
    if not val or not isinstance(val, str): return ""
    s = val.strip().upper()
    m = re.search(r'\bANSWER\s*:\s*([A-F])\b', s)
    if m: return m.group(1)
    m = re.search(r'\b([A-F])\b', s)
    return m.group(1) if m else ""

# ── Load CSV — keep only access_gap ───────────────────────────────────────────
print(f"Loading {args.csv}...")
rows = []
with open(args.csv, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        rows.append(row)

ag_rows    = [r for r in rows if r["quadrant"] == "access_gap"]
en_texts   = [r["input_english"] for r in ag_rows]
ar_texts   = [r["input_arabic"]  for r in ag_rows]
gt_letters = [extract_letter(r["ground_truth"]) for r in ag_rows]
N = len(ag_rows)
print(f"  Total rows: {len(rows)} | access_gap: {N}")

# ── Load model ────────────────────────────────────────────────────────────────
print("\nLoading ALLaM 7B...")
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained(
    args.model_path, local_files_only=True, trust_remote_code=True)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    args.model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    local_files_only=True,
    trust_remote_code=True,
)
model.eval()
print(f"  Model type: {type(model).__name__}")

transformer_layers = model.model.layers
n_layers = len(transformer_layers)
print(f"  n_layers: {n_layers}")
assert n_layers == 32, f"Expected 32 layers for ALLaM 7B, got {n_layers}"

for L in PROBE_LAYERS:
    assert 1 <= L <= n_layers, f"Probe layer {L} out of range [1, {n_layers}]"

# ── Verify answer tokens ──────────────────────────────────────────────────────
print("  Verifying answer token IDs ...")
for letter, tid in ANSWER_TOKEN_IDS.items():
    decoded = tokenizer.decode([tid]).strip()
    status = "✓" if decoded.upper() == letter else f"[WARN] decodes as {repr(decoded)}"
    print(f"    {letter} → {tid}  {status}")

# ── Tokenise helper ───────────────────────────────────────────────────────────
def tokenize_batch(texts):
    all_ids = []
    for t in texts:
        if not t: t = "[empty]"
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": t},
        ]
        try:
            ids = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_tensors=None)
            if not isinstance(ids, list):
                try:    ids = ids["input_ids"]
                except: ids = ids.ids
        except Exception:
            ids = tokenizer.encode(t, add_special_tokens=True)
        all_ids.append(list(ids)[:args.max_len])

    bs    = len(texts)
    max_l = max(len(x) for x in all_ids)
    pad   = tokenizer.pad_token_id
    input_ids = torch.full((bs, max_l), pad, dtype=torch.long)
    attn_mask = torch.zeros((bs, max_l), dtype=torch.long)
    for j, ids in enumerate(all_ids):
        sl = len(ids)
        input_ids[j, max_l - sl:] = torch.tensor(ids)
        attn_mask[j, max_l - sl:] = 1
    # With left-padding, every sequence's last real token is always at
    # position max_l - 1 (sequences are right-aligned). DO NOT use
    # attn_mask.sum(dim=1) - 1 — that gives sequence length - 1, not
    # the last token's position, and breaks for any bs > 1.
    seq_lens = torch.full((bs,), max_l - 1, dtype=torch.long)
    return input_ids, attn_mask, seq_lens

# ── P(correct) from output logits ────────────────────────────────────────────
def correct_probs(logits, seq_lens, gt_batch):
    bs = logits.shape[0]
    probs_all = torch.softmax(logits.float(), dim=-1)
    result = np.zeros(bs, dtype=np.float32)
    for j, gt in enumerate(gt_batch):
        if gt not in ANSWER_TOKEN_IDS: continue
        result[j] = probs_all[j, seq_lens[j], ANSWER_TOKEN_IDS[gt]].item()
    return result

# ── Phase 1: English forward pass — cache hidden states ──────────────────────
print(f"\nPhase 1: English forward pass — caching layers {PROBE_LAYERS} ...")
en_hs        = {L: [] for L in PROBE_LAYERS}
en_base_prob = []

for i in range(0, N, args.batch_size):
    batch_en = en_texts[i : i + args.batch_size]
    batch_gt = gt_letters[i : i + args.batch_size]
    bs       = len(batch_en)
    input_ids, attn_mask, seq_lens = tokenize_batch(batch_en)
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids.to("cuda:0"),
            attention_mask=attn_mask.to("cuda:0"),
            output_hidden_states=True,
        )
    for L in PROBE_LAYERS:
        hs   = outputs.hidden_states[L]
        last = hs[torch.arange(bs), seq_lens].float().cpu()
        en_hs[L].append(last.numpy())
    probs = correct_probs(outputs.logits.cpu(), seq_lens, batch_gt)
    en_base_prob.extend(probs.tolist())
    if i % 20 == 0:
        print(f"  [EN] {min(i+bs, N)}/{N}")

for L in PROBE_LAYERS:
    en_hs[L] = np.concatenate(en_hs[L], axis=0)
en_base_prob = np.array(en_base_prob)
print(f"  English baseline mean P(correct): {en_base_prob.mean():.3f}")

# ── Phase 2: Arabic baseline ──────────────────────────────────────────────────
print(f"\nPhase 2: Arabic baseline (no patching) ...")
ar_base_prob = []

for i in range(0, N, args.batch_size):
    batch_ar = ar_texts[i : i + args.batch_size]
    batch_gt = gt_letters[i : i + args.batch_size]
    bs       = len(batch_ar)
    input_ids, attn_mask, seq_lens = tokenize_batch(batch_ar)
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids.to("cuda:0"),
            attention_mask=attn_mask.to("cuda:0"),
        )
    probs = correct_probs(outputs.logits.cpu(), seq_lens, batch_gt)
    ar_base_prob.extend(probs.tolist())
    if i % 20 == 0:
        print(f"  [AR base] {min(i+bs, N)}/{N}")

ar_base_prob = np.array(ar_base_prob)
print(f"  Arabic baseline mean P(correct): {ar_base_prob.mean():.3f}")

# ── Phase 3: Patching ─────────────────────────────────────────────────────────
patch_results = {}

for config_name, patch_layers in PATCH_CONFIGS.items():
    print(f"\nPhase 3 [{config_name}]: patching at layers {patch_layers} ...")
    config_probs = []

    for i in range(0, N, args.batch_size):
        batch_ar = ar_texts[i : i + args.batch_size]
        batch_gt = gt_letters[i : i + args.batch_size]
        bs       = len(batch_ar)
        input_ids, attn_mask, seq_lens = tokenize_batch(batch_ar)

        batch_en_hs = {L: torch.tensor(en_hs[L][i : i + bs]) for L in patch_layers}

        handles = []
        for L in patch_layers:
            en_h  = batch_en_hs[L]
            ar_sl = seq_lens

            def make_hook(en_h_=en_h, ar_sl_=ar_sl, bs_=bs):
                def hook_fn(module, inp, output):
                    hs_out = output[0].clone() if isinstance(output, tuple) else output.clone()
                    for j in range(bs_):
                        hs_out[j, ar_sl_[j]] = en_h_[j].to(hs_out.device).to(hs_out.dtype)
                    return (hs_out,) + output[1:] if isinstance(output, tuple) else hs_out
                return hook_fn

            handle = transformer_layers[L - 1].register_forward_hook(make_hook())
            handles.append(handle)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids.to("cuda:0"),
                attention_mask=attn_mask.to("cuda:0"),
            )
        for h in handles:
            h.remove()

        probs = correct_probs(outputs.logits.cpu(), seq_lens, batch_gt)
        config_probs.extend(probs.tolist())
        if i % 20 == 0:
            print(f"  [{config_name}] {min(i+bs, N)}/{N}")

    patch_results[config_name] = np.array(config_probs)
    print(f"  Mean P(correct): {patch_results[config_name].mean():.3f}")

# ── Save ──────────────────────────────────────────────────────────────────────
out_npz = os.path.join(args.out_dir, "patching_results.npz")
save_dict = dict(en_base_prob=en_base_prob, ar_base_prob=ar_base_prob,
                 gt_letters=np.array(gt_letters))
save_dict.update(patch_results)
np.savez(out_npz, **save_dict)
print(f"\nSaved → {out_npz}")

# ── Summary ───────────────────────────────────────────────────────────────────
ar_mean = ar_base_prob.mean()
en_mean = en_base_prob.mean()
gap     = en_mean - ar_mean

def recovery(val):
    return 100 * (val - ar_mean) / gap if gap > 0 else 0.0

print("\n══ Activation Patching Results ══════════════════════════════")
print(f"  {'Config':<22} {'Mean P':>10}  {'Recovery':>10}")
print(f"  {'ar_base':<22} {ar_mean:>10.3f}  {'(0%)':>10}")
for c in PATCH_CONFIGS:
    m = patch_results[c].mean()
    print(f"  {c:<22} {m:>10.3f}  {recovery(m):>9.1f}%")
print(f"  {'en_base':<22} {en_mean:>10.3f}  {'(100%)':>10}")

# ── Figure: Bar chart ─────────────────────────────────────────────────────────
all_configs = ["ar_base"] + list(PATCH_CONFIGS.keys()) + ["en_base"]
all_means   = [ar_base_prob.mean()] + [patch_results[c].mean() for c in PATCH_CONFIGS] + [en_base_prob.mean()]
all_sems    = [ar_base_prob.std() / np.sqrt(N)] + [patch_results[c].std() / np.sqrt(N) for c in PATCH_CONFIGS] + [en_base_prob.std() / np.sqrt(N)]

n_patches = len(PATCH_CONFIGS)
cmap   = plt.cm.RdYlGn(np.linspace(0.15, 0.85, n_patches))
colors = ["#87CEEB"] + list(cmap) + ["#2A9D8F"]

x_labels = (["Arabic\n(base)"] +
             [c.replace("patch_", "") for c in PATCH_CONFIGS] +
             ["English\n(base)"])

fig, ax = plt.subplots(figsize=(14, 6))
bars = ax.bar(range(len(all_configs)), all_means, color=colors,
              yerr=all_sems, capsize=4, alpha=0.88,
              edgecolor="white", linewidth=0.5)
for bar, m in zip(bars, all_means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{m:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
for ci, c in enumerate(PATCH_CONFIGS.keys()):
    ax.text(ci + 1, -0.018, f"{recovery(patch_results[c].mean()):.0f}%",
            ha="center", va="top", fontsize=8, color="#555555")
ax.text(-0.5, -0.018, "Recovery:", ha="left", va="top", fontsize=8, color="#555555")
ax.axhline(ar_mean, color="#87CEEB", linewidth=1.5, linestyle="--", alpha=0.8,
           label=f"Arabic baseline ({ar_mean:.3f})")
ax.axhline(en_mean, color="#2A9D8F", linewidth=1.5, linestyle="--", alpha=0.8,
           label=f"English baseline ({en_mean:.3f})")
ax.set_xticks(range(len(all_configs)))
ax.set_xticklabels(x_labels, fontsize=9)
ax.set_ylabel("Mean P(correct answer letter) at model output", fontsize=11)
ax.set_title(
    "Activation Patching: English → Arabic Hidden State Injection\n"
    "ALLaM 7B  ·  MedAraBench  ·  access_gap (En✓ Ar✗)",
    fontsize=12, fontweight="bold")
ax.legend(fontsize=10, frameon=False, loc="upper left")
ax.set_ylim(-0.03, max(all_means) * 1.25)
ax.grid(axis="y", alpha=0.2)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
for ext in [".pdf", ".png"]:
    out_fig = os.path.join(args.out_dir, f"fig_patching_allam{ext}")
    plt.savefig(out_fig, bbox_inches="tight", dpi=150)
    print(f"Saved → {out_fig}")
plt.close()
print("\nDone.")