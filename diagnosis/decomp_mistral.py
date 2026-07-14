"""
attn_mlp_decomp_mistral.py
---------------------------
Attention vs MLP decomposition of activation patching for Mistral-Small-3.2-24B.

For each critical layer (where full patching gives ≥40% recovery), we run:
  - attn-only patch:  inject English self_attn output into Arabic forward pass
  - mlp-only patch:   inject English mlp output into Arabic forward pass
and compare both to the full-layer patch recovery.

This isolates which sublayer drives the knowledge recovery at each layer.

Usage:
  python attn_mlp_decomp_mistral.py \\
      --csv        mistral_sampled_quadrants.csv \\
      --model_path /scratch/ca2627/huggingface/models--mistralai--Mistral-Small-3.2-24B-Instruct-2506/snapshots/<hash> \\
      --out_dir    ./attn_mlp_decomp_mistral_out
"""

import os, csv, re, argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

parser = argparse.ArgumentParser()
parser.add_argument("--csv",        required=True)
parser.add_argument("--model_path", required=True)
parser.add_argument("--out_dir",    default="./attn_mlp_decomp_mistral_out")
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--max_len",    type=int, default=512)
args = parser.parse_args()
os.makedirs(args.out_dir, exist_ok=True)

# Critical layers only — where full patching gives ≥40% recovery for Mistral
CRITICAL_LAYERS = [24, 28, 32, 34, 36, 38, 40]

ANSWER_TOKEN_IDS = {
    "A": 1065, "B": 1066, "C": 1067,
    "D": 1068, "E": 1069, "F": 1070,
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

# ── Load CSV ──────────────────────────────────────────────────────────────────
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
print(f"  access_gap: {N}")

# ── Load model ────────────────────────────────────────────────────────────────
print("\nLoading Mistral-Small-3.2-24B...")
from transformers import AutoTokenizer, Mistral3ForConditionalGeneration

tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = Mistral3ForConditionalGeneration.from_pretrained(
    args.model_path, dtype=torch.bfloat16,
    device_map="auto", local_files_only=True)
model.eval()

def get_transformer_layers(m):
    for fn in [
        lambda m: m.model.language_model.model.layers,
        lambda m: m.model.language_model.layers,
        lambda m: m.model.layers,
    ]:
        try:
            layers = fn(m)
            if layers and len(layers) > 0:
                print(f"  Layers found (n={len(layers)})")
                return layers
        except AttributeError:
            continue
    raise RuntimeError("Cannot find transformer layers.")

transformer_layers = get_transformer_layers(model)
n_layers = len(transformer_layers)
print(f"  n_layers: {n_layers}")

# ── Tokenise ──────────────────────────────────────────────────────────────────
def tokenize_batch(texts):
    all_ids = []
    for t in texts:
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": t}]
        chat_str = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        encoded = tokenizer(chat_str)
        ids = encoded["input_ids"]
        if hasattr(ids, "ids"): ids = ids.ids
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
    seq_lens = torch.full((bs,), max_l - 1, dtype=torch.long)
    return input_ids, attn_mask, seq_lens

def correct_probs(logits, seq_lens, gt_batch):
    bs = logits.shape[0]
    probs_all = torch.softmax(logits.float(), dim=-1)
    result = np.zeros(bs, dtype=np.float32)
    for j, gt in enumerate(gt_batch):
        if gt not in ANSWER_TOKEN_IDS: continue
        result[j] = probs_all[j, seq_lens[j], ANSWER_TOKEN_IDS[gt]].item()
    return result

# ── Phase 1: English — cache attn output, mlp output, full hidden state ───────
print(f"\nPhase 1: English forward pass — caching attn/mlp at layers {CRITICAL_LAYERS} ...")

en_attn_out  = {L: [] for L in CRITICAL_LAYERS}   # raw self_attn output
en_mlp_out   = {L: [] for L in CRITICAL_LAYERS}   # raw mlp output
en_full_hs   = {L: [] for L in CRITICAL_LAYERS}   # full hidden state (for full-patch baseline)
en_base_prob = []

# Temporary storage — populated by hooks during each batch
_tmp_attn = {}
_tmp_mlp  = {}

# Register persistent caching hooks for English pass
cache_handles = []
for L in CRITICAL_LAYERS:
    def make_attn_cache(L_=L):
        def fn(module, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            _tmp_attn[L_] = o.detach().float().cpu()
        return fn
    def make_mlp_cache(L_=L):
        def fn(module, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            _tmp_mlp[L_] = o.detach().float().cpu()
        return fn
    cache_handles.append(
        transformer_layers[L - 1].self_attn.register_forward_hook(make_attn_cache()))
    cache_handles.append(
        transformer_layers[L - 1].mlp.register_forward_hook(make_mlp_cache()))

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
    # Extract last-position vectors from hook captures
    for L in CRITICAL_LAYERS:
        en_attn_out[L].append(_tmp_attn[L][torch.arange(bs), seq_lens].numpy())
        en_mlp_out[L].append( _tmp_mlp[L][torch.arange(bs), seq_lens].numpy())
        hs_full = outputs.hidden_states[L]
        en_full_hs[L].append(hs_full[torch.arange(bs), seq_lens].float().cpu().numpy())
    probs = correct_probs(outputs.logits.cpu(), seq_lens, batch_gt)
    en_base_prob.extend(probs.tolist())
    if i % 20 == 0:
        print(f"  [EN] {min(i+bs, N)}/{N}")

for h in cache_handles:
    h.remove()

for L in CRITICAL_LAYERS:
    en_attn_out[L] = np.concatenate(en_attn_out[L], axis=0)
    en_mlp_out[L]  = np.concatenate(en_mlp_out[L],  axis=0)
    en_full_hs[L]  = np.concatenate(en_full_hs[L],  axis=0)

en_base_prob = np.array(en_base_prob)
print(f"  English baseline mean P(correct): {en_base_prob.mean():.3f}")

# ── Phase 2: Arabic baseline ──────────────────────────────────────────────────
print("\nPhase 2: Arabic baseline ...")
ar_base_prob = []
for i in range(0, N, args.batch_size):
    batch_ar = ar_texts[i : i + args.batch_size]
    batch_gt = gt_letters[i : i + args.batch_size]
    bs       = len(batch_ar)
    input_ids, attn_mask, seq_lens = tokenize_batch(batch_ar)
    with torch.no_grad():
        outputs = model(input_ids=input_ids.to("cuda:0"),
                        attention_mask=attn_mask.to("cuda:0"))
    ar_base_prob.extend(correct_probs(outputs.logits.cpu(), seq_lens, batch_gt).tolist())
    if i % 20 == 0:
        print(f"  [AR base] {min(i+bs, N)}/{N}")
ar_base_prob = np.array(ar_base_prob)
print(f"  Arabic baseline mean P(correct): {ar_base_prob.mean():.3f}")

# ── Phase 3: Decomposition patching ──────────────────────────────────────────
# For each critical layer run: attn-only, mlp-only, full-layer patch
def run_patch(patch_type, L, cached_en):
    """
    patch_type: 'attn' | 'mlp' | 'full'
    cached_en:  np.array (N, d) — English cached activation for this layer/type
    Returns:    np.array (N,) of P(correct)
    """
    config_probs = []
    for i in range(0, N, args.batch_size):
        batch_ar = ar_texts[i : i + args.batch_size]
        batch_gt = gt_letters[i : i + args.batch_size]
        bs       = len(batch_ar)
        input_ids, attn_mask, seq_lens = tokenize_batch(batch_ar)
        en_h = torch.tensor(cached_en[i : i + bs])

        if patch_type == 'full':
            target = transformer_layers[L - 1]
        elif patch_type == 'attn':
            target = transformer_layers[L - 1].self_attn
        else:  # mlp
            target = transformer_layers[L - 1].mlp

        def make_hook(en_h_=en_h, sl_=seq_lens, bs_=bs):
            def hook_fn(module, inp, output):
                o = output[0].clone() if isinstance(output, tuple) else output.clone()
                for j in range(bs_):
                    o[j, sl_[j]] = en_h_[j].to(o.device).to(o.dtype)
                return (o,) + output[1:] if isinstance(output, tuple) else o
            return hook_fn

        handle = target.register_forward_hook(make_hook())
        with torch.no_grad():
            outputs = model(input_ids=input_ids.to("cuda:0"),
                            attention_mask=attn_mask.to("cuda:0"))
        handle.remove()
        config_probs.extend(correct_probs(outputs.logits.cpu(), seq_lens, batch_gt).tolist())
        if i % 20 == 0:
            print(f"  [L{L}/{patch_type}] {min(i+bs, N)}/{N}")
    return np.array(config_probs)

results = {}   # (layer, type) → np.array
for L in CRITICAL_LAYERS:
    print(f"\n--- Layer {L} ---")
    results[(L, 'attn')] = run_patch('attn', L, en_attn_out[L])
    results[(L, 'mlp')]  = run_patch('mlp',  L, en_mlp_out[L])
    results[(L, 'full')] = run_patch('full', L, en_full_hs[L])

# ── Save ──────────────────────────────────────────────────────────────────────
out_npz = os.path.join(args.out_dir, "decomp_results.npz")
save_dict = dict(en_base_prob=en_base_prob, ar_base_prob=ar_base_prob,
                 critical_layers=np.array(CRITICAL_LAYERS))
for (L, t), v in results.items():
    save_dict[f"L{L}_{t}"] = v
np.savez(out_npz, **save_dict)
print(f"\nSaved → {out_npz}")

# ── Summary ───────────────────────────────────────────────────────────────────
ar_mean = ar_base_prob.mean()
en_mean = en_base_prob.mean()
gap     = en_mean - ar_mean

def recovery(val):
    return 100 * (val - ar_mean) / gap if gap > 0 else 0.0

print("\n══ Attn vs MLP Decomposition ══════════════════════════════════")
print(f"  ar_base = {ar_mean:.3f} | en_base = {en_mean:.3f} | gap = {gap:.3f}")
print(f"\n  {'Layer':<8} {'Attn':>10} {'MLP':>10} {'Full':>10} {'Attn%':>8} {'MLP%':>8} {'Full%':>8}")
for L in CRITICAL_LAYERS:
    a = results[(L,'attn')].mean()
    m = results[(L,'mlp')].mean()
    f = results[(L,'full')].mean()
    print(f"  L{L:<7} {a:>10.3f} {m:>10.3f} {f:>10.3f} "
          f"{recovery(a):>7.1f}% {recovery(m):>7.1f}% {recovery(f):>7.1f}%")

# ── Figure ────────────────────────────────────────────────────────────────────
PALETTE = {'Base': '#87CEEB', 'CoT': '#2A9D8F',
           'attn': '#C0392B', 'mlp': '#F4C430', 'full': '#0D3349'}

fig, ax = plt.subplots(figsize=(10, 5))

layers_x = list(range(len(CRITICAL_LAYERS)))
attn_rec = [recovery(results[(L,'attn')].mean()) for L in CRITICAL_LAYERS]
mlp_rec  = [recovery(results[(L,'mlp')].mean())  for L in CRITICAL_LAYERS]
full_rec = [recovery(results[(L,'full')].mean())  for L in CRITICAL_LAYERS]

attn_sem = [results[(L,'attn')].std()/np.sqrt(N)/gap*100 for L in CRITICAL_LAYERS]
mlp_sem  = [results[(L,'mlp')].std()/np.sqrt(N)/gap*100  for L in CRITICAL_LAYERS]
full_sem = [results[(L,'full')].std()/np.sqrt(N)/gap*100  for L in CRITICAL_LAYERS]

ax.plot(layers_x, attn_rec, 'o-', color=PALETTE['attn'], lw=2, label='Attn-only patch')
ax.plot(layers_x, mlp_rec,  's-', color=PALETTE['mlp'],  lw=2, label='MLP-only patch')
ax.plot(layers_x, full_rec, '^-', color=PALETTE['full'], lw=2, label='Full-layer patch')
ax.fill_between(layers_x,
                [a-e for a,e in zip(attn_rec, attn_sem)],
                [a+e for a,e in zip(attn_rec, attn_sem)],
                color=PALETTE['attn'], alpha=0.15)
ax.fill_between(layers_x,
                [a-e for a,e in zip(mlp_rec, mlp_sem)],
                [a+e for a,e in zip(mlp_rec, mlp_sem)],
                color=PALETTE['mlp'], alpha=0.15)

ax.axhline(0,   color=PALETTE['Base'], lw=1.5, ls='--', alpha=0.7, label='Arabic baseline (0%)')
ax.axhline(100, color=PALETTE['CoT'],  lw=1.5, ls='--', alpha=0.7, label='English baseline (100%)')
ax.set_xticks(layers_x)
ax.set_xticklabels([f"L{L}" for L in CRITICAL_LAYERS], fontsize=10)
ax.set_ylabel("Recovery % toward English baseline", fontsize=11)
ax.set_xlabel("Layer", fontsize=11)
ax.set_title(
    "Attention vs MLP Decomposition of Activation Patching\n"
    "Mistral-Small-3.2-24B  ·  MedAraBench  ·  access_gap (En✓ Ar✗)",
    fontsize=12, fontweight="bold")
ax.legend(fontsize=10, frameon=False)
ax.grid(axis="y", alpha=0.2)
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
for ext in [".pdf", ".png"]:
    plt.savefig(os.path.join(args.out_dir, f"fig_decomp_mistral{ext}"),
                bbox_inches="tight", dpi=150)
    print(f"Saved → {os.path.join(args.out_dir, f'fig_decomp_mistral{ext}')}")
plt.close()
print("\nDone.")