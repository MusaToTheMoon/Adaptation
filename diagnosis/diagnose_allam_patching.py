"""
diagnose_allam_patching.py
---------------------------
Targeted diagnostic for ALLaM activation patching issues.

Checks:
  1. What tokens the model actually predicts (top-10) at last position for EN/AR inputs
  2. What the model actually generates (greedy, 5 tokens) for EN/AR inputs
  3. Whether seq_lens is pointing to the right position
  4. Whether NaN appears after multi-layer patching

Usage:
  python diagnose_allam_patching.py \
      --csv allam_sampled_quadrants.csv \
      --model_path $ALLAM
"""

import os, csv, re, argparse
import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--csv",        required=True)
parser.add_argument("--model_path", required=True)
parser.add_argument("--max_len",    type=int, default=512)
args = parser.parse_args()

SYSTEM_PROMPT = (
    "You are a medical expert. "
    "Answer the following multiple choice question "
    "by responding with only the letter of the correct option: A, B, C, or D. "
    "Do not explain your answer."
)

ANSWER_TOKEN_IDS = {"A": 395, "B": 482, "C": 415, "D": 526}

def extract_letter(val):
    if not val or not isinstance(val, str): return ""
    s = val.strip().upper()
    m = re.search(r'\bANSWER\s*:\s*([A-F])\b', s)
    if m: return m.group(1)
    m = re.search(r'\b([A-F])\b', s)
    return m.group(1) if m else ""

# ── Load a few access_gap examples ────────────────────────────────────────────
rows = []
with open(args.csv, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        rows.append(row)
ag_rows = [r for r in rows if r["quadrant"] == "access_gap"][:5]
print(f"Using {len(ag_rows)} access_gap examples for diagnosis\n")

# ── Load model ────────────────────────────────────────────────────────────────
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, trust_remote_code=True)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    args.model_path, torch_dtype=torch.bfloat16,
    device_map="auto", local_files_only=True, trust_remote_code=True)
model.eval()
print(f"Model: {type(model).__name__}\n")

# ── Token ID sanity check ──────────────────────────────────────────────────────
print("=" * 60)
print("SECTION 1: Answer token ID verification")
print("=" * 60)
for letter, tid in ANSWER_TOKEN_IDS.items():
    decoded = tokenizer.decode([tid])
    print(f"  Token {tid} → {repr(decoded)}   (expected '{letter}')")

# Also check Arabic equivalents just in case
arabic_letters = {"أ": None, "ب": None, "ج": None, "د": None, "A": None, "B": None, "C": None, "D": None}
for ch in arabic_letters:
    tids = tokenizer.encode(ch, add_special_tokens=False)
    arabic_letters[ch] = tids
    print(f"  '{ch}' encodes to: {tids}")

# ── Tokenize helper (single example, no batching) ─────────────────────────────
def tokenize_single(text):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": text},
    ]
    try:
        ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors=None)
        if not isinstance(ids, list):
            try: ids = ids["input_ids"]
            except: ids = ids.ids
    except Exception:
        ids = tokenizer.encode(text, add_special_tokens=True)
    ids = list(ids)[:args.max_len]
    input_ids = torch.tensor([ids], dtype=torch.long)
    attn_mask  = torch.ones_like(input_ids)
    seq_len    = len(ids) - 1   # last position (no padding, single example)
    return input_ids, attn_mask, seq_len, ids

# ── Per-example diagnosis ──────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 2: Per-example diagnosis (EN and AR)")
print("=" * 60)

for idx, row in enumerate(ag_rows):
    gt = extract_letter(row["ground_truth"])
    print(f"\n--- Example {idx+1}  GT={gt} ---")

    for lang, text in [("EN", row["input_english"]), ("AR", row["input_arabic"])]:
        input_ids, attn_mask, seq_len, ids = tokenize_single(text)

        print(f"\n  [{lang}] seq_len={seq_len}, n_tokens={len(ids)}")
        print(f"  [{lang}] Last 3 token IDs: {ids[-3:]}")
        print(f"  [{lang}] Last 3 tokens decoded: {[repr(tokenizer.decode([t])) for t in ids[-3:]]}")

        # Forward pass — check logit at last position
        with torch.no_grad():
            out = model(input_ids=input_ids.to("cuda:0"),
                        attention_mask=attn_mask.to("cuda:0"))

        logits_last = out.logits[0, seq_len].float()
        probs_last  = torch.softmax(logits_last, dim=-1)

        # Check if NaN
        if torch.isnan(logits_last).any():
            print(f"  [{lang}] *** NaN in logits! ***")

        # Top-10 predicted tokens
        top10_ids  = probs_last.argsort(descending=True)[:10]
        top10      = [(tokenizer.decode([i.item()]).strip(), f"{probs_last[i].item():.4f}")
                      for i in top10_ids]
        print(f"  [{lang}] Top-10 next tokens: {top10}")

        # P at our answer token IDs
        for letter, tid in ANSWER_TOKEN_IDS.items():
            p = probs_last[tid].item()
            marker = " ← GT" if letter == gt else ""
            print(f"  [{lang}] P(token {tid}='{letter}'): {p:.4f}{marker}")

        # Greedy generation (5 tokens)
        with torch.no_grad():
            gen = model.generate(input_ids=input_ids.to("cuda:0"),
                                 attention_mask=attn_mask.to("cuda:0"),
                                 max_new_tokens=5, do_sample=False)
        gen_ids  = gen[0][len(ids):].tolist()
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        print(f"  [{lang}] Greedy generation (5 tokens): {repr(gen_text)}  token_ids={gen_ids}")

print("\n\nDone. Review SECTION 2 to identify correct answer token IDs.")