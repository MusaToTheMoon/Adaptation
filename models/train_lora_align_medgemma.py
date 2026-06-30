#!/usr/bin/env python3
"""
train_lora_align_medgemma.py
----------------------------
Minimal MedGemma adapter for train_lora_align.py.

This file intentionally reuses the original training loop, loss, LoRA config,
beta calibration, logging, and metadata code. The only replaced pieces are the
Mistral-specific tokenizer/model/logit-lens helpers.
"""

import os
import sys
from typing import List, Dict

import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

import train_lora_align as base


HF_CACHE = "/scratch/mk8737/huggingface"
os.environ.setdefault("HF_HOME", HF_CACHE)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def load_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=HF_CACHE,
        local_files_only=os.path.isdir(model_name),
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def tokenize_target_text(tokenizer, text: str) -> List[int]:
    ids = tokenizer.encode(text.strip(), add_special_tokens=False)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is not None:
        ids = ids + [eos_id]
    return ids


def tokenize_plain_text(tokenizer, text: str, max_length: int) -> List[int]:
    """
    Match the original alignment input: BOS + raw text, no chat template, no EOS.
    """
    ids = tokenizer.encode(text.strip(), add_special_tokens=False)
    bos_id = getattr(tokenizer, "bos_token_id", None)
    if bos_id is not None:
        ids = [bos_id] + ids
    return ids[:max_length]


def _as_token_list(encoded):
    if isinstance(encoded, list):
        return encoded
    if isinstance(encoded, dict):
        return encoded["input_ids"]
    if hasattr(encoded, "input_ids"):
        return encoded.input_ids
    if hasattr(encoded, "ids"):
        return encoded.ids
    return list(encoded)


def encode_mcq_example(
    tokenizer,
    system_prompt: str,
    user_text: str,
    gold_letter: str,
) -> Dict[str, List[int]]:
    messages = [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": user_text},
    ]
    prompt_ids = _as_token_list(
        tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors=None,
        )
    )
    target_ids = tokenize_target_text(tokenizer, gold_letter)
    input_ids = prompt_ids + target_ids
    labels = [-100] * len(prompt_ids) + target_ids
    return {"input_ids": input_ids, "labels": labels}


def load_model(model_name: str, use_qlora: bool):
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    device_map = {"": local_rank} if local_rank >= 0 else "auto"
    local_only = os.path.isdir(model_name)

    if use_qlora:
        bnb_config = base.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = _load_causal_model(
            model_name,
            device_map=device_map,
            local_only=local_only,
            quantization_config=bnb_config,
        )
        model = base.prepare_model_for_kbit_training(model)
    else:
        model = _load_causal_model(
            model_name,
            device_map=device_map,
            local_only=local_only,
        )
    return model


def _load_causal_model(model_name: str, device_map, local_only: bool, quantization_config=None):
    common_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": device_map,
        "cache_dir": HF_CACHE,
        "local_files_only": local_only,
    }
    if quantization_config is not None:
        common_kwargs["quantization_config"] = quantization_config

    for loader_name, loader in [
        (
            "Gemma3ForCausalLM",
            lambda: __import__("transformers", fromlist=["Gemma3ForCausalLM"])
            .Gemma3ForCausalLM.from_pretrained(model_name, **common_kwargs),
        ),
        (
            "AutoModelForCausalLM",
            lambda: AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                **common_kwargs,
            ),
        ),
    ]:
        try:
            print(f"  Trying {loader_name} ...")
            return loader()
        except Exception as exc:
            print(f"  {loader_name} failed: {exc}")
    raise RuntimeError(f"Could not load MedGemma model: {model_name}")


def get_norm_and_lmhead(peft_model):
    if hasattr(peft_model, "module"):
        peft_model = peft_model.module
    base_model = peft_model.base_model.model

    if hasattr(base_model, "model") and hasattr(base_model.model, "norm") and hasattr(base_model, "lm_head"):
        return base_model.model.norm, base_model.lm_head

    if hasattr(base_model, "language_model"):
        lm = base_model.language_model
        if hasattr(lm, "model") and hasattr(lm.model, "norm") and hasattr(lm, "lm_head"):
            return lm.model.norm, lm.lm_head
        if hasattr(lm, "norm") and hasattr(lm, "lm_head"):
            return lm.norm, lm.lm_head

    return base.get_norm_and_lmhead(peft_model)


base.load_tokenizer = load_tokenizer
base.tokenize_target_text = tokenize_target_text
base.tokenize_plain_text = tokenize_plain_text
base.encode_mcq_example = encode_mcq_example
base.load_model = load_model
base.get_norm_and_lmhead = get_norm_and_lmhead
base.HF_CACHE = HF_CACHE


if "--model_name" not in sys.argv:
    sys.argv.extend(["--model_name", "google/medgemma-27b-text-it"])


if __name__ == "__main__":
    base.main()
