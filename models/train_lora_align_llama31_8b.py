#!/usr/bin/env python3
"""
train_lora_align_llama31_8b.py
------------------------------
Llama-3.1-8B-Instruct adapter for train_lora_align.py.

This reuses the shared training loop, LoRA config, CE+KL alignment loss,
beta calibration, logging, and metadata code. Only tokenizer/model loading
and chat-template encoding are replaced.
"""

import os
import sys
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import train_lora_align as base


HF_CACHE = "/scratch/mk8737/huggingface"
os.environ.setdefault("HF_HOME", HF_CACHE)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

_BASE_GET_NORM_AND_LMHEAD = base.get_norm_and_lmhead


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

    common_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": device_map,
        "cache_dir": HF_CACHE,
        "local_files_only": local_only,
        "trust_remote_code": True,
    }

    if use_qlora:
        bnb_config = base.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        common_kwargs["quantization_config"] = bnb_config

    model = AutoModelForCausalLM.from_pretrained(model_name, **common_kwargs)
    if hasattr(model, "config"):
        model.config.use_cache = False
    if hasattr(model, "generation_config"):
        model.generation_config.use_cache = False
    if use_qlora:
        model = base.prepare_model_for_kbit_training(model)
    return model


def get_norm_and_lmhead(peft_model):
    return _BASE_GET_NORM_AND_LMHEAD(peft_model)


base.load_tokenizer = load_tokenizer
base.tokenize_target_text = tokenize_target_text
base.tokenize_plain_text = tokenize_plain_text
base.encode_mcq_example = encode_mcq_example
base.load_model = load_model
base.get_norm_and_lmhead = get_norm_and_lmhead
base.HF_CACHE = HF_CACHE


if "--model_name" not in sys.argv:
    sys.argv.extend(["--model_name", "meta-llama/Llama-3.1-8B-Instruct"])


if __name__ == "__main__":
    base.main()
