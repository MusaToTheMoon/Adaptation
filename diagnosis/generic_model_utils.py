import os
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


SYSTEM_PROMPT = (
    "You are a medical expert. "
    "Answer the following multiple choice question "
    "by responding with only the letter of the correct option: A, B, C, or D. "
    "Do not explain your answer."
)


def load_tokenizer(model_path: str):
    tok = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    if tok.pad_token_id is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return tok


def disable_model_cache(model):
    if hasattr(model, "config"):
        model.config.use_cache = False
    if hasattr(model, "generation_config"):
        model.generation_config.use_cache = False
    return model


def disable_config_cache(config):
    config.use_cache = False
    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        text_config.use_cache = False
    return config


def default_torch_dtype():
    if not torch.cuda.is_available():
        return torch.float32
    major, minor = torch.cuda.get_device_capability(0)
    if major >= 8:
        return torch.bfloat16
    return torch.float16


def load_causal_lm(model_path: str, dtype=None):
    if dtype is None:
        dtype = default_torch_dtype()
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        capability = torch.cuda.get_device_capability(0)
        print(f"  Loading dtype: {dtype} on {name} (capability {capability[0]}.{capability[1]})", flush=True)
    else:
        print(f"  Loading dtype: {dtype} on CPU", flush=True)

    loaders = []
    cfg = AutoConfig.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    disable_config_cache(cfg)
    model_type = getattr(cfg, "model_type", "")
    architectures = set(getattr(cfg, "architectures", []) or [])

    if "Gemma3ForCausalLM" in architectures:
        try:
            from transformers import Gemma3ForCausalLM
            loaders.append(
                (
                    "Gemma3ForCausalLM",
                    lambda: Gemma3ForCausalLM.from_pretrained(
                        model_path,
                        torch_dtype=dtype,
                        device_map="auto",
                        low_cpu_mem_usage=True,
                        local_files_only=True,
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
                        model_path,
                        torch_dtype=dtype,
                        device_map="auto",
                        low_cpu_mem_usage=True,
                        local_files_only=True,
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
                        model_path,
                        torch_dtype=dtype,
                        device_map="auto",
                        low_cpu_mem_usage=True,
                        local_files_only=True,
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
                model_path,
                torch_dtype=dtype,
                device_map="auto",
                low_cpu_mem_usage=True,
                local_files_only=True,
                trust_remote_code=True,
                config=cfg,
            ),
        )
    )

    last_error = None
    for name, fn in loaders:
        try:
            print(f"  Trying {name} ...", flush=True)
            model = fn()
            print(f"  Loaded via {name}", flush=True)
            disable_model_cache(model)
            model.eval()
            return model
        except Exception as exc:
            print(f"  {name} failed: {exc}", flush=True)
            last_error = exc
    raise RuntimeError(f"Could not load model from {model_path}: {last_error}")


def get_layers(model):
    candidates = [
        ("model.layers", lambda m: m.model.layers),
        ("model.model.layers", lambda m: m.model.model.layers),
        ("transformer.h", lambda m: m.transformer.h),
        ("model.transformer.h", lambda m: m.model.transformer.h),
        ("gpt_neox.layers", lambda m: m.gpt_neox.layers),
        ("model.gpt_neox.layers", lambda m: m.model.gpt_neox.layers),
        ("language_model.model.layers", lambda m: m.language_model.model.layers),
        ("model.language_model.model.layers", lambda m: m.model.language_model.model.layers),
        ("model.language_model.layers", lambda m: m.model.language_model.layers),
    ]
    for name, fn in candidates:
        try:
            layers = fn(model)
            if layers is not None and len(layers) > 0:
                print(f"  Transformer layers at {name} (n={len(layers)})")
                return layers
        except Exception:
            pass
    raise RuntimeError("Could not locate transformer layers.")


def get_norm_and_lm_head(model):
    candidates = [
        ("model.norm + lm_head", lambda m: (m.model.norm, m.lm_head)),
        ("model.model.norm + lm_head", lambda m: (m.model.model.norm, m.lm_head)),
        ("model.final_layernorm + lm_head", lambda m: (m.model.final_layernorm, m.lm_head)),
        ("final_layernorm + lm_head", lambda m: (m.final_layernorm, m.lm_head)),
        ("transformer.ln_f + lm_head", lambda m: (m.transformer.ln_f, m.lm_head)),
        ("model.transformer.ln_f + lm_head", lambda m: (m.model.transformer.ln_f, m.lm_head)),
        ("gpt_neox.final_layer_norm + embed_out", lambda m: (m.gpt_neox.final_layer_norm, m.embed_out)),
        ("model.gpt_neox.final_layer_norm + embed_out", lambda m: (m.model.gpt_neox.final_layer_norm, m.embed_out)),
        ("language_model.model.norm + language_model.lm_head", lambda m: (m.language_model.model.norm, m.language_model.lm_head)),
        ("model.language_model.model.norm + lm_head", lambda m: (m.model.language_model.model.norm, m.lm_head)),
        ("model.language_model.norm + lm_head", lambda m: (m.model.language_model.norm, m.lm_head)),
    ]
    for name, fn in candidates:
        try:
            norm, lm_head = fn(model)
            if norm is not None and lm_head is not None:
                print(f"  Projection at {name}")
                return norm, lm_head
        except Exception:
            pass
    raise RuntimeError("Could not locate final norm and LM head.")


def build_prompt(tokenizer, text: str, max_len: int) -> List[int]:
    text = text or "[empty]"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]
    try:
        chat_str = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded = tokenizer(chat_str)
        ids = encoded["input_ids"]
        if hasattr(ids, "ids"):
            ids = ids.ids
        elif hasattr(ids, "tolist"):
            ids = ids.tolist()
        elif isinstance(ids, dict):
            ids = ids["input_ids"]
    except Exception:
        try:
            ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors=None,
            )
        except Exception:
            prompt = f"{SYSTEM_PROMPT}\n\n{text}\n\nAnswer:"
            ids = tokenizer.encode(prompt, add_special_tokens=True)
        if not isinstance(ids, list):
            ids = ids["input_ids"] if isinstance(ids, dict) else ids.tolist()
    return list(ids)[:max_len]


def batch_tokenize(tokenizer, texts: Sequence[str], max_len: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    encoded = [build_prompt(tokenizer, t, max_len) for t in texts]
    bs = len(encoded)
    max_l = max(len(ids) for ids in encoded)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    input_ids = torch.full((bs, max_l), pad, dtype=torch.long)
    attn_mask = torch.zeros((bs, max_l), dtype=torch.long)
    for i, ids in enumerate(encoded):
        sl = len(ids)
        input_ids[i, max_l - sl:] = torch.tensor(ids, dtype=torch.long)
        attn_mask[i, max_l - sl:] = 1
    seq_lens = torch.full((bs,), max_l - 1, dtype=torch.long)
    return input_ids, attn_mask, seq_lens


def answer_token_ids(tokenizer) -> Dict[str, List[int]]:
    out = {}
    forms = ["{}", " {}", "\n{}", "ANSWER: {}", "Answer: {}"]
    for letter in "ABCDEF":
        chosen = None
        for fmt in forms:
            try:
                toks = tokenizer.encode(fmt.format(letter), add_special_tokens=False)
            except Exception:
                continue
            if len(toks) == 1:
                chosen = toks[0]
                break
            if toks and chosen is None:
                chosen = toks[-1]
        if chosen is None:
            raise RuntimeError(f"Could not derive token id for answer letter {letter}")
        out[letter] = [chosen]
    print("  Answer token ids:")
    for letter, ids in out.items():
        dec = []
        for tid in ids:
            try:
                dec.append(repr(tokenizer.decode([tid])))
            except Exception:
                dec.append("?")
        print(f"    {letter}: {ids} decoded={dec}")
    return out


def correct_probs(logits: torch.Tensor, seq_lens: torch.Tensor, gt_letters: Sequence[str], answer_ids: Dict[str, List[int]]):
    probs = torch.softmax(logits.float(), dim=-1)
    vals = []
    for i, gt in enumerate(gt_letters):
        ids = answer_ids.get(gt, [])
        if not ids:
            vals.append(0.0)
            continue
        idx = torch.tensor(ids, dtype=torch.long, device=probs.device)
        vals.append(float(probs[i, int(seq_lens[i]), idx].sum().item()))
    return vals


def default_probe_layers(n_layers: int) -> List[int]:
    raw = [max(1, round(n_layers * f)) for f in (0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875)]
    raw += [max(1, n_layers - 3), max(1, n_layers - 2), max(1, n_layers - 1), n_layers]
    return sorted(set(x for x in raw if 1 <= x <= n_layers))


def parse_layers(value: str, n_layers: int) -> List[int]:
    if not value:
        return default_probe_layers(n_layers)
    layers = sorted(set(int(x.strip()) for x in value.split(",") if x.strip()))
    bad = [x for x in layers if x < 1 or x > n_layers]
    if bad:
        raise ValueError(f"Probe layers out of range 1..{n_layers}: {bad}")
    return layers


def patch_configs(probe_layers: Sequence[int]) -> Dict[str, List[int]]:
    layers = list(probe_layers)
    cfg = {f"patch_L{l}": [l] for l in layers}
    if len(layers) >= 3:
        cfg[f"patch_L{layers[-3]}_{layers[-1]}"] = layers[-3:]
    if len(layers) >= 5:
        cfg[f"patch_L{layers[-5]}_{layers[-1]}"] = layers[-5:]
    return cfg
