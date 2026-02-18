'''
STILL NEED TO FIX FoR NEW TASKS
'''

import os
import re
import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

HF_CACHE = "/scratch/ca2627/huggingface"

os.environ.setdefault("HF_HOME", HF_CACHE)  # Only set if not already set
os.environ["HF_HUB_OFFLINE"] = "1"

os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
os.environ.setdefault("TORCH_USE_CUDA_DSA", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

class Llama70MCQHandler:
    """
    Llama-3.3-70B-Instruct handler for MCQ tasks (A–F).
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.3-70B-Instruct",
        cache_dir: str = HF_CACHE,
        matching_mode: str = "letter",
    ):
        print(f"[Llama33MCQ] Handler file: {__file__}")
        print(f"[Llama33MCQ] HF_HOME={os.environ.get('HF_HOME')}")
        print(f"[Llama33MCQ] cache_dir={cache_dir}")
        print(f"[Llama33MCQ] matching_mode={matching_mode}")

        self.model_name = model_name
        self.cache_dir = cache_dir or HF_CACHE
        self.matching_mode = matching_mode

        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

        num_gpus = torch.cuda.device_count()
        print(f"[Llama33MCQ] Available GPUs: {num_gpus}")
        if num_gpus == 0:
            raise RuntimeError("No CUDA GPUs available for Llama33MCQ.")

        for i in range(num_gpus):
            print(
                f"  GPU {i}: {torch.cuda.get_device_name(i)} - "
                f"{torch.cuda.memory_allocated(i) / 1024**3:.2f} GB allocated"
            )

        print("[Llama33MCQ] Loading tokenizer (offline, local cache only)...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            local_files_only=True,
        )

        # IMPORTANT for decoder-only batching
        self.tokenizer.padding_side = "left"

        # Ensure pad token exists
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        print("[Llama33MCQ] Setting up 4-bit quantization...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,  # CHANGED (was bfloat16)
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        print("[Llama33MCQ] Loading model in 4-bit (offline, local cache only)...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,  # ADDED
            cache_dir=self.cache_dir,
            local_files_only=True,
        )

        print("[Llama33MCQ] Model + tokenizer loaded (using model.generate).")

        if hasattr(self.model, "hf_device_map"):
            print("[Llama33MCQ] Model device distribution:")
            for layer, dev in self.model.hf_device_map.items():
                print(f"  {layer}: {dev}")

        for i in range(num_gpus):
            alloc = torch.cuda.memory_allocated(i) / 1024**3
            reserv = torch.cuda.memory_reserved(i) / 1024**3
            print(f"  GPU {i} after load: {alloc:.2f}GB allocated, {reserv:.2f}GB reserved")

    # ----------------------------
    # helpers (internal)
    # ----------------------------
    def _extract_user_text(self, input_text: str):
        options = re.findall(
            r"[A-F]\s*[\.\)]\s*.*?(?=\s+[A-F]\s*[\.\)]|\s*$)",
            input_text,
            flags=re.DOTALL,
        )
        if not options:
            return None

        question = input_text.split(options[0])[0].strip()
        options_block = "\n".join(options)
        return f"{question}\n\nOptions:\n{options_block}"

    def _system_prompt(self, instruction: str) -> str:
        return instruction.strip()

    def _format_prompt(self, system_prompt: str, user_text: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ]
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception as e:
            print("[Llama33MCQ] chat_template failed, using fallback:", e)
            return (
                f"<|system|>\n{messages[0]['content']}\n"
                f"<|user|>\n{messages[1]['content']}\n"
                f"<|assistant|>\n"
            )

    def _postprocess(self, raw_text: str):
        raw_text = (raw_text or "").strip()
        if not raw_text:
            return None

        # modes that return raw
        if self.matching_mode in ("text", "confidence_letter"):
            return raw_text

        # letter mode: extract
        upper = raw_text.upper()
        m = re.search(r"\bANSWER\s*[:=]\s*([A-F])\b", upper)
        if m:
            return m.group(1)

        m = re.search(r"\b(?:YOUR\s+ANSWER|ANSWER)\s*(?:IS|:)?\s*([A-F])\b", upper)
        if not m:
            m = re.search(r"^\s*([A-F])\s*[\.\)]", upper)
        if not m:
            m = re.search(r"\b([A-F])\b", upper)

        return m.group(1) if m else None

    # ----------------------------
    # batching support (FAST: model.generate)
    # ----------------------------
    def prompt_batch(self, input_texts, instruction: str, task_type=None, max_tokens: int = 68):
        max_tokens = 120
        system_prompt = self._system_prompt(instruction)

        prompt_strs = [None] * len(input_texts)
        valid_positions = []

        for i, input_text in enumerate(input_texts):
            user_text = self._extract_user_text(input_text)
            if user_text is None:
                continue
            prompt_strs[i] = self._format_prompt(system_prompt, user_text)
            valid_positions.append(i)

        if not valid_positions:
            return [None] * len(input_texts)

        valid_prompts = [prompt_strs[i] for i in valid_positions]

        # DEBUG: show prompt length once
        tok_lens = [len(self.tokenizer(p, add_special_tokens=False).input_ids) for p in valid_prompts]
        print(f"[Llama33MCQ] prompt tokens: min={min(tok_lens)}, avg={sum(tok_lens)/len(tok_lens):.1f}, max={max(tok_lens)}")
        print("[Llama33MCQ] sample prompt head:", repr(valid_prompts[0][:300]))
        print("[Llama33MCQ] sample prompt tail:", repr(valid_prompts[0][-300:]))

        # Tokenize once (batched)
        enc = self.tokenizer(
            valid_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        input_ids = enc["input_ids"].to(self.model.device)
        attention_mask = enc["attention_mask"].to(self.model.device)

        gen_kwargs = dict(
            max_new_tokens=max_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # greedy decoding
        gen_kwargs.update(dict(do_sample=False))

        try:
            with torch.inference_mode():
                out_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **gen_kwargs,
                )
        except Exception as e:
            print("[Llama33MCQ] Error during batch generation:", e)
            return [None] * len(input_texts)

        print(f"[Llama33MCQ] GENERATION CONFIG → max_new_tokens={gen_kwargs['max_new_tokens']}")

        # Decode only newly generated tokens per sample (handles left-padding correctly)
        gen_texts = []
        seq_len = input_ids.shape[1]  # full padded input length

        for row in range(out_ids.shape[0]):
            new_tokens = out_ids[row, seq_len:]  # strip the entire padded prompt
            text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            gen_texts.append((text or "").strip())

        results = [None] * len(input_texts)

        for out_i, pos in enumerate(valid_positions):
            raw_text = gen_texts[out_i]
            print(f"[Llama33MCQ] MCQ raw generated: {repr(raw_text)}")
            results[pos] = self._postprocess(raw_text)

        return results

    # ----------------------------
    # per-example prompt() (wraps batch)
    # ----------------------------
    def prompt(self, input_text: str, instruction: str, task_type=None, max_tokens: int = 15):
        max_tokens = 120
        outs = self.prompt_batch(
            [input_text],
            instruction=instruction,
            task_type=task_type,
            max_tokens=max_tokens,
        )
        return outs[0] if outs else None
