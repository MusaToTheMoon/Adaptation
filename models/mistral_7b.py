import os
import re
from typing import List, Optional

# SET HF_HOME **BEFORE** importing transformers/mistral libraries
# Otherwise they read the system default during import
HF_CACHE = "/scratch/mk8737/huggingface"  # update this to a path on your system
os.environ.setdefault("HF_HOME", HF_CACHE)  # Only set if not already set

print(f"1 HF_HOME={os.environ.get('HF_HOME')}")

import torch
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from transformers import AutoModelForCausalLM

print(f"2 HF_HOME={os.environ.get('HF_HOME')}")

os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
os.environ.setdefault("TORCH_USE_CUDA_DSA", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


class Mistral7BMCQHandler:
    """
    Unified handler for:
      - task_type="mcq"              -> returns a single letter A–F (or None)
      - task_type="answer_generation"-> returns generated text (or "")

    Input sample (dict):
      - For MCQ: question + opa/opb/opc/opd (+ optional ope/opf)
      - For Answer generation: question (and optionally any extra fields you add later)
    """

    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.3",
        cache_dir: str = HF_CACHE,
        offline: bool = True,
    ):
        print(f"[Mistral7BMCQ] Handler file: {__file__}")

        print(f"21 HF_HOME={os.environ.get('HF_HOME')}")

        print(f"[Mistral7BMCQ] HF_HOME={os.environ.get('HF_HOME')}")
        print(f"[Mistral7BMCQ] cache_dir={cache_dir}")
        print(f"[Mistral7BMCQ] offline={offline}")

        self.model_name = model_name
        self.cache_dir = cache_dir or HF_CACHE
        self.offline = offline

        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

        self.local_files_only = bool(self.offline)
        if self.offline:
            os.environ["HF_HUB_OFFLINE"] = "1"
        else:
            os.environ.pop("HF_HUB_OFFLINE", None)

        # -------------------------
        # Inspect GPUs (debug)
        # -------------------------
        num_gpus = torch.cuda.device_count()
        print(f"[Mistral7BMCQ] Available GPUs: {num_gpus}")
        if num_gpus == 0:
            raise RuntimeError("No CUDA GPUs available for Mistral7BMCQ.")

        for i in range(num_gpus):
            print(
                f"  GPU {i}: {torch.cuda.get_device_name(i)} - "
                f"{torch.cuda.memory_allocated(i) / 1024**3:.2f} GB allocated"
            )

        # -------------------------
        # Load tokenizer
        # -------------------------
        print("[Mistral7BMCQ] Loading tokenizer...")
        self.tokenizer = MistralTokenizer.from_hf_hub(self.model_name)

        # --------------------------
        # Load model
        # --------------------------
        print("[Mistral7BMCQ] Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            cache_dir=self.cache_dir,
            local_files_only=self.local_files_only,
        )

        print("[Mistral7BMCQ] Model loaded.")
        if hasattr(self.model, "hf_device_map"):
            print("[Mistral7BMCQ] Model device distribution:")
            for layer, dev in self.model.hf_device_map.items():
                print(f"  {layer}: {dev}")

        for i in range(num_gpus):
            alloc = torch.cuda.memory_allocated(i) / 1024**3
            reserv = torch.cuda.memory_reserved(i) / 1024**3
            print(f"  GPU {i} after load: {alloc:.2f}GB allocated, {reserv:.2f}GB reserved")

    @staticmethod
    def _build_mcq_text(sample: dict) -> str:
        """
        Build a clean MCQ block from your new JSON schema.

        Expected:
          sample["question"]
          sample["opa"], sample["opb"], sample["opc"], sample["opd"]
          optional: sample["ope"], sample["opf"]
        """
        stem = (sample.get("question") or "").strip()
        if not stem:
            return ""

        # Collect options in order A..F from opa..opf if present
        option_map = {
            "A": sample.get("opa"),
            "B": sample.get("opb"),
            "C": sample.get("opc"),
            "D": sample.get("opd"),
            "E": sample.get("ope"),
            "F": sample.get("opf"),
        }

        lines = []
        for letter in ["A", "B", "C", "D", "E", "F"]:
            txt = option_map.get(letter)
            if txt is None:
                continue
            txt = str(txt).strip()
            if txt == "":
                continue
            lines.append(f"{letter}) {txt}")

        if len(lines) < 2:
            # still allow it, but it’s probably malformed data
            pass

        return stem + "\n\n" + "\n".join(lines)

    @staticmethod
    def _build_ansgen_text(sample: dict) -> str:
        """
        For answer_generation:
        Only expose the question stem (no options).
        """
        question = (sample.get("question") or "").strip()
        if not question:
            return ""

        return question

    # -------------------------
    # Core generation
    # -------------------------
    def _generate(self, system_prompt: str, user_text: str, max_tokens: int, do_sample: bool = False) -> str:
        messages = [
            {"role": "system", "content": (system_prompt or "").strip()},
            {"role": "user", "content": [{"type": "text", "text": user_text}]},
        ]

        try:
            torch.cuda.empty_cache()

            req = ChatCompletionRequest(messages=messages)
            tokenized = self.tokenizer.encode_chat_completion(req)

            input_ids = torch.tensor([tokenized.tokens], dtype=torch.long, device=self.model.device)
            attention_mask = torch.ones_like(input_ids)

            with torch.inference_mode():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_tokens,
                    do_sample=do_sample,
                )

            if outputs is None or outputs.shape[0] == 0:
                print(f"[Mistral7BMCQ] Generation produced no output tokens (max_new_tokens={max_tokens})")
                return ""

            input_len = len(tokenized.tokens)
            gen_ids = outputs[0][input_len:]
            generated_tokens = int(gen_ids.shape[0])
            print(
                f"[Mistral7BMCQ] Token usage: generated={generated_tokens} "
                f"max_new_tokens={max_tokens} prompt_tokens={input_len}"
            )
            if isinstance(gen_ids, torch.Tensor):
                gen_ids = gen_ids.detach().cpu().tolist()
            raw_text = self.tokenizer.decode(gen_ids).strip()
            return raw_text or ""

        finally:
            torch.cuda.empty_cache()

    # -------------------------
    # Public API
    # -------------------------
    def prompt(self, sample: dict, instruction: str, max_tokens: int = 12, task_type: str = "mcq"):
        """
        task_type:
          - "mcq": returns A-F or None
          - "answer_generation": returns generated string (may be empty string)
        """
        task_type = (task_type or "mcq").strip().lower()

        if task_type == "mcq":
            user_text = self._build_mcq_text(sample)
            if not user_text:
                print("[Mistral7BMCQ] Empty stem/options; cannot build MCQ prompt.")
                return None

            system_prompt = (instruction or "").strip()

            raw_text = self._generate(system_prompt=system_prompt, user_text=user_text, max_tokens=max_tokens)

            if not raw_text:
                return None

            print(f"[Mistral7BMCQ] MCQ raw generated: {repr(raw_text)}")
            upper = raw_text.upper()

            m = re.search(r"\bANSWER\s*[:=]\s*([A-F])\b", upper)
            if m:
                return m.group(1)

            m = re.search(r"\b([A-F])\b", upper)
            if m:
                return m.group(1)

            print("[Mistral7BMCQ] Could not extract a clean letter.")
            return None

        if task_type == "answer_generation":
            user_text = self._build_ansgen_text(sample)
            if not user_text:
                print("[Mistral7BMCQ] Empty question; cannot build answer-generation prompt.")
                return ""

            system_prompt = (instruction or "").strip()

            raw_text = self._generate(system_prompt=system_prompt, user_text=user_text, max_tokens=max_tokens)

            if not raw_text:
                return ""

            one_line = raw_text.split("\n")[0].strip()
            print(f"[Mistral7BMCQ] Answer-gen raw (one line): {repr(one_line[:300])}")
            return one_line

        raise ValueError(f"Unsupported task_type={task_type}. Expected 'mcq' or 'answer_generation'.")

    def prompt_batch(
        self,
        samples: List[dict],
        instruction: str,
        max_tokens: int = 12,
        task_type: str = "mcq",
    ):
        # Simple safe batching (sequential). Real batching is possible but more work with mistral_common.
        return [self.prompt(s, instruction=instruction, max_tokens=max_tokens, task_type=task_type) for s in samples]