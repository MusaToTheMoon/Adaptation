import os
import re

# SET HF_HOME **BEFORE** importing transformers libraries
# Otherwise they read the system default during import
HF_CACHE = "/scratch/mk8737/huggingface"  # update this to a path on your system
os.environ.setdefault("HF_HOME", HF_CACHE)  # Only set if not already set

print(f"1 HF_HOME={os.environ.get('HF_HOME')}")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print(f"2 HF_HOME={os.environ.get('HF_HOME')}")

os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
os.environ.setdefault("TORCH_USE_CUDA_DSA", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


class ALLaM7BInstPrevMCQHandler:
    """
    MCQ-only handler.
    - Input is a dict (one sample) with keys: question, opa/opb/opc/opd/(ope/opf optional)
    - Output is a single letter A–F.
    """

    def __init__(
        self,
        model_name: str = "humain-ai/ALLaM-7B-Instruct-preview",
        cache_dir: str = HF_CACHE,
        offline: bool = True,
    ):
        print(f"[ALLaM7BInstPrevMCQ] Handler file: {__file__}")

        print(f"21 HF_HOME={os.environ.get('HF_HOME')}")

        print(f"[ALLaM7BInstPrevMCQ] HF_HOME={os.environ.get('HF_HOME')}")
        print(f"[ALLaM7BInstPrevMCQ] cache_dir={cache_dir}")
        print(f"[ALLaM7BInstPrevMCQ] offline={offline}")

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
        print(f"[ALLaM7BInstPrevMCQ] Available GPUs: {num_gpus}")
        if num_gpus == 0:
            raise RuntimeError("No CUDA GPUs available for ALLaM7BInstPrevMCQ.")

        for i in range(num_gpus):
            print(
                f"  GPU {i}: {torch.cuda.get_device_name(i)} - "
                f"{torch.cuda.memory_allocated(i) / 1024**3:.2f} GB allocated"
            )

        # -------------------------
        # Load tokenizer
        # -------------------------
        print("[ALLaM7BInstPrevMCQ] Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            local_files_only=self.local_files_only,
            trust_remote_code=True,
            use_fast=False,
        )

        # --------------------------
        # Load model
        # --------------------------
        print("[ALLaM7BInstPrevMCQ] Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            cache_dir=self.cache_dir,
            local_files_only=self.local_files_only,
            trust_remote_code=True,
        )

        print("[ALLaM7BInstPrevMCQ] Model loaded.")
        if hasattr(self.model, "hf_device_map"):
            print("[ALLaM7BInstPrevMCQ] Model device distribution:")
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

    def prompt(self, sample: dict, instruction: str, max_tokens: int = 12):
        """
        MCQ-only interface:
        - sample is ONE JSON record (dict).
        - returns: 'A'..'F' or None
        """

        user_text = self._build_mcq_text(sample)
        if not user_text:
            print("[ALLaM7BInstPrevMCQ] Empty stem/options; cannot build prompt.")
            return None

        system_prompt = instruction.strip()

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ]

        print("[ALLaM7BInstPrevMCQ] MAX TOKENS:", max_tokens)
        try:
            torch.cuda.empty_cache()

            chat_input = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
            inputs = self.tokenizer(
                chat_input,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,  # greedy
                )

        except Exception as e:
            print("[ALLaM7BInstPrevMCQ] Error during generation:", e)
            return None
        finally:
            torch.cuda.empty_cache()

        if outputs is None or outputs.shape[0] == 0:
            print("[ALLaM7BInstPrevMCQ] Empty generation output.")
            return None

        input_len = inputs["input_ids"].shape[-1]
        gen_ids = outputs[0][input_len:]
        raw_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        if not raw_text:
            return None

        print(f"[ALLaM7BInstPrevMCQ] MCQ raw generated: {repr(raw_text)}")

        # Extract letter
        upper = raw_text.upper()

        # Prefer strict format: ANSWER: X
        m = re.search(r"\bANSWER\s*[:=]\s*([A-F])\b", upper)
        if m:
            return m.group(1)

        # Fallbacks (if model deviates)
        m = re.search(r"\b([A-F])\b", upper)
        if m:
            return m.group(1)

        print("[ALLaM7BInstPrevMCQ] Could not extract a clean letter.")
        return None
