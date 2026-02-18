import os
import re
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

HF_CACHE = "/scratch/mk8737/huggingface"
os.environ.setdefault("HF_HOME", HF_CACHE)  # Only set if not already set

os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
os.environ.setdefault("TORCH_USE_CUDA_DSA", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


class MedGemma27BMCQHandler:
    """
    MCQ-only handler.
    - Input is a dict (one sample) with keys: question, opa/opb/opc/opd/(ope/opf optional)
    - Output is a single letter A–F.
    """

    def __init__(
        self,
        model_name: str = "google/medgemma-27b-text-it",
        cache_dir: str = HF_CACHE,
        offline: bool = True,
    ):
        print(f"[MedGemma27BMCQ] Handler file: {__file__}")
        print(f"[MedGemma27BMCQ] HF_HOME={os.environ.get('HF_HOME')}")
        print(f"[MedGemma27BMCQ] cache_dir={cache_dir}")
        print(f"[MedGemma27BMCQ] offline={offline}")

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
        print(f"[MedGemma27BMCQ] Available GPUs: {num_gpus}")
        if num_gpus == 0:
            raise RuntimeError("No CUDA GPUs available for MedGemma27BMCQ.")

        for i in range(num_gpus):
            print(
                f"  GPU {i}: {torch.cuda.get_device_name(i)} - "
                f"{torch.cuda.memory_allocated(i) / 1024**3:.2f} GB allocated"
            )

        # -------------------------
        # Load tokenizer
        # -------------------------
        print("[MedGemma27BMCQ] Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            local_files_only=self.local_files_only,
            use_fast=True,
        )

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token


        # --------------------------
        # Load model
        # --------------------------
        print("[MedGemma27BMCQ] Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            cache_dir=self.cache_dir,
            local_files_only=self.local_files_only,
        )

        print("[MedGemma27BMCQ] Model loaded.")
        if hasattr(self.model, "hf_device_map"):
            print("[MedGemma27BMCQ] Model device distribution:")
            for layer, dev in self.model.hf_device_map.items():
                print(f"  {layer}: {dev}")

        for i in range(num_gpus):
            alloc = torch.cuda.memory_allocated(i) / 1024**3
            reserv = torch.cuda.memory_reserved(i) / 1024**3
            print(f"  GPU {i} after load: {alloc:.2f}GB allocated, {reserv:.2f}GB reserved")

    @staticmethod
    def _build_mcq_text(sample: dict) -> str:
        """
        Build a clean MCQ block from your JSON schema.

        Expected:
          sample["question"]
          sample["opa"], sample["opb"], sample["opc"], sample["opd"]
          optional: sample["ope"], sample["opf"]
        """
        stem = (sample.get("question") or "").strip()
        if not stem:
            return ""

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

        return stem + "\n\n" + "\n".join(lines)

    def prompt(self, sample: dict, instruction: str, max_tokens: int = 12):
        """
        MCQ-only interface:
        - sample is ONE JSON record (dict).
        - returns: 'A'..'F' or None
        """
        user_text = self._build_mcq_text(sample)
        if not user_text:
            print("[MedGemma27BMCQ] Empty stem/options; cannot build prompt.")
            return None

        system_prompt = (instruction or "").strip()

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ]

        print("[MedGemma27BMCQ] MAX TOKENS:", max_tokens)

        try:
            torch.cuda.empty_cache()

            # MedGemma snippet style: apply_chat_template -> tensors on model.device
            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )

            # Put inputs on the main model device (works with device_map="auto" too)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            input_len = inputs["input_ids"].shape[-1]

            with torch.inference_mode():
                generation = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,  # greedy
                )

            gen_ids = generation[0][input_len:]
            raw_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        except Exception as e:
            print("[MedGemma27BMCQ] Error during generation:", e)
            return None
        finally:
            torch.cuda.empty_cache()

        if not raw_text:
            return None

        print(f"[MedGemma27BMCQ] MCQ raw generated: {repr(raw_text)}")

        upper = raw_text.upper()

        # Prefer strict format: ANSWER: X
        m = re.search(r"\bANSWER\s*[:=]\s*([A-F])\b", upper)
        if m:
            return m.group(1)

        # Fallback: first standalone A-F
        m = re.search(r"\b([A-F])\b", upper)
        if m:
            return m.group(1)

        print("[MedGemma27BMCQ] Could not extract a clean letter.")
        return None
