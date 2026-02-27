import os
import re

# SET HF_HOME **BEFORE** importing transformers/mistral libraries
# Otherwise they read the system default during import
HF_CACHE = "/scratch/mk8737/huggingface" # update this to a path on your system
os.environ.setdefault("HF_HOME", HF_CACHE)  # Only set if not already set

print(f"1 HF_HOME={os.environ.get('HF_HOME')}")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DeepseekV3Config, DeepseekV3ForCausalLM

print(f"2 HF_HOME={os.environ.get('HF_HOME')}")

os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
os.environ.setdefault("TORCH_USE_CUDA_DSA", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


class DeepSeek32MCQHandler:
    """
    MCQ-only handler.
    - Input is a dict (one sample) with keys: question, opa/opb/opc/opd/(ope/opf optional)
    - Output is a single letter A–F.
    """

    def __init__(
        self,
        model_name: str = "deepseek-ai/DeepSeek-V3.2",
        cache_dir: str = HF_CACHE,
        offline: bool = True,
    ):
        print(f"[DeepSeek32MCQ] Handler file: {__file__}")

        print(f"21 HF_HOME={os.environ.get('HF_HOME')}")

        print(f"[DeepSeek32MCQ] HF_HOME={os.environ.get('HF_HOME')}")
        print(f"[DeepSeek32MCQ] cache_dir={cache_dir}")
        print(f"[DeepSeek32MCQ] offline={offline}")

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
        print(f"[DeepSeek32MCQ] Available GPUs: {num_gpus}")
        if num_gpus == 0:
            raise RuntimeError("No CUDA GPUs available for DeepSeek32MCQ.")

        for i in range(num_gpus):
            print(
                f"  GPU {i}: {torch.cuda.get_device_name(i)} - "
                f"{torch.cuda.memory_allocated(i) / 1024**3:.2f} GB allocated"
            )

        # -------------------------
        # Load tokenizer
        # -------------------------
        print("[DeepSeek32MCQ] Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            local_files_only=self.local_files_only,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # --------------------------
        # Load model
        # --------------------------
        print("[DeepSeek32MCQ] Loading model...")
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                cache_dir=self.cache_dir,
                local_files_only=self.local_files_only,
                trust_remote_code=True,
            )
        except Exception as e:
            if "deepseek_v32" not in str(e):
                raise

            print("[DeepSeek32MCQ] AutoModel could not resolve deepseek_v32; falling back to DeepseekV3 classes.")
            cfg = DeepseekV3Config.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                local_files_only=self.local_files_only,
            )
            self.model = DeepseekV3ForCausalLM.from_pretrained(
                self.model_name,
                config=cfg,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                cache_dir=self.cache_dir,
                local_files_only=self.local_files_only,
            )

        print("[DeepSeek32MCQ] Model loaded.")
        if hasattr(self.model, "hf_device_map"):
            print("[DeepSeek32MCQ] Model device distribution:")
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
        except Exception:
            return (
                f"System:\n{system_prompt}\n\n"
                f"User:\n{user_text}\n\n"
                f"Assistant:\n"
            )

    def prompt(self, sample: dict, instruction: str, max_tokens: int = 12):
        """
        MCQ-only interface:
        - sample is ONE JSON record (dict).
        - returns: 'A'..'F' or None
        """

        user_text = self._build_mcq_text(sample)
        if not user_text:
            print("[DeepSeek32MCQ] Empty stem/options; cannot build prompt.")
            return None

        system_prompt = instruction.strip()

        print("[DeepSeek32MCQ] MAX TOKENS:", max_tokens)
        try:
            torch.cuda.empty_cache()

            prompt_text = self._format_prompt(system_prompt, user_text)

            enc = self.tokenizer(prompt_text, return_tensors="pt")
            input_ids = enc["input_ids"].to(self.model.device)
            attention_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_tokens,
                    do_sample=False,  # greedy
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

        except Exception as e:
            print("[DeepSeek32MCQ] Error during generation:", e)
            return None
        finally:
            torch.cuda.empty_cache()

        if outputs is None or outputs.shape[0] == 0:
            print("[DeepSeek32MCQ] Empty generation output.")
            return None

        input_len = input_ids.shape[1]
        gen_ids = outputs[0][input_len:]
        if isinstance(gen_ids, torch.Tensor):
            gen_ids = gen_ids.detach().cpu().tolist()
        raw_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        if not raw_text:
            return None

        print(f"[DeepSeek32MCQ] MCQ raw generated: {repr(raw_text)}")

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

        print("[DeepSeek32MCQ] Could not extract a clean letter.")
        return None