import os
import re

# SET HF_HOME **BEFORE** importing transformers/mistral libraries
# Otherwise they read the system default during import
HF_CACHE = "/scratch/mk8737/huggingface" # update this to a path on your system
os.environ.setdefault("HF_HOME", HF_CACHE)  # Only set if not already set

print(f"1 HF_HOME={os.environ.get('HF_HOME')}")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

print(f"2 HF_HOME={os.environ.get('HF_HOME')}")

if os.environ.get("LLAMA_DEBUG_CUDA", "0") == "1":
    os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
    os.environ.setdefault("TORCH_USE_CUDA_DSA", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


class Llama33_70BInstMCQHandler:
    """
        Unified handler for:
            - task_type="mcq"              -> returns a single letter A–F (or None)
            - task_type="answer_generation"-> returns generated text (or "")
            - task_type="dialogue_completion"-> returns generated text (one line, "ANSWER:" stripped)

        Input sample (dict):
            - For MCQ: question + opa/opb/opc/opd (+ optional ope/opf)
            - For Answer generation: question
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.3-70B-Instruct",
        cache_dir: str = HF_CACHE,
        offline: bool = True,
        load_in_4bit: bool = True,
    ):
        print(f"[Llama33_70BInstMCQ] Handler file: {__file__}")

        print(f"21 HF_HOME={os.environ.get('HF_HOME')}")

        print(f"[Llama33_70BInstMCQ] HF_HOME={os.environ.get('HF_HOME')}")
        print(f"[Llama33_70BInstMCQ] cache_dir={cache_dir}")
        print(f"[Llama33_70BInstMCQ] offline={offline}")
        print(f"[Llama33_70BInstMCQ] load_in_4bit={load_in_4bit}")

        self.model_name = model_name
        self.cache_dir = cache_dir or HF_CACHE
        self.offline = offline
        self.load_in_4bit = bool(load_in_4bit)

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
        print(f"[Llama33_70BInstMCQ] Available GPUs: {num_gpus}")
        if num_gpus == 0:
            raise RuntimeError("No CUDA GPUs available for Llama33_70BInstMCQ.")

        for i in range(num_gpus):
            print(
                f"  GPU {i}: {torch.cuda.get_device_name(i)} - "
                f"{torch.cuda.memory_allocated(i) / 1024**3:.2f} GB allocated"
            )

        # -------------------------
        # Load tokenizer
        # -------------------------
        print("[Llama33_70BInstMCQ] Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            local_files_only=self.local_files_only,
        )
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # --------------------------
        # Load model
        # --------------------------
        print("[Llama33_70BInstMCQ] Loading model...")

        quantization_config = None
        model_dtype = torch.bfloat16
        if self.load_in_4bit:
            print("[Llama33_70BInstMCQ] Setting up 4-bit quantization...")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            model_dtype = torch.float16

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=model_dtype,
            quantization_config=quantization_config,
            device_map="auto",
            cache_dir=self.cache_dir,
            local_files_only=self.local_files_only,
        )
        self.model.eval()

        print("[Llama33_70BInstMCQ] Model loaded.")
        if hasattr(self.model, "hf_device_map"):
            print("[Llama33_70BInstMCQ] Model device distribution:")
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

    @staticmethod
    def _build_dialogue_text(sample: dict) -> str:
        """
        Task 3 input: the doctor-patient dialogue with the final doctor turn
        removed. Accepts PascalCase "Dialogue" or snake_case fallbacks.
        """
        for key in ("Dialogue", "dialogue", "conversation", "context"):
            v = sample.get(key)
            if v:
                if isinstance(v, str):
                    return v.strip()
                if isinstance(v, list):
                    lines = []
                    for t in v:
                        if isinstance(t, dict):
                            role = str(t.get("role") or t.get("speaker") or "").strip()
                            text = str(t.get("text") or t.get("content") or "").strip()
                            if not text:
                                continue
                            lines.append(f"{role}: {text}" if role else text)
                        else:
                            s = str(t).strip()
                            if s:
                                lines.append(s)
                    return "\n".join(lines)
        return ""

    def prompt(
        self,
        sample: dict,
        instruction: str,
        max_tokens: int = 12,
        task_type: str = "mcq",
        **kwargs,
    ):
        """
        task_type:
          - "mcq": returns A-F or None
                    - "answer_generation": returns generated string (may be empty string)
                    - "dialogue_completion": returns generated string (may be empty string)
        """
        task_type = (task_type or "mcq").strip().lower()

        if task_type == "mcq":
            user_text = self._build_mcq_text(sample)
            if not user_text:
                print("[Llama33_70BInstMCQ] Empty stem/options; cannot build prompt.")
                return None
        elif task_type == "answer_generation":
            user_text = self._build_ansgen_text(sample)
            if not user_text:
                print("[Llama33_70BInstMCQ] Empty question; cannot build answer-generation prompt.")
                return ""
        elif task_type == "dialogue_completion":
            user_text = self._build_dialogue_text(sample)
            if not user_text:
                print("[Llama33_70BInstMCQ] Empty dialogue; cannot build dialogue-completion prompt.")
                return ""
        else:
            raise ValueError(
                f"Unsupported task_type={task_type}. Expected 'mcq', 'answer_generation', or 'dialogue_completion'."
            )

        system_prompt = instruction.strip()

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ]

        print("[Llama33_70BInstMCQ] MAX TOKENS:", max_tokens)
        try:
            torch.cuda.empty_cache()

            prompt_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            enc = self.tokenizer(prompt_text, return_tensors="pt")
            input_ids = enc["input_ids"].to(self.model.device)
            attention_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(self.model.device)

            with torch.inference_mode():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_tokens,
                    do_sample=False,  # greedy
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

        except Exception as e:
            print("[Llama33_70BInstMCQ] Error during generation:", e)
            return None if task_type == "mcq" else ""
        finally:
            torch.cuda.empty_cache()

        if outputs is None or outputs.shape[0] == 0:
            print("[Llama33_70BInstMCQ] Empty generation output.")
            return None if task_type == "mcq" else ""

        input_len = input_ids.shape[1]
        gen_ids = outputs[0][input_len:]
        if isinstance(gen_ids, torch.Tensor):
            gen_ids = gen_ids.detach().cpu().tolist()
        raw_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        if not raw_text:
            return None if task_type == "mcq" else ""

        if task_type == "answer_generation":
            one_line = raw_text.split("\n")[0].strip()
            print(f"[Llama33_70BInstMCQ] Answer-gen raw (one line): {repr(one_line[:300])}")
            return one_line

        if task_type == "dialogue_completion":
            one_line = raw_text.split("\n")[0].strip()
            m = re.match(r"^\s*ANSWER\s*[:=]\s*(.*)$", one_line, flags=re.IGNORECASE)
            if m:
                one_line = m.group(1).strip()
            print(f"[Llama33_70BInstMCQ] Dialogue-completion raw (one line): {repr(one_line[:300])}")
            return one_line

        print(f"[Llama33_70BInstMCQ] MCQ raw generated: {repr(raw_text)}")

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

        print("[Llama33_70BInstMCQ] Could not extract a clean letter.")
        return None

    def prompt_batch(
        self,
        samples,
        instruction: str,
        max_tokens: int = 12,
        task_type: str = "mcq",
        **kwargs,
    ):
        """
        Batched interface:
        - samples is a list of JSON records (dict)
        - for task_type="mcq": returns list of 'A'..'F' or None
        - for task_type="answer_generation": returns list of strings
        """
        task_type = (task_type or "mcq").strip().lower()

        if task_type in {"answer_generation", "dialogue_completion"}:
            return [
                self.prompt(s, instruction=instruction, max_tokens=max_tokens, task_type=task_type)
                for s in samples
            ]

        if task_type != "mcq":
            raise ValueError(
                f"Unsupported task_type={task_type}. Expected 'mcq', 'answer_generation', or 'dialogue_completion'."
            )

        system_prompt = instruction.strip()
        results = [None] * len(samples)

        prompt_texts = []
        valid_positions = []

        for i, sample in enumerate(samples):
            user_text = self._build_mcq_text(sample)
            if not user_text:
                continue

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ]

            prompt_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            prompt_texts.append(prompt_text)
            valid_positions.append(i)

        if not prompt_texts:
            return results

        try:
            enc = self.tokenizer(
                prompt_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            input_ids = enc["input_ids"].to(self.model.device)
            attention_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(self.model.device)

            with torch.inference_mode():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

        except Exception as e:
            print("[Llama33_70BInstMCQ] Error during batch generation:", e)
            return results

        if outputs is None or outputs.shape[0] == 0:
            print("[Llama33_70BInstMCQ] Empty batch generation output.")
            return results

        input_len = input_ids.shape[1]

        for row_idx, pos in enumerate(valid_positions):
            gen_ids = outputs[row_idx][input_len:]
            if isinstance(gen_ids, torch.Tensor):
                gen_ids = gen_ids.detach().cpu().tolist()
            raw_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

            if not raw_text:
                results[pos] = None
                continue

            upper = raw_text.upper()

            m = re.search(r"\bANSWER\s*[:=]\s*([A-F])\b", upper)
            if m:
                results[pos] = m.group(1)
                continue

            m = re.search(r"\b([A-F])\b", upper)
            if m:
                results[pos] = m.group(1)
                continue

            results[pos] = None

        return results