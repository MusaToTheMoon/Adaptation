import os
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

HF_CACHE = "/scratch/mk8737/huggingface"
os.environ["HF_HOME"] = HF_CACHE

# Keep your debugging defaults consistent across handlers
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
os.environ.setdefault("TORCH_USE_CUDA_DSA", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


class Jais2ChatMCQHandler:
    """
    MCQ-only handler for inceptionai/Jais-2-8B-Chat.

    Contract (matches your eval harness):
      - __init__(model_name, cache_dir, offline)
      - prompt(sample: dict, instruction: str, max_tokens: int=12) -> str|None
        where return is a single letter A-F.

    Notes:
      - Prefers tokenizer.apply_chat_template(..., add_generation_prompt=True)
      - Drops token_type_ids (as per model card example)
      - Works with options opa/opb/opc/opd/(ope/opf optional)
    """

    def __init__(
        self,
        model_name: str = "inceptionai/Jais-2-8B-Chat",
        cache_dir: str = HF_CACHE,
        offline: bool = True,
        torch_dtype=torch.bfloat16,
        use_fast: bool = True,
    ):
        print(f"[Jais2ChatMCQ] Handler file: {__file__}")
        print(f"[Jais2ChatMCQ] HF_HOME={os.environ.get('HF_HOME')}")
        print(f"[Jais2ChatMCQ] cache_dir={cache_dir}")
        print(f"[Jais2ChatMCQ] offline={offline}")

        self.model_name = model_name
        self.cache_dir = cache_dir or HF_CACHE
        self.offline = offline
        self.torch_dtype = torch_dtype

        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

        # Offline toggle (same pattern as your other handlers)
        self.local_files_only = bool(self.offline)
        if self.offline:
            os.environ["HF_HUB_OFFLINE"] = "1"
        else:
            os.environ.pop("HF_HUB_OFFLINE", None)

        num_gpus = torch.cuda.device_count()
        print(f"[Jais2ChatMCQ] Available GPUs: {num_gpus}")
        if num_gpus == 0:
            raise RuntimeError("No CUDA GPUs available for Jais-2-8B-Chat.")

        print("[Jais2ChatMCQ] Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            local_files_only=self.local_files_only,
            use_fast=use_fast,
        )

        # Ensure pad token exists
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.has_chat_template = bool(getattr(self.tokenizer, "chat_template", None))
        print(f"[Jais2ChatMCQ] tokenizer.chat_template available? {self.has_chat_template}")

        print("[Jais2ChatMCQ] Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.torch_dtype,
            device_map="auto",
            cache_dir=self.cache_dir,
            local_files_only=self.local_files_only,
        )
        print("[Jais2ChatMCQ] Model loaded.")

    # -----------------------------
    # Prompt building (MCQ)
    # -----------------------------
    @staticmethod
    def _build_mcq_text(sample: dict) -> str:
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
            if txt:
                lines.append(f"{letter}) {txt}")

        if not lines:
            return stem

        return stem + "\n\n" + "\n".join(lines)

    @staticmethod
    def _build_ansgen_text(sample: dict) -> str:
        question = (sample.get("question") or "").strip()
        if not question:
            return ""
        return question

    @staticmethod
    def _system_text(instruction: str) -> str:
        instruction = (instruction or "").strip()
        base = (
            "You are a medical assistant. Answer the multiple-choice question.\n"
            "Return only one letter (A-F) in the format: Answer: X."
        )
        if instruction:
            base += f"\n\nINSTRUCTION: {instruction}"
        return base

    def _plain_prompt(self, instruction: str, user_text: str) -> str:
        # Fallback for tokenizers without chat_template
        return (
            f"{self._system_text(instruction)}\n\n"
            f"QUESTION:\n{user_text}\n\n"
            "Answer:"
        )

    @staticmethod
    def _plain_ansgen_prompt(instruction: str, user_text: str) -> str:
        return (
            f"{(instruction or '').strip()}\n\n"
            f"QUESTION:\n{user_text}\n\n"
            "Answer in one concise line."
        )

    # -----------------------------
    # Main API for eval harness
    # -----------------------------
    def prompt(
        self,
        sample: dict,
        instruction: str,
        max_tokens: int = 12,
        task_type: str = "mcq",
        **kwargs,
    ):
        task_type = (task_type or "mcq").strip().lower()

        if task_type == "mcq":
            user_text = self._build_mcq_text(sample)
            if not user_text:
                print("[Jais2ChatMCQ] Empty stem/options; cannot build prompt.")
                return None
            system_prompt = self._system_text(instruction)
        elif task_type == "answer_generation":
            user_text = self._build_ansgen_text(sample)
            if not user_text:
                print("[Jais2ChatMCQ] Empty question; cannot build answer-generation prompt.")
                return ""
            system_prompt = (instruction or "").strip()
        else:
            raise ValueError(f"Unsupported task_type={task_type}. Expected 'mcq' or 'answer_generation'.")

        try:
            torch.cuda.empty_cache()

            if self.has_chat_template:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_text},
                ]
                inputs = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                )
            else:
                if task_type == "mcq":
                    prompt_str = self._plain_prompt(instruction, user_text)
                else:
                    prompt_str = self._plain_ansgen_prompt(instruction, user_text)
                inputs = self.tokenizer(
                    prompt_str,
                    return_tensors="pt",
                    add_special_tokens=True,
                )

            # Model card example removes token_type_ids
            if isinstance(inputs, dict) and "token_type_ids" in inputs:
                inputs.pop("token_type_ids", None)

            # Move tensors to model device
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            input_len = inputs["input_ids"].shape[-1]

            with torch.inference_mode():
                generation = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            gen_ids = generation[0][input_len:]
            raw_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        except Exception as e:
            print("[Jais2ChatMCQ] Error during generation:", e)
            return None if task_type == "mcq" else ""
        finally:
            torch.cuda.empty_cache()

        if not raw_text:
            return None if task_type == "mcq" else ""

        if task_type == "answer_generation":
            one_line = raw_text.split("\n")[0].strip()
            print(f"[Jais2ChatMCQ] Answer-gen raw (one line): {repr(one_line[:300])}")
            return one_line

        print(f"[Jais2ChatMCQ] MCQ raw generated: {repr(raw_text)}")
        upper = raw_text.upper()

        # Prefer strict format: ANSWER: X
        m = re.search(r"\bANSWER\s*[:=]\s*([A-F])\b", upper)
        if m:
            return m.group(1)

        # Common Arabic equivalent sometimes shows up
        m = re.search(r"\b(?:الإجابة|الاجابة)\s*[:=]\s*([A-F])\b", raw_text)
        if m:
            return m.group(1).upper()

        # Fallback: first standalone A-F
        m = re.search(r"\b([A-F])\b", upper)
        if m:
            return m.group(1)

        print("[Jais2ChatMCQ] Could not extract a clean letter.")
        return None