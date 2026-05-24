import os
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

HF_CACHE = "/scratch/mk8737/huggingface"
os.environ["HF_HOME"] = HF_CACHE

os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
os.environ.setdefault("TORCH_USE_CUDA_DSA", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


class Fanar19BMCQHandler:
    """
    Unified handler for QCRI/Fanar-1-9B.
    - task_type="mcq": returns a single letter A-F (or None)
    - task_type="answer_generation": returns generated text (or "")
    - task_type="dialogue_completion": returns generated text (one line, "ANSWER:" stripped)
    """

    def __init__(
        self,
        model_name: str = "QCRI/Fanar-1-9B",
        cache_dir: str = HF_CACHE,
        offline: bool = True,
        torch_dtype=torch.bfloat16,
    ):
        print(f"[Fanar19BMCQ] Handler file: {__file__}")
        print(f"[Fanar19BMCQ] HF_HOME={os.environ.get('HF_HOME')}")
        print(f"[Fanar19BMCQ] cache_dir={cache_dir}")
        print(f"[Fanar19BMCQ] offline={offline}")

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
        print(f"[Fanar19BMCQ] Available GPUs: {num_gpus}")
        if num_gpus == 0:
            raise RuntimeError("No CUDA GPUs available for Fanar19BMCQ.")

        for i in range(num_gpus):
            print(
                f"  GPU {i}: {torch.cuda.get_device_name(i)} - "
                f"{torch.cuda.memory_allocated(i) / 1024**3:.2f} GB allocated"
            )

        # -------------------------
        # Load tokenizer
        # -------------------------
        print("[Fanar19BMCQ] Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            local_files_only=self.local_files_only,
            use_fast=True,
        )

        # Ensure pad_token exists for batching/generation
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        # --------------------------
        # Load model
        # --------------------------
        print("[Fanar19BMCQ] Loading model...")

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            torch_dtype=torch_dtype,
            cache_dir=self.cache_dir,
            local_files_only=self.local_files_only,
        )

        print("[Fanar19BMCQ] Model loaded.")
        if hasattr(self.model, "hf_device_map"):
            print("[Fanar19BMCQ] Model device distribution:")
            for layer, dev in self.model.hf_device_map.items():
                print(f"  {layer}: {dev}")

        for i in range(num_gpus):
            alloc = torch.cuda.memory_allocated(i) / 1024**3
            reserv = torch.cuda.memory_reserved(i) / 1024**3
            print(f"  GPU {i} after load: {alloc:.2f}GB allocated, {reserv:.2f}GB reserved")

    @staticmethod
    def _build_mcq_text(sample: dict) -> str:
        """
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

    @staticmethod
    def _build_ansgen_text(sample: dict) -> str:
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

    @staticmethod
    def _extract_mcq_letter(raw_text: str):
        upper = (raw_text or "").upper()

        # Prefer strict format: ANSWER: X
        m = re.search(r"\bANSWER\s*[:=]\s*([A-F])\b", upper)
        if m:
            return m.group(1)

        # Fallback: first standalone A-F
        m = re.search(r"\b([A-F])\b", upper)
        if m:
            return m.group(1)

        return None

    @staticmethod
    def _resolve_max_tokens(task_type: str, max_tokens: int) -> int:
        requested = int(max_tokens or 0)
        if task_type in {"answer_generation", "dialogue_completion"}:
            # Preserve configured task2 token budget from config (default 256).
            return requested if requested > 0 else 256
        requested = requested if requested > 0 else 12
        return max(requested, 8)

    def _build_full_prompt(self, system_prompt: str, user_text: str, task_type: str) -> str:
        if task_type == "mcq":
            return (
                f"{system_prompt}\n\n"
                f"{user_text}\n\n"
                "Return ONLY one letter among A, B, C, D, E, F.\n\n"
                "ANSWER:"
            )
        return (
            f"{system_prompt}\n\n"
            f"{user_text}\n\n"
            "Answer in one concise line.\n"
        )

    def prompt(
        self,
        sample: dict,
        instruction: str,
        max_tokens: int = 12,
        temperature: float = 0.0,
        task_type: str = "mcq",
        **kwargs,
    ):
        """
        MCQ-only interface:
        - sample: ONE JSON record (dict)
        - returns: 'A'..'F' or None

        Behavior:
        - If temperature <= 0: greedy decode (like your MedGemma do_sample=False)
        - Else: sampling with temperature
        """
        task_type = (task_type or "mcq").strip().lower()

        if task_type == "mcq":
            user_text = self._build_mcq_text(sample)
            if not user_text:
                print("[Fanar19BMCQ] Empty stem/options; cannot build prompt.")
                return None
        elif task_type == "answer_generation":
            user_text = self._build_ansgen_text(sample)
            if not user_text:
                print("[Fanar19BMCQ] Empty question; cannot build answer-generation prompt.")
                return ""
        elif task_type == "dialogue_completion":
            user_text = self._build_dialogue_text(sample)
            if not user_text:
                print("[Fanar19BMCQ] Empty dialogue; cannot build dialogue-completion prompt.")
                return ""
        else:
            raise ValueError(
                f"Unsupported task_type={task_type}. Expected 'mcq', 'answer_generation', or 'dialogue_completion'."
            )

        system_prompt = (instruction or "").strip()
        full_prompt = self._build_full_prompt(system_prompt, user_text, task_type)
        max_tokens = self._resolve_max_tokens(task_type, max_tokens)

        print("[Fanar19BMCQ] MAX TOKENS:", max_tokens)
        print("[Fanar19BMCQ] TEMPERATURE:", temperature)

        try:
            # Fanar model card suggests return_token_type_ids=False
            inputs = self.tokenizer(
                full_prompt,
                return_tensors="pt",
                return_token_type_ids=False,
            )

            # Put inputs on the "main" device (works in most device_map="auto" cases)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            input_len = inputs["input_ids"].shape[-1]

            do_sample = bool(temperature and temperature > 0)
            gen_kwargs = dict(
                max_new_tokens=max_tokens,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            if do_sample:
                gen_kwargs["temperature"] = float(temperature)

            with torch.inference_mode():
                generation = self.model.generate(
                    **inputs,
                    **gen_kwargs,
                )

            gen_ids = generation[0][input_len:]
            raw_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        except Exception as e:
            print("[Fanar19BMCQ] Error during generation:", e)
            return None if task_type == "mcq" else ""

        if not raw_text:
            return None if task_type == "mcq" else ""

        if task_type == "answer_generation":
            one_line = raw_text.split("\n")[0].strip()
            print(f"[Fanar19BMCQ] Answer-gen raw (one line): {repr(one_line[:300])}")
            return one_line

        if task_type == "dialogue_completion":
            one_line = raw_text.split("\n")[0].strip()
            m = re.match(r"^\s*ANSWER\s*[:=]\s*(.*)$", one_line, flags=re.IGNORECASE)
            if m:
                one_line = m.group(1).strip()
            print(f"[Fanar19BMCQ] Dialogue-completion raw (one line): {repr(one_line[:300])}")
            return one_line

        print(f"[Fanar19BMCQ] MCQ raw generated: {repr(raw_text)}")

        answer = self._extract_mcq_letter(raw_text)
        if answer:
            return answer

        print("[Fanar19BMCQ] Could not extract a clean letter.")
        return None

    def prompt_batch(
        self,
        samples,
        instruction: str,
        max_tokens: int = 12,
        temperature: float = 0.0,
        task_type: str = "mcq",
        **kwargs,
    ):
        task_type = (task_type or "mcq").strip().lower()
        if task_type not in {"mcq", "answer_generation", "dialogue_completion"}:
            raise ValueError(
                f"Unsupported task_type={task_type}. Expected 'mcq', 'answer_generation', or 'dialogue_completion'."
            )

        if not samples:
            return []

        max_tokens = self._resolve_max_tokens(task_type, max_tokens)
        system_prompt = (instruction or "").strip()
        do_sample = bool(temperature and temperature > 0)

        results = [None if task_type == "mcq" else "" for _ in samples]
        prompts = []
        valid_positions = []

        for i, sample in enumerate(samples):
            if task_type == "mcq":
                user_text = self._build_mcq_text(sample)
            elif task_type == "answer_generation":
                user_text = self._build_ansgen_text(sample)
            else:
                user_text = self._build_dialogue_text(sample)

            if not user_text:
                continue

            prompts.append(self._build_full_prompt(system_prompt, user_text, task_type))
            valid_positions.append(i)

        if not prompts:
            return results

        print(f"[Fanar19BMCQ] prompt_batch size={len(prompts)} MAX TOKENS: {max_tokens}")

        try:
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                return_token_type_ids=False,
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            attention_mask = inputs.get("attention_mask")
            input_lengths = attention_mask.sum(dim=1) if attention_mask is not None else None

            gen_kwargs = dict(
                max_new_tokens=max_tokens,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            if do_sample:
                gen_kwargs["temperature"] = float(temperature)

            with torch.inference_mode():
                generations = self.model.generate(
                    **inputs,
                    **gen_kwargs,
                )

        except Exception as e:
            print("[Fanar19BMCQ] Error during batch generation:", e)
            return results

        for row_idx, pos in enumerate(valid_positions):
            if input_lengths is None:
                prompt_len = inputs["input_ids"].shape[1]
            else:
                prompt_len = int(input_lengths[row_idx].item())

            gen_ids = generations[row_idx][prompt_len:]
            raw_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

            if not raw_text:
                continue

            if task_type == "answer_generation":
                results[pos] = raw_text.split("\n")[0].strip()
                continue

            if task_type == "dialogue_completion":
                one_line = raw_text.split("\n")[0].strip()
                m = re.match(r"^\s*ANSWER\s*[:=]\s*(.*)$", one_line, flags=re.IGNORECASE)
                if m:
                    one_line = m.group(1).strip()
                results[pos] = one_line
                continue

            results[pos] = self._extract_mcq_letter(raw_text)

        return results