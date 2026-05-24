import os
import re

# SET HF_HOME **BEFORE** importing transformers/mistral libraries
# Otherwise they read the system default during import
HF_CACHE = "/scratch/mk8737/huggingface" # update this to a path on your system
os.environ.setdefault("HF_HOME", HF_CACHE)  # Only set if not already set

print(f"1 HF_HOME={os.environ.get('HF_HOME')}")

import torch
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from transformers import Mistral3ForConditionalGeneration

print(f"2 HF_HOME={os.environ.get('HF_HOME')}")

os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
os.environ.setdefault("TORCH_USE_CUDA_DSA", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


class MistralSmallMCQHandler:
    """
    Unified handler.
    - task_type="mcq": returns a single letter A-F (or None)
    - task_type="answer_generation": returns generated text (or "")
    - task_type="dialogue_completion": returns generated text (one line, "ANSWER:" stripped)
    """

    def __init__(
        self,
        model_name: str = "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
        cache_dir: str = HF_CACHE,
        offline: bool = True,
    ):
        print(f"[MistralSmallMCQ] Handler file: {__file__}")
        
        print(f"21 HF_HOME={os.environ.get('HF_HOME')}")

        print(f"[MistralSmallMCQ] HF_HOME={os.environ.get('HF_HOME')}")
        print(f"[MistralSmallMCQ] cache_dir={cache_dir}")
        print(f"[MistralSmallMCQ] offline={offline}")

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
        print(f"[MistralSmallMCQ] Available GPUs: {num_gpus}")
        if num_gpus == 0:
            raise RuntimeError("No CUDA GPUs available for MistralSmallMCQ.")

        for i in range(num_gpus):
            print(
                f"  GPU {i}: {torch.cuda.get_device_name(i)} - "
                f"{torch.cuda.memory_allocated(i) / 1024**3:.2f} GB allocated"
            )

        # -------------------------
        # Load tokenizer
        # -------------------------
        print("[MistralSmallMCQ] Loading tokenizer...")
        self.tokenizer = MistralTokenizer.from_hf_hub(self.model_name)

        # --------------------------
        # Load model
        # --------------------------
        print("[MistralSmallMCQ] Loading model...")
        self.model = Mistral3ForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            cache_dir=self.cache_dir,
            local_files_only=self.local_files_only,
        )

        print("[MistralSmallMCQ] Model loaded.")
        if hasattr(self.model, "hf_device_map"):
            print("[MistralSmallMCQ] Model device distribution:")
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
        Unified interface:
        - task_type="mcq": returns 'A'..'F' or None
        - task_type="answer_generation": returns generated one-line text or ""
        - task_type="dialogue_completion": returns generated one-line text or ""
        """
        task_type = (task_type or "mcq").strip().lower()
        if task_type == "mcq":
            user_text = self._build_mcq_text(sample)
            if not user_text:
                print("[MistralSmallMCQ] Empty stem/options; cannot build prompt.")
                return None
            default_value = None
        elif task_type == "answer_generation":
            user_text = self._build_ansgen_text(sample)
            if not user_text:
                print("[MistralSmallMCQ] Empty question; cannot build answer-generation prompt.")
                return ""
            default_value = ""
        elif task_type == "dialogue_completion":
            user_text = self._build_dialogue_text(sample)
            if not user_text:
                print("[MistralSmallMCQ] Empty dialogue; cannot build dialogue-completion prompt.")
                return ""
            default_value = ""
        else:
            raise ValueError(
                f"Unsupported task_type={task_type}. Expected 'mcq', 'answer_generation', or 'dialogue_completion'."
            )

        system_prompt = instruction.strip()

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [{"type": "text", "text": user_text}],
            },
        ]

        print("[MistralSmallMCQ] MAX TOKENS:", max_tokens)
        try:
            torch.cuda.empty_cache()

            req = ChatCompletionRequest(messages=messages)
            tokenized = self.tokenizer.encode_chat_completion(req)

            input_ids = torch.tensor(
                [tokenized.tokens],
                dtype=torch.long,
                device=self.model.device,
            )
            attention_mask = torch.ones_like(input_ids)

            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_tokens,
                    do_sample=False,  # greedy
                )

        except Exception as e:
            print("[MistralSmallMCQ] Error during generation:", e)
            return default_value
        finally:
            torch.cuda.empty_cache()

        if outputs is None or outputs.shape[0] == 0:
            print("[MistralSmallMCQ] Empty generation output.")
            return default_value

        input_len = len(tokenized.tokens)
        gen_ids = outputs[0][input_len:]
        raw_text = self.tokenizer.decode(gen_ids).strip()

        if not raw_text:
            return default_value

        if task_type == "answer_generation":
            one_line = raw_text.split("\n")[0].strip()
            print(f"[MistralSmallMCQ] Answer-gen raw (one line): {repr(one_line[:200])}")
            return one_line

        if task_type == "dialogue_completion":
            one_line = raw_text.split("\n")[0].strip()
            m = re.match(r"^\s*ANSWER\s*[:=]\s*(.*)$", one_line, flags=re.IGNORECASE)
            if m:
                one_line = m.group(1).strip()
            print(f"[MistralSmallMCQ] Dialogue-completion raw (one line): {repr(one_line[:200])}")
            return one_line

        print(f"[MistralSmallMCQ] MCQ raw generated: {repr(raw_text)}")

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

        print("[MistralSmallMCQ] Could not extract a clean letter.")
        return None

    def prompt_batch(
        self,
        samples,
        instruction: str,
        max_tokens: int = 12,
        task_type: str = "mcq",
        **kwargs,
    ):
        return [
            self.prompt(
                s,
                instruction=instruction,
                max_tokens=max_tokens,
                task_type=task_type,
                **kwargs,
            )
            for s in samples
        ]
