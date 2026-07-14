"""
AutoCAP baseline — MedGemma backend.

Same three-round pipeline as models/autocap.py (Zhang et al., ACL 2024
Findings, "AutoCAP: Towards Automatic Cross-lingual Alignment Planning for
Zero-shot Chain-of-Thought"); see that module's docstring for the full
paper description. This variant swaps the Mistral-specific tokenizer/model
loading (mistral_common + Mistral3ForConditionalGeneration) for the
AutoTokenizer/AutoModelForCausalLM + chat-template conventions used by
models/medgemma.py, since MedGemma is not a Mistral-family checkpoint.
"""

import json
import os
import re
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HF_CACHE = "/scratch/mk8737/huggingface"
os.environ.setdefault("HF_HOME", HF_CACHE)

os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
os.environ.setdefault("TORCH_USE_CUDA_DSA", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


# ---------------------------------------------------------------------------
# Per-language metadata used by ALSP (mirrors the paper's L_info block).
# ---------------------------------------------------------------------------
_LANGUAGE_INFO: Dict[str, Dict[str, str]] = {
    "Arabic":   {"family": "Afro-Asiatic",  "branch": "Semitic",   "pretrain_pct": "~5%"},
    "English":  {"family": "Indo-European", "branch": "Germanic",  "pretrain_pct": "~46%"},
    "French":   {"family": "Indo-European", "branch": "Romance",   "pretrain_pct": "~8%"},
    "German":   {"family": "Indo-European", "branch": "Germanic",  "pretrain_pct": "~6%"},
    "Spanish":  {"family": "Indo-European", "branch": "Romance",   "pretrain_pct": "~5%"},
    "Chinese":  {"family": "Sino-Tibetan",  "branch": "Sinitic",   "pretrain_pct": "~4%"},
    "Japanese": {"family": "Japonic",       "branch": "Japanese",  "pretrain_pct": "~3%"},
    "Russian":  {"family": "Indo-European", "branch": "Slavic",    "pretrain_pct": "~6%"},
}


def _lang_info_line(lang: str) -> str:
    """Return a single formatted metadata line for one language."""
    info = _LANGUAGE_INFO.get(lang)
    if info is None:
        return f"- {lang}: (no metadata available)"
    return (
        f"- {lang}: Family: {info['family']}; "
        f"Branch: {info['branch']}; "
        f"Pre-training data: {info['pretrain_pct']}"
    )


class AutoCAPMedGemmaHandler:
    """
    AutoCAP handler (Zhang et al., ACL 2024 Findings), MedGemma backend,
    adapted for three task types:
      - task_type="mcq"                 -> returns letter A-F or None
      - task_type="answer_generation"   -> returns free-text answer (Arabic) or ""
      - task_type="dialogue_completion" -> returns doctor turn (one line, Arabic) or ""
      - prompt_batch(samples, ...)      -> list of the above

    If return_debug=True, prompt() returns a dict:
      {
        "final_prediction":   <letter or str>,
        "selected_languages": [...],
        "weights":            {lang: float},
        "language_answers":   {lang: extracted_answer},
        "language_reasoning": {lang: full_cot_text},
      }
    """

    def __init__(
        self,
        model_name: str = "google/medgemma-27b-text-it",
        cache_dir: str = HF_CACHE,
        offline: bool = True,
        candidate_languages: Optional[List[str]] = None,
        top_k_languages: int = 3,
        selection_max_tokens: int = 128,
        weight_max_tokens: int = 128,
        cot_max_tokens: int = 512,
        reasoning_max_tokens: int = None,  # legacy alias from old configs; mapped to cot_max_tokens
        do_sample: bool = False,
    ):
        print(f"[AutoCAPMedGemma] Handler file: {__file__}")
        print(f"[AutoCAPMedGemma] HF_HOME={os.environ.get('HF_HOME')}")
        print(f"[AutoCAPMedGemma] cache_dir={cache_dir}")
        print(f"[AutoCAPMedGemma] offline={offline}")

        self.model_name = model_name
        self.cache_dir = cache_dir or HF_CACHE
        self.offline = bool(offline)
        self.top_k_languages = int(top_k_languages)
        self.selection_max_tokens = int(selection_max_tokens)
        self.weight_max_tokens = int(weight_max_tokens)
        if reasoning_max_tokens is not None:
            self.cot_max_tokens = int(reasoning_max_tokens)
        else:
            self.cot_max_tokens = int(cot_max_tokens)
        print(f"[AutoCAPMedGemma] cot_max_tokens={self.cot_max_tokens} "
              f"({'from reasoning_max_tokens alias' if reasoning_max_tokens is not None else 'default'})")
        self.do_sample = bool(do_sample)

        self.candidate_languages = candidate_languages or ["Arabic", "English", "French"]

        if self.top_k_languages < 1:
            raise ValueError("top_k_languages must be >= 1")
        if self.top_k_languages > len(self.candidate_languages):
            raise ValueError("top_k_languages cannot exceed number of candidate languages")

        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

        self.local_files_only = bool(self.offline)
        if self.offline:
            os.environ["HF_HUB_OFFLINE"] = "1"
        else:
            os.environ.pop("HF_HUB_OFFLINE", None)

        num_gpus = torch.cuda.device_count()
        print(f"[AutoCAPMedGemma] Available GPUs: {num_gpus}")
        if num_gpus == 0:
            raise RuntimeError("No CUDA GPUs available for AutoCAPMedGemmaHandler.")
        for i in range(num_gpus):
            print(
                f"  GPU {i}: {torch.cuda.get_device_name(i)} - "
                f"{torch.cuda.memory_allocated(i) / 1024**3:.2f} GB allocated"
            )

        print("[AutoCAPMedGemma] Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            local_files_only=self.local_files_only,
            use_fast=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Left-padding is required so every batch item's last real token
        # lines up at the same position for autoregressive generation.
        self.tokenizer.padding_side = "left"

        print("[AutoCAPMedGemma] Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            cache_dir=self.cache_dir,
            local_files_only=self.local_files_only,
        )
        print("[AutoCAPMedGemma] Model loaded.")
        if hasattr(self.model, "hf_device_map"):
            print("[AutoCAPMedGemma] Model device distribution:")
            for layer, dev in self.model.hf_device_map.items():
                print(f"  {layer}: {dev}")
        for i in range(num_gpus):
            alloc = torch.cuda.memory_allocated(i) / 1024**3
            reserv = torch.cuda.memory_reserved(i) / 1024**3
            print(f"  GPU {i} after load: {alloc:.2f}GB allocated, {reserv:.2f}GB reserved")

    # -----------------------------------------------------------------------
    # Input text builders
    # -----------------------------------------------------------------------
    @staticmethod
    def _build_mcq_text(sample: dict) -> str:
        stem = (sample.get("question") or "").strip()
        if not stem:
            return ""
        option_map = {
            "A": sample.get("opa"), "B": sample.get("opb"),
            "C": sample.get("opc"), "D": sample.get("opd"),
            "E": sample.get("ope"), "F": sample.get("opf"),
        }
        lines = []
        for letter in ["A", "B", "C", "D", "E", "F"]:
            txt = option_map.get(letter)
            if txt is None:
                continue
            txt = str(txt).strip()
            if txt:
                lines.append(f"{letter}) {txt}")
        return stem + ("\n\n" + "\n".join(lines) if lines else "")

    @staticmethod
    def _build_ansgen_text(sample: dict) -> str:
        return (sample.get("question") or "").strip()

    @staticmethod
    def _build_dialogue_text(sample: dict) -> str:
        """
        Build the doctor-patient dialogue with the final doctor turn missing.
        Reads from sample["Dialogue"] (PascalCase, the actual Task-3 field name)
        first, then falls back to snake_case and other common keys.
        """
        raw = (
            sample.get("Dialogue")
            or sample.get("dialogue")
            or sample.get("conversation")
            or sample.get("context")
            or sample.get("turns")
            or sample.get("question")
        )
        if raw is None:
            return ""
        if isinstance(raw, str):
            return raw.strip()
        if isinstance(raw, list) and len(raw) > 0:
            if bool(sample.get("last_turn_included", False)):
                raw = raw[:-1]
            lines = []
            for t in raw:
                if isinstance(t, dict):
                    role = str(t.get("role") or t.get("speaker") or "").strip()
                    text = str(
                        t.get("text") or t.get("content") or t.get("utterance") or ""
                    ).strip()
                    if not text:
                        continue
                    lines.append(f"{role}: {text}" if role else text)
                else:
                    s = str(t).strip()
                    if s:
                        lines.append(s)
            return "\n".join(lines)
        return ""

    # -----------------------------------------------------------------------
    # Output cleaners / CoT answer extractors
    # -----------------------------------------------------------------------
    @staticmethod
    def _extract_letter_from_cot(text: str) -> Optional[str]:
        if not text:
            return None
        upper = text.strip().upper()
        for pat in [
            r"\bFINAL\s*ANSWER\s*[:=]\s*([A-F])\b",
            r"\bTHE\s+ANSWER\s+IS\s*[:=]?\s*([A-F])\b",
            r"\bANSWER\s*[:=]\s*([A-F])\b",
            r"\bCORRECT\s+ANSWER\s*[:=]?\s*([A-F])\b",
            r"\bOPTION\s*([A-F])\b",
        ]:
            m = re.search(pat, upper)
            if m:
                return m.group(1)
        matches = list(re.finditer(r"\b([A-F])\b", upper))
        if matches:
            return matches[-1].group(1)
        return None

    @staticmethod
    def _extract_ansgen_from_cot(text: str) -> str:
        if not text:
            return ""
        for pat in [
            r"(?:Final\s+answer\s+in\s+Arabic|الإجابة\s+النهائية|الجواب\s+النهائي)\s*[:=]\s*(.+?)(?:\n|$)",
            r"Final\s+[Aa]nswer\s*[:=]\s*(.+?)(?:\n|$)",
            r"ANSWER\s*[:=]\s*(.+?)(?:\n|$)",
        ]:
            m = re.search(pat, text, flags=re.IGNORECASE)
            if m:
                ans = m.group(1).strip()
                if ans:
                    return ans
        lines = [ln.strip() for ln in text.strip().split("\n") if ln.strip()]
        if lines:
            last = lines[-1]
            m2 = re.match(r"^\s*(?:ANSWER|Answer)\s*[:=]\s*(.+)$", last)
            if m2:
                last = m2.group(1).strip()
            if len(last) >= 2 and last[0] == last[-1] and last[0] in ('"', "'", "“", "”"):
                last = last[1:-1].strip()
            return last
        return ""

    @staticmethod
    def _extract_dialogue_from_cot(text: str) -> str:
        if not text:
            return ""
        m = re.search(r"ANSWER\s*[:=]\s*(.+?)(?:\n|$)", text, flags=re.IGNORECASE)
        if m:
            ans = m.group(1).strip()
            if ans:
                return ans
        lines = [ln.strip() for ln in text.strip().split("\n") if ln.strip()]
        if lines:
            last = lines[-1]
            m2 = re.match(r"^\s*ANSWER\s*[:=]\s*(.+)$", last, flags=re.IGNORECASE)
            if m2:
                last = m2.group(1).strip()
            return last
        return ""

    # -----------------------------------------------------------------------
    # Misc helpers
    # -----------------------------------------------------------------------
    @staticmethod
    def _safe_json_loads(text: str):
        if not text:
            return None
        s = text.strip()
        try:
            return json.loads(s)
        except Exception:
            pass
        m = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", s, flags=re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass
        m = re.search(r"(\{.*\})", s, flags=re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass
        m = re.search(r"(\[.*\])", s, flags=re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass
        return None

    @staticmethod
    def _normalize_weights(
        weights: Dict[str, float], selected_languages: List[str]
    ) -> Dict[str, float]:
        cleaned = {}
        for lang in selected_languages:
            val = weights.get(lang, 0.0)
            try:
                val = float(val)
            except Exception:
                val = 0.0
            cleaned[lang] = max(val, 0.0)
        total = sum(cleaned.values())
        if total <= 0:
            uniform = 1.0 / max(len(selected_languages), 1)
            return {lang: uniform for lang in selected_languages}
        return {lang: cleaned[lang] / total for lang in selected_languages}

    # -----------------------------------------------------------------------
    # Core generation
    # -----------------------------------------------------------------------
    def _generate(self, system_prompt: str, user_text: str, max_tokens: int) -> str:
        messages = [
            {"role": "system", "content": (system_prompt or "").strip()},
            {"role": "user", "content": user_text},
        ]
        try:
            torch.cuda.empty_cache()
            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            input_len = inputs["input_ids"].shape[-1]
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=self.do_sample,
                )
            gen_ids = outputs[0][input_len:]
            return self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip() or ""
        finally:
            torch.cuda.empty_cache()

    def _generate_batch(
        self, prompts: List[Tuple[str, str]], max_tokens: int
    ) -> List[str]:
        """
        Run multiple (system_prompt, user_text) pairs through the model in ONE
        batched GPU forward pass. See models/autocap.py._generate_batch for the
        rationale (the per-language CoT calls are the bottleneck).
        """
        if not prompts:
            return []
        if len(prompts) == 1:
            return [self._generate(prompts[0][0], prompts[0][1], max_tokens)]

        try:
            torch.cuda.empty_cache()

            token_id_lists: List[List[int]] = []
            for system_prompt, user_text in prompts:
                messages = [
                    {"role": "system", "content": (system_prompt or "").strip()},
                    {"role": "user", "content": user_text},
                ]
                ids = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=False,
                    return_tensors=None,
                )
                token_id_lists.append(list(ids))

            pad_id = self.tokenizer.pad_token_id

            max_input_len = max(len(ids) for ids in token_id_lists)
            padded_ids: List[List[int]] = []
            attn_masks: List[List[int]] = []
            for ids in token_id_lists:
                pad_len = max_input_len - len(ids)
                padded_ids.append([pad_id] * pad_len + ids)
                attn_masks.append([0] * pad_len + [1] * len(ids))

            input_ids = torch.tensor(
                padded_ids, dtype=torch.long, device=self.model.device
            )
            attention_mask = torch.tensor(
                attn_masks, dtype=torch.long, device=self.model.device
            )

            with torch.inference_mode():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_tokens,
                    do_sample=self.do_sample,
                    pad_token_id=pad_id,
                )

            results: List[str] = []
            for i in range(len(prompts)):
                gen_ids = outputs[i][max_input_len:].tolist()
                # Truncate at the first pad/EOS — other sequences in the batch
                # may have continued generating padding past this item's EOS.
                try:
                    eos_pos = gen_ids.index(pad_id)
                    gen_ids = gen_ids[:eos_pos]
                except ValueError:
                    pass
                raw = (
                    self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                    if gen_ids
                    else ""
                )
                results.append(raw)

            return results

        except torch.cuda.OutOfMemoryError:
            print(
                f"[AutoCAPMedGemma] _generate_batch: OOM with batch_size={len(prompts)}, "
                "falling back to sequential _generate() calls."
            )
            torch.cuda.empty_cache()
            return [self._generate(sp, ut, max_tokens) for sp, ut in prompts]

        finally:
            torch.cuda.empty_cache()

    # -----------------------------------------------------------------------
    # Prompt builders — Round 1: ALSP  (generalized across task types)
    # -----------------------------------------------------------------------
    def _build_language_selection_prompts(
        self, content_text: str, task_type: str = "mcq"
    ) -> Tuple[str, str]:
        task_desc = {
            "mcq":                "solving an Arabic medical multiple-choice question",
            "answer_generation":  "answering an Arabic open-ended medical question",
            "dialogue_completion":"generating the doctor's next response in an Arabic medical dialogue",
        }.get(task_type, "solving an Arabic medical task")

        system_prompt = (
            "You are an expert planner for multilingual medical reasoning.\n"
            f"Your task is to automatically select {self.top_k_languages} languages "
            f"optimal for cross-lingual reasoning on a given Arabic sample "
            f"({task_desc}).\n"
            "Prioritize expected reasoning accuracy over fluency or style.\n"
            "Return valid JSON only. Do not include explanations."
        )

        lang_info_block = "\n".join(
            _lang_info_line(lang) for lang in self.candidate_languages
        )
        candidates_str = ", ".join(self.candidate_languages)

        user_prompt = f"""Your task is to select exactly {self.top_k_languages} languages optimal \
for cross-lingual reasoning on the following Arabic input.

Choose languages based on:
  1. expected medical reasoning accuracy in that language,
  2. ability to preserve the meaning of the Arabic input,
  3. estimated model strength in that language (guided by pre-training data share),
  4. diversity of reliable reasoning paths.

Language information:
{lang_info_block}

Candidate languages: {candidates_str}

Return exactly this JSON format (no other text):
{{"languages": ["English", "Arabic", "French"]}}

Input:
{content_text}"""

        return system_prompt, user_prompt

    # -----------------------------------------------------------------------
    # Prompt builders — Round 2: AWAP  (generalized across task types)
    # -----------------------------------------------------------------------
    def _build_weight_assignment_prompts(
        self, content_text: str, selected_languages: List[str], task_type: str = "mcq"
    ) -> Tuple[str, str]:
        task_desc = {
            "mcq":                "an Arabic medical multiple-choice question",
            "answer_generation":  "an Arabic open-ended medical question",
            "dialogue_completion":"an Arabic medical dialogue requiring a doctor's response",
        }.get(task_type, "an Arabic medical task")

        system_prompt = (
            "You are an expert planner for multilingual medical reasoning.\n"
            f"After language selection, please automatically assign an alignment "
            f"weight score for multilingual reasoning aggregation on {task_desc}.\n"
            "Weight range: [0, 1]. Higher weight = more reliable reasoning path.\n"
            "Return valid JSON only. Do not include explanations."
        )

        lang_str = ", ".join(selected_languages)
        weights_stub = ", ".join(f'"{lang}": 0.0' for lang in selected_languages)

        user_prompt = f"""Assign a confidence weight [0, 1] to each selected reasoning language.
A higher weight means reasoning in that language is more likely to produce the correct answer
for the following Arabic input.

Selected languages: {lang_str}

Return exactly this JSON format (no other text):
{{"weights": {{{weights_stub}}}}}

Input:
{content_text}"""

        return system_prompt, user_prompt

    # -----------------------------------------------------------------------
    # Prompt builders — Round 3: CoT reasoning  (one per task type)
    # -----------------------------------------------------------------------
    @staticmethod
    def _build_mcq_cot_prompts(
        mcq_text: str, reasoning_language: str
    ) -> Tuple[str, str]:
        system_prompt = (
            f"You are a highly careful medical expert. "
            f"Act as an expert in medical reasoning in {reasoning_language}.\n"
            "The question is written in Arabic. "
            f"Reason step-by-step in {reasoning_language} and end your response with:\n"
            "Answer: [single uppercase letter A, B, C, D, E, or F]"
        )
        user_prompt = (
            f"Let's understand this Arabic medical question step-by-step in {reasoning_language}!\n\n"
            f"Question:\n{mcq_text}"
        )
        return system_prompt, user_prompt

    @staticmethod
    def _build_ansgen_cot_prompts(
        question_text: str, reasoning_language: str, instruction: str = ""
    ) -> Tuple[str, str]:
        inst = (instruction or "").strip()
        system_prompt = (
            f"You are a highly capable medical assistant. "
            f"Act as an expert in medical reasoning in {reasoning_language}.\n"
            "The question is written in Arabic. "
            f"Reason step-by-step in {reasoning_language} and end your response with:\n"
            "Final answer in Arabic: [concise answer in Arabic]"
        )
        if inst:
            system_prompt += f"\n\nAdditional instruction:\n{inst}"

        user_prompt = (
            f"Let's understand this Arabic medical question step-by-step in {reasoning_language}!\n\n"
            f"Question:\n{question_text}"
        )
        return system_prompt, user_prompt

    @staticmethod
    def _build_dialogue_cot_prompts(
        dialogue_text: str, reasoning_language: str, instruction: str = ""
    ) -> Tuple[str, str]:
        inst = (instruction or "").strip()
        system_prompt = (
            f"You are a highly capable medical assistant completing a doctor-patient dialogue. "
            f"Act as an expert in medical reasoning in {reasoning_language}.\n"
            "The dialogue is written in Arabic. "
            f"Reason step-by-step in {reasoning_language} about what the doctor should say next, "
            "then end your response with:\n"
            "ANSWER: [doctor's response in Arabic — exactly one line]"
        )
        if inst:
            system_prompt += f"\n\nAdditional instruction:\n{inst}"

        user_prompt = (
            f"Let's understand this Arabic medical dialogue step-by-step in {reasoning_language} "
            "and determine the doctor's next response!\n\n"
            f"Dialogue:\n{dialogue_text}"
        )
        return system_prompt, user_prompt

    # -----------------------------------------------------------------------
    # Stage helpers  (each returns (extracted_answer, full_cot_text))
    # -----------------------------------------------------------------------
    def _select_languages(
        self, content_text: str, task_type: str = "mcq"
    ) -> List[str]:
        system_prompt, user_prompt = self._build_language_selection_prompts(
            content_text, task_type
        )
        raw = self._generate(system_prompt, user_prompt, self.selection_max_tokens)
        print(f"[AutoCAPMedGemma] Language selection raw: {repr(raw)}")

        parsed = self._safe_json_loads(raw)
        selected = []
        if isinstance(parsed, dict) and isinstance(parsed.get("languages"), list):
            for x in parsed["languages"]:
                if isinstance(x, str) and x in self.candidate_languages and x not in selected:
                    selected.append(x)

        if len(selected) < self.top_k_languages:
            fallback = [l for l in self.candidate_languages if l not in selected]
            selected.extend(fallback[: self.top_k_languages - len(selected)])

        return selected[: self.top_k_languages]

    def _assign_weights(
        self,
        content_text: str,
        selected_languages: List[str],
        task_type: str = "mcq",
    ) -> Dict[str, float]:
        system_prompt, user_prompt = self._build_weight_assignment_prompts(
            content_text, selected_languages, task_type
        )
        raw = self._generate(system_prompt, user_prompt, self.weight_max_tokens)
        print(f"[AutoCAPMedGemma] Weight assignment raw: {repr(raw)}")

        parsed = self._safe_json_loads(raw)
        weights: Dict[str, float] = {}
        if isinstance(parsed, dict) and isinstance(parsed.get("weights"), dict):
            for lang in selected_languages:
                if lang in parsed["weights"]:
                    weights[lang] = parsed["weights"][lang]

        return self._normalize_weights(weights, selected_languages)

    def _reason_mcq(
        self, mcq_text: str, reasoning_language: str
    ) -> Tuple[Optional[str], str]:
        system_prompt, user_prompt = self._build_mcq_cot_prompts(
            mcq_text, reasoning_language
        )
        raw = self._generate(system_prompt, user_prompt, self.cot_max_tokens)
        print(f"[AutoCAPMedGemma] MCQ CoT ({reasoning_language}): {repr(raw[:300])}")
        return self._extract_letter_from_cot(raw), raw

    def _reason_ansgen(
        self, question_text: str, reasoning_language: str, instruction: str = ""
    ) -> Tuple[str, str]:
        system_prompt, user_prompt = self._build_ansgen_cot_prompts(
            question_text, reasoning_language, instruction
        )
        raw = self._generate(system_prompt, user_prompt, self.cot_max_tokens)
        print(f"[AutoCAPMedGemma] Answer-gen CoT ({reasoning_language}): {repr(raw[:300])}")
        return self._extract_ansgen_from_cot(raw), raw

    def _reason_dialogue(
        self, dialogue_text: str, reasoning_language: str, instruction: str = ""
    ) -> Tuple[str, str]:
        system_prompt, user_prompt = self._build_dialogue_cot_prompts(
            dialogue_text, reasoning_language, instruction
        )
        raw = self._generate(system_prompt, user_prompt, self.cot_max_tokens)
        print(f"[AutoCAPMedGemma] Dialogue CoT ({reasoning_language}): {repr(raw[:300])}")
        return self._extract_dialogue_from_cot(raw), raw

    # -----------------------------------------------------------------------
    # Batched CoT reasoning stage  (replaces the per-language for loops)
    # -----------------------------------------------------------------------
    def _reason_languages(
        self,
        content_text: str,
        selected_languages: List[str],
        task_type: str,
        instruction: str = "",
    ) -> Dict[str, Tuple[Optional[str], str]]:
        if not selected_languages:
            return {}

        prompt_pairs: List[Tuple[str, str]] = []
        for lang in selected_languages:
            if task_type == "mcq":
                sp, up = self._build_mcq_cot_prompts(content_text, lang)
            elif task_type == "answer_generation":
                sp, up = self._build_ansgen_cot_prompts(content_text, lang, instruction)
            else:  # dialogue_completion / dialogue / next_turn
                sp, up = self._build_dialogue_cot_prompts(content_text, lang, instruction)
            prompt_pairs.append((sp, up))

        raw_outputs = self._generate_batch(prompt_pairs, self.cot_max_tokens)

        results: Dict[str, Tuple[Optional[str], str]] = {}
        for lang, raw in zip(selected_languages, raw_outputs):
            print(f"[AutoCAPMedGemma] {task_type} CoT ({lang}): {repr(raw[:300])}")
            if task_type == "mcq":
                answer: Optional[str] = self._extract_letter_from_cot(raw)
            elif task_type == "answer_generation":
                answer = self._extract_ansgen_from_cot(raw)
            else:
                answer = self._extract_dialogue_from_cot(raw)
            results[lang] = (answer, raw)

        return results

    # -----------------------------------------------------------------------
    # Aggregation
    # -----------------------------------------------------------------------
    @staticmethod
    def _weighted_vote(
        answers: Dict[str, str], weights: Dict[str, float]
    ) -> Optional[str]:
        score: Dict[str, float] = {}
        for lang, letter in answers.items():
            if letter is None:
                continue
            w = float(weights.get(lang, 0.0))
            score[letter] = score.get(letter, 0.0) + w
        if not score:
            return None
        return max(score, key=lambda k: (score[k], -ord(k)))

    @staticmethod
    def _weighted_selection(
        answers: Dict[str, str], weights: Dict[str, float]
    ) -> str:
        best_lang, best_w = None, -1.0
        for lang, ans in answers.items():
            if not ans:
                continue
            w = float(weights.get(lang, 0.0))
            if best_lang is None or w > best_w:
                best_lang, best_w = lang, w
        return answers.get(best_lang, "") if best_lang else ""

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------
    def prompt(
        self,
        sample: dict,
        instruction: str = "",
        max_tokens: int = 12,        # ignored; handler uses cot_max_tokens
        task_type: str = "mcq",
        return_debug: bool = False,
    ):
        task_type = (task_type or "mcq").strip().lower()

        if task_type == "mcq":
            mcq_text = self._build_mcq_text(sample)
            if not mcq_text:
                print("[AutoCAPMedGemma] Empty stem/options; cannot build MCQ prompt.")
                empty = {
                    "final_prediction": None, "selected_languages": [],
                    "weights": {}, "language_answers": {}, "language_reasoning": {},
                }
                return empty if return_debug else None

            selected = self._select_languages(mcq_text, "mcq")
            weights  = self._assign_weights(mcq_text, selected, "mcq")

            lang_results = self._reason_languages(mcq_text, selected, "mcq")
            language_answers:   Dict[str, str] = {}
            language_reasoning: Dict[str, str] = {}
            for lang, (letter, cot) in lang_results.items():
                language_reasoning[lang] = cot
                if letter is not None:
                    language_answers[lang] = letter

            final_prediction = self._weighted_vote(language_answers, weights)

            debug = {
                "final_prediction":   final_prediction,
                "selected_languages": selected,
                "weights":            weights,
                "language_answers":   language_answers,
                "language_reasoning": language_reasoning,
            }
            print(f"[AutoCAPMedGemma] MCQ result: pred={final_prediction} | "
                  f"answers={language_answers} | weights={weights}")
            return debug if return_debug else final_prediction

        elif task_type == "answer_generation":
            question_text = self._build_ansgen_text(sample)
            if not question_text:
                print("[AutoCAPMedGemma] Empty question; cannot build answer-generation prompt.")
                empty = {
                    "final_prediction": "", "selected_languages": [],
                    "weights": {}, "language_answers": {}, "language_reasoning": {},
                }
                return empty if return_debug else ""

            selected = self._select_languages(question_text, "answer_generation")
            weights  = self._assign_weights(question_text, selected, "answer_generation")

            lang_results = self._reason_languages(
                question_text, selected, "answer_generation", instruction
            )
            language_answers:   Dict[str, str] = {}
            language_reasoning: Dict[str, str] = {}
            for lang, (answer, cot) in lang_results.items():
                language_reasoning[lang] = cot
                if answer:
                    language_answers[lang] = answer

            final_prediction = self._weighted_selection(language_answers, weights)

            debug = {
                "final_prediction":   final_prediction,
                "selected_languages": selected,
                "weights":            weights,
                "language_answers":   language_answers,
                "language_reasoning": language_reasoning,
            }
            print(f"[AutoCAPMedGemma] Answer-gen result: pred={repr(final_prediction[:100])} | "
                  f"weights={weights}")
            return debug if return_debug else final_prediction

        elif task_type in ("dialogue_completion", "dialogue", "next_turn"):
            dialogue_text = self._build_dialogue_text(sample)
            if not dialogue_text:
                print("[AutoCAPMedGemma] Empty dialogue; cannot build dialogue-completion prompt.")
                empty = {
                    "final_prediction": "", "selected_languages": [],
                    "weights": {}, "language_answers": {}, "language_reasoning": {},
                }
                return empty if return_debug else ""

            selected = self._select_languages(dialogue_text, "dialogue_completion")
            weights  = self._assign_weights(dialogue_text, selected, "dialogue_completion")

            lang_results = self._reason_languages(
                dialogue_text, selected, "dialogue_completion", instruction
            )
            language_answers:   Dict[str, str] = {}
            language_reasoning: Dict[str, str] = {}
            for lang, (answer, cot) in lang_results.items():
                language_reasoning[lang] = cot
                if answer:
                    language_answers[lang] = answer

            final_prediction = self._weighted_selection(language_answers, weights)

            debug = {
                "final_prediction":   final_prediction,
                "selected_languages": selected,
                "weights":            weights,
                "language_answers":   language_answers,
                "language_reasoning": language_reasoning,
            }
            print(f"[AutoCAPMedGemma] Dialogue result: pred={repr(final_prediction[:100])} | "
                  f"weights={weights}")
            return debug if return_debug else final_prediction

        else:
            raise ValueError(
                f"Unsupported task_type={task_type!r}. "
                "Expected 'mcq', 'answer_generation', or 'dialogue_completion'."
            )

    def prompt_batch(
        self,
        samples: List[dict],
        instruction: str = "",
        max_tokens: int = 12,
        task_type: str = "mcq",
    ):
        return [
            self.prompt(
                s,
                instruction=instruction,
                max_tokens=max_tokens,
                task_type=task_type,
                return_debug=False,
            )
            for s in samples
        ]
