"""
AutoCAP baseline — faithful adaptation of:
  "AUTO CAP: Towards Automatic Cross-lingual Alignment Planning
   for Zero-shot Chain-of-Thought" (Zhang et al., ACL 2024 Findings)

Pipeline (multi-round, matching the paper):
  Round 1  — Automatic Language Selection Prompting (ALSP):
               LLM selects top-K reasoning languages from a candidate pool,
               informed by per-language metadata (family, branch, pre-train %).
  Round 2  — Automatic Weight Allocation Prompting (AWAP):
               LLM assigns a reliability weight [0,1] to each selected language.
  Round 3  — CoT Reasoning:
               For each selected language, the LLM reasons step-by-step
               ("Let's understand the task in {lang} step-by-step!") and
               produces a final answer.
  Aggregation —
    task_type="mcq"                 : weighted self-consistency vote over
                                      extracted letters (Eq. 8 of the paper).
    task_type="answer_generation"   : pick the answer from the highest-weight
                                      language (weighted selection — natural
                                      extension of Eq. 8 to free-text outputs).
    task_type="dialogue_completion" : same weighted selection.

Task 2 / Task 3 extension:
  The original paper only evaluated discrete-answer tasks (math, NLI, paraphrase).
  For free-text generation the weighted vote over identical strings is degenerate,
  so we use weighted selection instead: the output of the highest-weight language
  is chosen as the final prediction.  The full per-language CoT chains are stored
  in the debug dict for inclusion in the appendix.
"""

import json
import os
import re
from typing import Dict, List, Optional, Tuple

import torch
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from transformers import Mistral3ForConditionalGeneration

HF_CACHE = "/scratch/ca2627/huggingface"
os.environ["HF_HOME"] = HF_CACHE

os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
os.environ.setdefault("TORCH_USE_CUDA_DSA", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


# ---------------------------------------------------------------------------
# Per-language metadata used by ALSP (mirrors the paper's L_info block).
# Fields: language family, branch, and approximate share of typical LLM
# pre-training data (estimates; exact figures are model-dependent).
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


class AutoCAPMCQHandler:
    """
    AutoCAP handler (Zhang et al., ACL 2024 Findings) adapted for three task types:
      - task_type="mcq"                 -> returns letter A-F or None
      - task_type="answer_generation"   -> returns free-text answer (Arabic) or ""
      - task_type="dialogue_completion" -> returns doctor turn (one line, Arabic) or ""
      - prompt_batch(samples, ...)      -> list of the above

    If return_debug=True, prompt() returns a dict:
      {
        "final_prediction":   <letter or str>,
        "selected_languages": [...],
        "weights":            {lang: float},
        "language_answers":   {lang: extracted_answer},   # what gets voted/selected
        "language_reasoning": {lang: full_cot_text},      # full CoT chains (for appendix)
      }
    """

    def __init__(
        self,
        model_name: str = "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
        cache_dir: str = HF_CACHE,
        offline: bool = True,
        candidate_languages: Optional[List[str]] = None,
        top_k_languages: int = 3,
        selection_max_tokens: int = 128,
        weight_max_tokens: int = 128,
        cot_max_tokens: int = 512,        # preferred name — CoT needs room
        reasoning_max_tokens: int = None, # legacy alias from old configs; mapped to cot_max_tokens
        do_sample: bool = False,
    ):
        print(f"[AutoCAPMCQ] Handler file: {__file__}")
        print(f"[AutoCAPMCQ] HF_HOME={os.environ.get('HF_HOME')}")
        print(f"[AutoCAPMCQ] cache_dir={cache_dir}")
        print(f"[AutoCAPMCQ] offline={offline}")

        self.model_name = model_name
        self.cache_dir = cache_dir or HF_CACHE
        self.offline = bool(offline)
        self.top_k_languages = int(top_k_languages)
        self.selection_max_tokens = int(selection_max_tokens)
        self.weight_max_tokens = int(weight_max_tokens)
        # reasoning_max_tokens is a legacy alias — if the YAML still uses the old
        # name it takes priority so existing configs don't silently get 512 tokens.
        if reasoning_max_tokens is not None:
            self.cot_max_tokens = int(reasoning_max_tokens)
        else:
            self.cot_max_tokens = int(cot_max_tokens)
        print(f"[AutoCAPMCQ] cot_max_tokens={self.cot_max_tokens} "
              f"({'from reasoning_max_tokens alias' if reasoning_max_tokens is not None else 'default'})")
        self.do_sample = bool(do_sample)

        self.candidate_languages = candidate_languages or ["Arabic", "English", "French"]

        if self.top_k_languages < 1:
            raise ValueError("top_k_languages must be >= 1")
        if self.top_k_languages > len(self.candidate_languages):
            raise ValueError("top_k_languages cannot exceed number of candidate languages")

        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

        self.local_files_only = True
        if self.offline:
            os.environ["HF_HUB_OFFLINE"] = "1"
        else:
            os.environ.pop("HF_HUB_OFFLINE", None)

        num_gpus = torch.cuda.device_count()
        print(f"[AutoCAPMCQ] Available GPUs: {num_gpus}")
        if num_gpus == 0:
            raise RuntimeError("No CUDA GPUs available for AutoCAPMCQHandler.")
        for i in range(num_gpus):
            print(
                f"  GPU {i}: {torch.cuda.get_device_name(i)} - "
                f"{torch.cuda.memory_allocated(i) / 1024**3:.2f} GB allocated"
            )

        print("[AutoCAPMCQ] Loading tokenizer...")
        if os.path.isdir(self.model_name):
            self.tokenizer = MistralTokenizer.from_file(
                os.path.join(self.model_name, "tekken.json")
            )
        else:
            self.tokenizer = MistralTokenizer.from_hf_hub(self.model_name)

        print("[AutoCAPMCQ] Loading model...")
        self.model = Mistral3ForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            cache_dir=self.cache_dir,
            local_files_only=self.local_files_only,
        )
        print("[AutoCAPMCQ] Model loaded.")
        if hasattr(self.model, "hf_device_map"):
            print("[AutoCAPMCQ] Model device distribution:")
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
        """
        Extract the final answer letter from a multi-line CoT output.
        Scans for explicit 'Answer: X' markers first; falls back to the
        last standalone letter in the text (the paper's CoT ends with the answer).
        """
        if not text:
            return None
        upper = text.strip().upper()
        # Explicit answer markers (scan the whole text, most specific first)
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
        # Fall back: last standalone A-F in the text
        matches = list(re.finditer(r"\b([A-F])\b", upper))
        if matches:
            return matches[-1].group(1)
        return None

    @staticmethod
    def _extract_ansgen_from_cot(text: str) -> str:
        """
        Extract the final Arabic answer from a CoT output.
        Looks for an explicit 'Final answer in Arabic: ...' marker;
        falls back to the last non-empty line (where CoT conventionally ends).
        """
        if not text:
            return ""
        # Explicit final-answer markers
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
        # Fall back: last non-empty line
        lines = [ln.strip() for ln in text.strip().split("\n") if ln.strip()]
        if lines:
            last = lines[-1]
            # Strip any residual prefix
            m2 = re.match(r"^\s*(?:ANSWER|Answer)\s*[:=]\s*(.+)$", last)
            if m2:
                last = m2.group(1).strip()
            # Strip paired wrapping quotes
            if len(last) >= 2 and last[0] == last[-1] and last[0] in ('"', "'", "“", "”"):
                last = last[1:-1].strip()
            return last
        return ""

    @staticmethod
    def _extract_dialogue_from_cot(text: str) -> str:
        """
        Extract the final doctor turn from a CoT output.
        Looks for an explicit 'ANSWER: ...' marker anywhere in the text
        (the reasoning prompt instructs the model to end with exactly this);
        falls back to the last non-empty line.
        """
        if not text:
            return ""
        # Look for ANSWER: marker — could appear mid-text after CoT reasoning
        m = re.search(r"ANSWER\s*[:=]\s*(.+?)(?:\n|$)", text, flags=re.IGNORECASE)
        if m:
            ans = m.group(1).strip()
            if ans:
                return ans
        # Fall back: last non-empty line
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
            {"role": "user",   "content": [{"type": "text", "text": user_text}]},
        ]
        try:
            torch.cuda.empty_cache()
            req = ChatCompletionRequest(messages=messages)
            tokenized = self.tokenizer.encode_chat_completion(req)
            input_ids = torch.tensor(
                [tokenized.tokens], dtype=torch.long, device=self.model.device
            )
            attention_mask = torch.ones_like(input_ids)
            with torch.inference_mode():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_tokens,
                    do_sample=self.do_sample,
                )
            if outputs is None or outputs.shape[0] == 0:
                return ""
            input_len = len(tokenized.tokens)
            gen_ids = outputs[0][input_len:]
            return self.tokenizer.decode(gen_ids).strip() or ""
        finally:
            torch.cuda.empty_cache()

    def _generate_batch(
        self, prompts: List[Tuple[str, str]], max_tokens: int
    ) -> List[str]:
        """
        Run multiple (system_prompt, user_text) pairs through the model in ONE
        batched GPU forward pass.

        Each prompt is tokenised independently, then all sequences are
        left-padded to the same length so every item in the batch starts
        generating from the same step.  Generated tokens are decoded and
        returned in the same order as `prompts`.

        Falls back to sequential _generate() calls on OOM so that no sample
        is ever silently dropped.

        Speedup rationale: the three per-language CoT calls are the bottleneck
        (512 max tokens each, ~80% of per-sample wall time).  Batching them
        trades a larger KV cache for ~2x–3x throughput on the decode phase.
        """
        if not prompts:
            return []
        # Single-prompt fast path — skip padding overhead
        if len(prompts) == 1:
            return [self._generate(prompts[0][0], prompts[0][1], max_tokens)]

        try:
            torch.cuda.empty_cache()

            # ---- Tokenise every prompt individually --------------------------
            token_id_lists: List[List[int]] = []
            for system_prompt, user_text in prompts:
                messages = [
                    {"role": "system", "content": (system_prompt or "").strip()},
                    {"role": "user",   "content": [{"type": "text", "text": user_text}]},
                ]
                req = ChatCompletionRequest(messages=messages)
                tokenized = self.tokenizer.encode_chat_completion(req)
                token_id_lists.append(list(tokenized.tokens))

            # ---- Resolve pad token ID (EOS is the Mistral convention) --------
            pad_id: int = 2  # Mistral EOS token — safe fallback
            try:
                pad_id = self.tokenizer.instruct_tokenizer.tokenizer.eos_id
            except AttributeError:
                pass

            # ---- Left-pad all sequences to the same length -------------------
            # Left-padding (not right) is required for autoregressive generation
            # so that every item's last real token is at the same position.
            max_input_len = max(len(ids) for ids in token_id_lists)
            padded_ids:  List[List[int]] = []
            attn_masks:  List[List[int]] = []
            for ids in token_id_lists:
                pad_len = max_input_len - len(ids)
                padded_ids.append([pad_id] * pad_len + ids)
                attn_masks.append([0]      * pad_len + [1] * len(ids))

            input_ids = torch.tensor(
                padded_ids, dtype=torch.long, device=self.model.device
            )
            attention_mask = torch.tensor(
                attn_masks, dtype=torch.long, device=self.model.device
            )

            # ---- Single batched generate call --------------------------------
            with torch.inference_mode():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_tokens,
                    do_sample=self.do_sample,
                    pad_token_id=pad_id,   # tells generate what to fill after EOS
                )

            # ---- Decode each item --------------------------------------------
            # Generated tokens occupy positions [max_input_len:] for every row
            # because all inputs were padded to max_input_len.
            results: List[str] = []
            for i in range(len(prompts)):
                gen_ids = outputs[i][max_input_len:].tolist()
                # Truncate at the first EOS — other sequences in the batch may
                # have continued generating padding tokens past this item's EOS.
                try:
                    eos_pos = gen_ids.index(pad_id)
                    gen_ids = gen_ids[:eos_pos]
                except ValueError:
                    pass
                raw = self.tokenizer.decode(gen_ids).strip() if gen_ids else ""
                results.append(raw)

            return results

        except torch.cuda.OutOfMemoryError:
            # OOM safety net: run items one-by-one rather than lose the sample.
            print(
                f"[AutoCAP] _generate_batch: OOM with batch_size={len(prompts)}, "
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
        """
        Build ALSP prompts (paper §3.1).
        Language metadata (L_info) is included verbatim as in the paper (Fig. 2).
        """
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
        """
        Build AWAP prompts (paper §3.2).
        Weight range is [0, 1] as in the paper.
        """
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
        """
        MCQ zero-shot CoT reasoning (paper §3, Fig. 2).
        Model reasons step-by-step in `reasoning_language`, ends with 'Answer: [letter]'.
        """
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
        """
        Answer-generation zero-shot CoT reasoning (Task 2 extension).
        Model reasons step-by-step in `reasoning_language`, ends with
        'Final answer in Arabic: [answer]'.
        `instruction` from the runner is appended to the system prompt.
        """
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
        """
        Dialogue-completion zero-shot CoT reasoning (Task 3 extension).
        Model reasons step-by-step in `reasoning_language` about the doctor's
        next turn, then ends with 'ANSWER: [Arabic doctor response]'.
        `instruction` from the runner (task3-MSA.txt) is appended to system prompt.
        """
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
        print(f"[AutoCAPMCQ] Language selection raw: {repr(raw)}")

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
        print(f"[AutoCAPMCQ] Weight assignment raw: {repr(raw)}")

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
        """Returns (extracted_letter_or_None, full_cot_text)."""
        system_prompt, user_prompt = self._build_mcq_cot_prompts(
            mcq_text, reasoning_language
        )
        raw = self._generate(system_prompt, user_prompt, self.cot_max_tokens)
        print(f"[AutoCAPMCQ] MCQ CoT ({reasoning_language}): {repr(raw[:300])}")
        return self._extract_letter_from_cot(raw), raw

    def _reason_ansgen(
        self, question_text: str, reasoning_language: str, instruction: str = ""
    ) -> Tuple[str, str]:
        """Returns (extracted_answer, full_cot_text)."""
        system_prompt, user_prompt = self._build_ansgen_cot_prompts(
            question_text, reasoning_language, instruction
        )
        raw = self._generate(system_prompt, user_prompt, self.cot_max_tokens)
        print(f"[AutoCAP] Answer-gen CoT ({reasoning_language}): {repr(raw[:300])}")
        return self._extract_ansgen_from_cot(raw), raw

    def _reason_dialogue(
        self, dialogue_text: str, reasoning_language: str, instruction: str = ""
    ) -> Tuple[str, str]:
        """Returns (extracted_doctor_turn, full_cot_text)."""
        system_prompt, user_prompt = self._build_dialogue_cot_prompts(
            dialogue_text, reasoning_language, instruction
        )
        raw = self._generate(system_prompt, user_prompt, self.cot_max_tokens)
        print(f"[AutoCAP] Dialogue CoT ({reasoning_language}): {repr(raw[:300])}")
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
        """
        Build CoT prompts for every selected language and run them in a single
        batched GPU forward pass via _generate_batch.

        Returns {language: (extracted_answer, full_cot_text)}.
          - "mcq"                : extracted_answer is a letter A-F or None
          - "answer_generation"  : extracted_answer is a non-empty string or ""
          - "dialogue_completion": extracted_answer is a non-empty string or ""

        The full_cot_text for every language is stored for the appendix.
        """
        if not selected_languages:
            return {}

        # Build all CoT prompts up front
        prompt_pairs: List[Tuple[str, str]] = []
        for lang in selected_languages:
            if task_type == "mcq":
                sp, up = self._build_mcq_cot_prompts(content_text, lang)
            elif task_type == "answer_generation":
                sp, up = self._build_ansgen_cot_prompts(content_text, lang, instruction)
            else:  # dialogue_completion / dialogue / next_turn
                sp, up = self._build_dialogue_cot_prompts(content_text, lang, instruction)
            prompt_pairs.append((sp, up))

        # One batched forward pass for all K languages
        raw_outputs = self._generate_batch(prompt_pairs, self.cot_max_tokens)

        # Extract answers and log CoT chains
        results: Dict[str, Tuple[Optional[str], str]] = {}
        for lang, raw in zip(selected_languages, raw_outputs):
            print(f"[AutoCAP] {task_type} CoT ({lang}): {repr(raw[:300])}")
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
        """
        MCQ: weighted self-consistency vote (Eq. 8 of the paper).
        answers  : {language: extracted_letter}
        weights  : {language: normalised_weight}
        """
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
        """
        Task 2 / Task 3: weighted selection (extension of Eq. 8 for free-text).
        Returns the answer from the highest-weight language that produced
        a non-empty response; falls back to any non-empty answer.
        """
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
        """
        Unified entry point.

        task_type="mcq"
            AutoCAP pipeline faithful to the paper.  Returns letter A-F or None.
            `instruction` is ignored (AutoCAP uses its own staged prompts for MCQ).

        task_type="answer_generation"
            AutoCAP pipeline extended to free-text.  Returns Arabic answer string.
            `instruction` is folded into the per-language CoT system prompt.

        task_type="dialogue_completion"
            AutoCAP pipeline extended to free-text.  Returns Arabic doctor turn
            (one line, ANSWER: stripped).
            `instruction` is folded into the per-language CoT system prompt.

        If return_debug=True, returns a dict with keys:
            final_prediction, selected_languages, weights,
            language_answers (extracted per-language results),
            language_reasoning (full per-language CoT chains — for appendix).
        """
        task_type = (task_type or "mcq").strip().lower()

        # ------------------------------------------------------------------
        # Task 1 — MCQ  (faithful to paper)
        # ------------------------------------------------------------------
        if task_type == "mcq":
            mcq_text = self._build_mcq_text(sample)
            if not mcq_text:
                print("[AutoCAPMCQ] Empty stem/options; cannot build MCQ prompt.")
                empty = {
                    "final_prediction": None, "selected_languages": [],
                    "weights": {}, "language_answers": {}, "language_reasoning": {},
                }
                return empty if return_debug else None

            selected = self._select_languages(mcq_text, "mcq")
            weights  = self._assign_weights(mcq_text, selected, "mcq")

            # Batch all K CoT calls into one forward pass
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
            print(f"[AutoCAPMCQ] MCQ result: pred={final_prediction} | "
                  f"answers={language_answers} | weights={weights}")
            return debug if return_debug else final_prediction

        # ------------------------------------------------------------------
        # Task 2 — Answer generation  (weighted selection extension)
        # ------------------------------------------------------------------
        elif task_type == "answer_generation":
            question_text = self._build_ansgen_text(sample)
            if not question_text:
                print("[AutoCAP] Empty question; cannot build answer-generation prompt.")
                empty = {
                    "final_prediction": "", "selected_languages": [],
                    "weights": {}, "language_answers": {}, "language_reasoning": {},
                }
                return empty if return_debug else ""

            selected = self._select_languages(question_text, "answer_generation")
            weights  = self._assign_weights(question_text, selected, "answer_generation")

            # Batch all K CoT calls into one forward pass
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
            print(f"[AutoCAP] Answer-gen result: pred={repr(final_prediction[:100])} | "
                  f"weights={weights}")
            return debug if return_debug else final_prediction

        # ------------------------------------------------------------------
        # Task 3 — Dialogue completion  (weighted selection extension)
        # ------------------------------------------------------------------
        elif task_type in ("dialogue_completion", "dialogue", "next_turn"):
            dialogue_text = self._build_dialogue_text(sample)
            if not dialogue_text:
                print("[AutoCAP] Empty dialogue; cannot build dialogue-completion prompt.")
                empty = {
                    "final_prediction": "", "selected_languages": [],
                    "weights": {}, "language_answers": {}, "language_reasoning": {},
                }
                return empty if return_debug else ""

            selected = self._select_languages(dialogue_text, "dialogue_completion")
            weights  = self._assign_weights(dialogue_text, selected, "dialogue_completion")

            # Batch all K CoT calls into one forward pass
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
            print(f"[AutoCAP] Dialogue result: pred={repr(final_prediction[:100])} | "
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