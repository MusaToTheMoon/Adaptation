import re

from openai import OpenAI


def extract_letter_from_text_en(text: str):
    """
    Extract a single MCQ letter A–F from model output.
    Returns 'A'..'F' or None.
    """
    if not text:
        return None
    t = str(text).upper()
    m = re.search(r"\bANSWER\s*[:=]\s*([A-F])\b", t)
    if m:
        return m.group(1)
    m = re.search(r"\b([A-F])\b", t)
    if m:
        return m.group(1)
    m = re.search(r"\b([A-F])(?=[\.\)\]:;\s]|$)", t)
    return m.group(1) if m else None


class GPT5FamilyMCQHandler:
    """
        Unified GPT-5 family handler for:
            - task_type="mcq"               -> returns a single letter A-F (or None)
            - task_type="answer_generation" -> returns generated text (or "")
            - task_type="dialogue_completion" -> returns generated text (one line, "ANSWER:" stripped)

        Input sample (dict):
            - For MCQ: question + opa/opb/opc/opd (+ optional ope/opf)
            - For answer generation: question
    """

    def __init__(self, api_key: str, model: str = "gpt-5"):
        self.log_tag = self._build_log_tag(model)
        print(f"[{self.log_tag}] model={model}")
        self.client = OpenAI(api_key=api_key)
        self.model = model

    @staticmethod
    def _build_log_tag(model: str) -> str:
        model_l = (model or "").lower()
        if model_l.startswith("gpt-5"):
            suffix = model_l[len("gpt-5"):].strip()
            digits = "".join(ch for ch in suffix if ch.isdigit())
            if digits:
                return f"GPT5{digits}MCQ"
            return "GPT5MCQ"
        return "GPT5FamilyMCQ"

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
        """Build answer-generation input from the question stem only."""
        return (sample.get("question") or "").strip()

    @staticmethod
    def _build_dialogue_text(sample: dict) -> str:
        """Build Task 3 dialogue input from PascalCase or snake_case fields."""
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
    def _parse_responses_text(resp) -> str:
        def _get(obj, key, default=None):
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        def _to_text(value):
            if isinstance(value, str):
                return value
            if isinstance(value, list):
                chunks = [_to_text(v) for v in value]
                return "".join([c for c in chunks if c])
            if isinstance(value, dict):
                for k in ("text", "value", "content"):
                    if k in value:
                        return _to_text(value[k])
                return ""
            for k in ("text", "value", "content"):
                v = getattr(value, k, None)
                if v is not None:
                    return _to_text(v)
            return ""

        txt = _to_text(_get(resp, "output_text", None)).strip()
        if txt:
            return txt

        parts = []
        for item in (_get(resp, "output", []) or []):
            item_type = _get(item, "type", "")

            if item_type in ("output_text", "text"):
                t = _to_text(_get(item, "text", item)).strip()
                if t:
                    parts.append(t)
                continue

            if item_type != "message":
                continue

            for c in (_get(item, "content", []) or []):
                ctype = _get(c, "type", "")
                if ctype not in ("output_text", "text"):
                    continue
                t = _to_text(_get(c, "text", c)).strip()
                if t:
                    parts.append(t)

        return "\n".join(parts).strip()

    def _create_with_reasoning_fallback(self, payload: dict):
        """Create a Responses API call with robust reasoning-effort fallback per model support."""
        attempts = [
            {"effort": "none"},
            {"effort": "low"},
        ]

        last_error = None
        for reasoning_cfg in attempts:
            try:
                req = dict(payload)
                req["reasoning"] = reasoning_cfg
                return self.client.responses.create(**req)
            except Exception as e:
                msg = str(e).lower()
                if "reasoning.effort" in msg and "unsupported" in msg:
                    last_error = e
                    continue
                raise

        if last_error is not None:
            raise last_error
        raise RuntimeError("Responses API call failed with reasoning fallback.")

    def prompt(
        self,
        sample: dict,
        instruction: str,
        max_tokens: int = 16,
        task_type: str = "mcq",
        **kwargs,
    ):
        """
        Unified interface:
        - sample is one JSON record (dict).
        - task_type="mcq": returns 'A'..'F' or None
        - task_type="answer_generation": returns generated text or ""
        - task_type="dialogue_completion": returns generated text or ""
        """
        task_type = (task_type or "mcq").strip().lower()

        if task_type == "mcq":
            user_text = self._build_mcq_text(sample)
            if not user_text:
                print(f"[{self.log_tag}] Empty stem/options; cannot build prompt.")
                return None
        elif task_type == "answer_generation":
            user_text = self._build_ansgen_text(sample)
            if not user_text:
                print(f"[{self.log_tag}] Empty question; cannot build answer-generation prompt.")
                return ""
        elif task_type == "dialogue_completion":
            user_text = self._build_dialogue_text(sample)
            if not user_text:
                print(f"[{self.log_tag}] Empty dialogue; cannot build dialogue-completion prompt.")
                return ""
        else:
            raise ValueError(
                f"Unsupported task_type={task_type}. Expected 'mcq', 'answer_generation', or 'dialogue_completion'."
            )

        system_prompt = (instruction or "Follow instructions strictly.").strip()
        if task_type == "mcq":
            safe_max_tokens = max(128, int(max_tokens or 16))
        else:
            safe_max_tokens = max(64, int(max_tokens or 256))

        print(f"[{self.log_tag}] TASK TYPE:", task_type)
        print(f"[{self.log_tag}] MAX TOKENS:", safe_max_tokens)

        raw = ""
        try:
            if task_type == "mcq":
                payload = dict(
                    model=self.model,
                    instructions=system_prompt,
                    input=(
                        f"{user_text}\n\n"
                        "Please output only the final answer letter (A, B, C, D, E, or F)."
                    ),
                    text={
                        "format": {
                            "type": "json_schema",
                            "name": "mcq_answer",
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "answer": {
                                        "type": "string",
                                        "enum": ["A", "B", "C", "D", "E", "F"],
                                    }
                                },
                                "required": ["answer"],
                                "additionalProperties": False,
                            },
                            "strict": True,
                        }
                    },
                    max_output_tokens=safe_max_tokens,
                )
            else:
                payload = dict(
                    model=self.model,
                    instructions=system_prompt,
                    input=user_text,
                    text={"format": {"type": "text"}},
                    max_output_tokens=safe_max_tokens,
                )

            resp = self._create_with_reasoning_fallback(payload)
            raw = self._parse_responses_text(resp)

            incomplete = getattr(resp, "incomplete_details", None)
            incomplete_reason = getattr(incomplete, "reason", None) if incomplete is not None else None

            if not raw:
                retry_payload = dict(payload)
                if task_type == "mcq":
                    retry_payload["text"] = {"format": {"type": "text"}}
                    retry_payload["input"] += (
                        "\n\nFinal answer (single letter A-F) only. "
                        "Do not include any other text."
                    )
                    retry_payload["max_output_tokens"] = (
                        max(256, safe_max_tokens * 2)
                        if incomplete_reason == "max_output_tokens"
                        else max(128, safe_max_tokens)
                    )
                else:
                    retry_payload["max_output_tokens"] = (
                        max(512, safe_max_tokens * 2)
                        if incomplete_reason == "max_output_tokens"
                        else max(64, safe_max_tokens)
                    )
                resp = self._create_with_reasoning_fallback(retry_payload)
                raw = self._parse_responses_text(resp)

                if not raw:
                    status = getattr(resp, "status", None)
                    incomplete = getattr(resp, "incomplete_details", None)
                    output_items = getattr(resp, "output", None)
                    out_len = len(output_items) if isinstance(output_items, list) else 0
                    print(
                        f"[{self.log_tag}] Empty output after retry. "
                        f"status={status} incomplete_details={incomplete} output_items={out_len}"
                    )

        except Exception as e:
            print(f"[{self.log_tag}] Error during generation:", e)
            return None if task_type == "mcq" else ""

        if not raw:
            print(f"[{self.log_tag}] Empty generation output.")
            return None if task_type == "mcq" else ""

        if task_type == "answer_generation":
            one_line = raw.split("\n")[0].strip()
            print(f"[{self.log_tag}] Answer-gen raw (one line): {repr(one_line[:300])}")
            return one_line

        if task_type == "dialogue_completion":
            one_line = raw.split("\n")[0].strip()
            m = re.match(r"^\s*ANSWER\s*[:=]\s*(.*)$", one_line, flags=re.IGNORECASE)
            if m:
                one_line = m.group(1).strip()
            print(f"[{self.log_tag}] Dialogue-completion raw (one line): {repr(one_line[:300])}")
            return one_line

        print(f"[{self.log_tag}] MCQ raw generated: {repr(raw)}")

        answer = extract_letter_from_text_en(raw)
        if answer:
            return answer

        print(f"[{self.log_tag}] Could not extract a clean letter.")
        return None

    def prompt_batch(
        self,
        samples,
        instruction: str,
        max_tokens: int = 16,
        task_type: str = "mcq",
        **kwargs,
    ):
        """
        Batched interface for compatibility with the evaluation pipeline.
        Uses per-sample Responses API calls to preserve behavior and output format.
        """
        if samples is None:
            return []

        results = []
        total = len(samples)
        print(f"[{self.log_tag}] prompt_batch size={total} task_type={task_type}")

        for idx, sample in enumerate(samples, start=1):
            print(f"[{self.log_tag}] prompt_batch item={idx}/{total}")
            result = self.prompt(
                sample,
                instruction=instruction,
                max_tokens=max_tokens,
                task_type=task_type,
                **kwargs,
            )
            results.append(result)

        return results
