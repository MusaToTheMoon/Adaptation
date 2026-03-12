# models/gemini3pro_handler.py
import os
import re
from typing import Optional, Dict, Any

from google import genai
from google.genai import types


def extract_letter_from_text(text: str) -> Optional[str]:
    if not text:
        return None
    s = str(text).strip().upper()

    m = re.search(r"\bANSWER\s*[:=]\s*([A-F])\b", s)
    if m:
        return m.group(1)

    m = re.search(r"\b([A-F])\b", s)
    return m.group(1) if m else None


class Gemini3ProHandler:
    """
    MCQ handler aligned with your HF eval pipeline.
    Accepts `sample` dict just like Fanar/Meditron.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-3.1-pro-preview",
    ):
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY missing.")

        self.client = genai.Client(api_key=api_key)
        self.model_name = model

    @staticmethod
    def _build_mcq_text(sample: Dict[str, Any]) -> str:
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
            if txt:
                lines.append(f"{letter}) {str(txt).strip()}")

        return stem + "\n\n" + "\n".join(lines)

    def prompt(
        self,
        sample: Dict[str, Any],
        instruction: str,
        max_tokens: int = 128,
        temperature: float = 0.0,
    ) -> Optional[str]:

        user_text = self._build_mcq_text(sample)
        if not user_text:
            return None

        system_prompt = (instruction or "").strip()

        try:
            cfg = types.GenerateContentConfig(
                system_instruction=system_prompt if system_prompt else None,
                temperature=float(temperature),
                top_p=1.0,
                candidate_count=1,
                max_output_tokens=int(max_tokens),
                stop_sequences=["\n"],  # helps prevent spillover
            )

            resp = self.client.models.generate_content(
                model=self.model_name,
                contents=[
                    types.Content(
                        role="user",
                        parts=[types.Part(text=user_text)]
                    )
                ],
                config=cfg,
            )

            raw = (getattr(resp, "text", "") or "").strip()
            print(f"[Gemini3Pro] Raw: {repr(raw)}")
            return extract_letter_from_text(raw)

        except Exception as e:
            print(f"[Gemini3Pro] Error: {e}")
            return None