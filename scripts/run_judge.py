"""
scripts/run_judge.py
=====================
Runs LLM-as-a-Judge evaluation on Task 2 (answer_generation) predictions.

Takes the predictions CSV produced by run_evaluation.py and adds a
'judge_label' column (Fully Correct / Partially Correct / Incorrect).
Aggregate judge counts are appended to the existing metrics JSON.

Usage:
    export OPENAI_API_KEY="sk-..."
    python scripts/run_judge.py configs/task2/medgemma.yaml
"""

import os
import sys
import json
import time
import logging
from pathlib import Path

import pandas as pd
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.utils import load_config  # noqa: E402

# ── Constants ──────────────────────────────────────────────────────────────────

MODEL        = "gpt-5.2"
MAX_TOKENS   = 50
TEMPERATURE  = 0
DELAY_SECS   = 0.5       # between API calls
MAX_RETRIES  = 3
RETRY_DELAY  = 5.0       # seconds to wait before retrying

VALID_LABELS = {"Fully Correct", "Partially Correct", "Incorrect"}

# ── Load judge prompt ──────────────────────────────────────────────────────────

def load_judge_prompt(config: dict) -> tuple[str, str]:
    """Load system and user prompt templates from the judge prompt txt file."""
    prompt_path = config.get("judge", {}).get("prompt_path")
    if not prompt_path:
        raise ValueError("config missing judge.prompt_path")

    with open(ROOT / prompt_path, "r", encoding="utf-8") as f:
        content = f.read()

    # The txt file has system and user separated by ---
    # If no separator, treat entire file as user prompt
    if "---" in content:
        system_part, user_part = content.split("---", 1)
        return system_part.strip(), user_part.strip()

    return (
        "You are an expert medical evaluator. You will be given a medical question, "
        "a reference answer, and a generated answer. Your task is to evaluate the "
        "generated answer by selecting exactly one label from the following options "
        "and responding only with the label in brackets [].",
        content.strip(),
    )

# ── Judge one row ──────────────────────────────────────────────────────────────

def call_judge(
    client: OpenAI,
    system_prompt: str,
    user_template: str,
    question_stem: str,
    reference_answer: str,
    generated_answer: str,
) -> str:
    """Call GPT-5.2 judge and return a clean label string."""

    user_prompt = user_template.format(
        question_stem=question_stem,
        reference_answer=reference_answer,
        generated_answer=generated_answer,
    )

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                # max_tokens=MAX_TOKENS,
                max_completion_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
            )
            raw = response.choices[0].message.content.strip()
            label = parse_label(raw)
            return label

        except Exception as e:
            logging.warning(f"[judge] attempt {attempt}/{MAX_RETRIES} failed: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)

    logging.error("[judge] all retries exhausted — returning empty label")
    return ""

# ── Label parser ───────────────────────────────────────────────────────────────

def parse_label(raw: str) -> str:
    """Extract clean label from bracketed model output e.g. [Fully Correct]."""
    import re
    m = re.search(r"\[([^\]]+)\]", raw)
    if m:
        candidate = m.group(1).strip()
        if candidate in VALID_LABELS:
            return candidate
        # try case-insensitive match
        for label in VALID_LABELS:
            if candidate.lower() == label.lower():
                return label

    # fallback: check if raw text contains a valid label directly
    for label in VALID_LABELS:
        if label.lower() in raw.lower():
            return label

    logging.warning(f"[judge] could not parse label from: {repr(raw)}")
    return ""

# ── Aggregate counts ───────────────────────────────────────────────────────────

def compute_judge_metrics(labels: list) -> dict:
    total = len(labels)
    valid = [l for l in labels if l in VALID_LABELS]
    empty = total - len(valid)

    counts = {label: valid.count(label) for label in VALID_LABELS}
    percentages = {
        f"{label.lower().replace(' ', '_')}_pct": (
            round(counts[label] / len(valid) * 100, 2) if valid else 0.0
        )
        for label in VALID_LABELS
    }

    return {
        "judge_model": MODEL,
        "judge_total": total,
        "judge_valid": len(valid),
        "judge_empty": empty,
        "judge_fully_correct":     counts["Fully Correct"],
        "judge_partially_correct": counts["Partially Correct"],
        "judge_incorrect":         counts["Incorrect"],
        **percentages,
    }

# ── Main ───────────────────────────────────────────────────────────────────────

def run_judge(config_path: str):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s|%(levelname)s|%(message)s",
    )

    config = load_config(config_path)

    # validate task type
    task_type = config.get("task", {}).get("type", "")
    if task_type != "answer_generation":
        raise ValueError(
            f"run_judge.py is only for answer_generation tasks, got task.type={task_type!r}"
        )

    # paths
    predictions_path = Path(config["output"]["predictions_path"])
    metrics_path     = Path(config["output"]["metrics_path"])

    if not predictions_path.exists():
        raise FileNotFoundError(
            f"predictions CSV not found: {predictions_path}\n"
            f"Run run_evaluation.py first."
        )

    # load data
    df = pd.read_csv(predictions_path)
    logging.info(f"loaded {len(df)} rows from {predictions_path}")

    required_cols = {"input", "prediction", "ground_truth"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"predictions CSV missing columns: {missing}")

    # load prompt
    system_prompt, user_template = load_judge_prompt(config)
    logging.info(f"judge prompt loaded from {config['judge']['prompt_path']}")

    # init client
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY environment variable not set")
    client = OpenAI(api_key=api_key)

    # run judge
    labels = []
    total = len(df)

    for i, row in df.iterrows():
        question_stem    = str(row["input"]).strip()
        reference_answer = str(row["ground_truth"]).strip()
        generated_answer = str(row["prediction"]).strip()

        logging.info(f"[{i+1}/{total}] judging id={row.get('id', i)}")

        label = call_judge(
            client=client,
            system_prompt=system_prompt,
            user_template=user_template,
            question_stem=question_stem,
            reference_answer=reference_answer,
            generated_answer=generated_answer,
        )

        labels.append(label)
        logging.info(f"  → {label!r}")

        if i < total - 1:
            time.sleep(DELAY_SECS)

    # add column to CSV
    df["judge_label"] = labels
    df.to_csv(predictions_path, index=False, encoding="utf-8")
    logging.info(f"saved predictions with judge labels to {predictions_path}")

    # compute aggregate metrics
    judge_metrics = compute_judge_metrics(labels)

    # append to existing metrics JSON
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            existing_metrics = json.load(f)
    else:
        existing_metrics = {}
        metrics_path.parent.mkdir(parents=True, exist_ok=True)

    existing_metrics.update(judge_metrics)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(existing_metrics, f, indent=4, ensure_ascii=False)

    logging.info(f"judge metrics appended to {metrics_path}")
    logging.info(json.dumps(judge_metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python scripts/run_judge.py <config.yaml>")
        sys.exit(2)
    run_judge(sys.argv[1])