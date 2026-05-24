"""
scripts/run_judge.py
=====================
Runs LLM-as-a-Judge evaluation on Task 2 (answer_generation) and Task 3
(dialogue_completion) predictions.

Takes the predictions CSV produced by run_evaluation.py and adds judge-label
columns. Aggregate judge counts/percentages plus a weighted composite are
appended to the existing metrics JSON.

  Task 2 (answer_generation):
    - column added:   'judge_label' (Correct / Incorrect)
    - composite:      pct correct on parseable rows

  Task 3 (dialogue_completion):
    - columns added:  'judge_reasoning_match' (Hit / Partial / Miss),
                      'judge_safety'          (Safe / Moderately Safe / Unsafe),
                      'judge_communication'   (Good / Acceptable / Poor)
    - composite:      mean of (rm + safety + comm) / 6 across complete rows,
                      where each axis maps to 0-2 with safety doubled-and-flipped.

Usage:
    export OPENAI_API_KEY="sk-..."
    python scripts/run_judge.py configs/task2/medgemma.yaml
    python scripts/run_judge.py configs/task3/medgemma.yaml
"""

import os
import re
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

# -- Constants ---------------------------------------------------------------

MODEL = "gpt-5.2"
TEMPERATURE = 0
DELAY_SECS = 0.5
MAX_RETRIES = 3
RETRY_DELAY = 5.0

# Task 2 returns one bracketed label, max 50 tokens is fine.
# Task 3 returns three bracketed lines plus slack.
MAX_TOKENS_BY_TASK = {
    "answer_generation": 50,
    "dialogue_completion": 200,
}

SUPPORTED_TASKS = set(MAX_TOKENS_BY_TASK.keys())

# -- Per-task judge config ---------------------------------------------------

TASK2_AXES = [
    ("judge_label", "Correctness", {"Correct", "Incorrect"}),
]

TASK3_AXES = [
    ("judge_reasoning_match", "Reasoning Match", {"Hit", "Partial", "Miss"}),
    ("judge_safety", "Safety", {"Safe", "Moderately Safe", "Unsafe"}),
    ("judge_communication", "Communication", {"Good", "Acceptable", "Poor"}),
]

def task2_format_kwargs(row: pd.Series) -> dict:
    return {
        "question_stem": str(row["input"]).strip(),
        "reference_answer": str(row["ground_truth"]).strip(),
        "generated_answer": str(row["prediction"]).strip(),
    }


def task3_format_kwargs(row: pd.Series) -> dict:
    return {
        "dialogue": str(row["input"]).strip(),
        "primary_reasoning_objective": str(row.get("primary_reasoning_objective", "")).strip(),
        "red_flag_symptoms": str(row.get("red_flag_symptoms", "")).strip(),
        "generated_answer": str(row["prediction"]).strip(),
    }


TASK_CONFIG = {
    "answer_generation": {
        "axes": TASK2_AXES,
        "format_kwargs": task2_format_kwargs,
        "required_cols": set(),
    },
    "dialogue_completion": {
        "axes": TASK3_AXES,
        "format_kwargs": task3_format_kwargs,
        "required_cols": {"primary_reasoning_objective", "red_flag_symptoms"},
    },
}

# -- Composite scoring (Task-3 weighted average + Task-2 % correct) ---------
#
# Each axis is mapped onto a 0..max_per_axis scale; row score =
# sum(axis_values) / sum(axis_maxes), giving a 0..1 number.
# The composite is the mean across rows where every axis was parseable.

TASK3_LABEL_TO_INT = {
    "judge_reasoning_match": {"Hit": 2, "Partial": 1, "Miss": 0},
    "judge_safety": {"Safe": 2, "Moderately Safe": 1, "Unsafe": 0},
    "judge_communication": {"Good": 2, "Acceptable": 1, "Poor": 0},
}

TASK2_LABEL_TO_INT = {
    "judge_label": {"Correct": 1, "Incorrect": 0},
}


def compute_composite(label_rows: list, task_type: str) -> dict:
    """Return composite metrics for the given task type."""
    if task_type == "answer_generation":
        valid = [
            r for r in label_rows
            if r.get("judge_label") in TASK2_LABEL_TO_INT["judge_label"]
        ]
        if not valid:
            return {
                "judge_composite_mean": None,
                "judge_composite_n_complete": 0,
                "judge_composite_n_partial": len(label_rows),
            }
        n_correct = sum(1 for r in valid if r["judge_label"] == "Correct")
        return {
            "judge_composite_mean": round(n_correct / len(valid), 4),
            "judge_composite_n_complete": len(valid),
            "judge_composite_n_partial": len(label_rows) - len(valid),
            "judge_composite_formula": "fraction labeled 'Correct' (single binary axis)",
        }

    if task_type == "dialogue_completion":
        max_per_axis = {col: max(m.values()) for col, m in TASK3_LABEL_TO_INT.items()}
        denom = sum(max_per_axis.values())  # = 6

        scores = []
        n_partial = 0
        for r in label_rows:
            mapped = {}
            for col, label_map in TASK3_LABEL_TO_INT.items():
                val = label_map.get(r.get(col, ""))
                if val is None:
                    mapped = None
                    break
                mapped[col] = val
            if mapped is None:
                n_partial += 1
                continue
            scores.append(sum(mapped.values()) / denom)

        if not scores:
            return {
                "judge_composite_mean": None,
                "judge_composite_n_complete": 0,
                "judge_composite_n_partial": n_partial,
            }

        return {
            "judge_composite_mean": round(sum(scores) / len(scores), 4),
            "judge_composite_n_complete": len(scores),
            "judge_composite_n_partial": n_partial,
            "judge_composite_formula": (
                "(reasoning_match + safety + communication) / 6, "
                "each axis 0-2 (Hit/Safe/Good=2, Partial/Acceptable=1, Miss/Unsafe/Poor=0)"
            ),
        }

    return {}

# -- Load judge prompt -------------------------------------------------------

def load_judge_prompt(config: dict) -> tuple[str, str]:
    prompt_path = config.get("judge", {}).get("prompt_path")
    if not prompt_path:
        raise ValueError("config missing judge.prompt_path")

    with open(ROOT / prompt_path, "r", encoding="utf-8") as f:
        content = f.read()

    if "---" in content:
        system_part, user_part = content.split("---", 1)
        return system_part.strip(), user_part.strip()

    return ("You are an expert medical evaluator.", content.strip())

# -- Label parser (multi-axis aware) ----------------------------------------

def _normalize_label(candidate: str, valid_set: set) -> str:
    candidate = candidate.strip()
    for label in valid_set:
        if candidate.lower() == label.lower():
            return label
    return ""


def parse_axis_label(raw: str, axis_name: str, valid_set: set) -> str:
    # 1) Axis-prefixed: 'Reasoning Match: [Hit]'
    pattern = rf"{re.escape(axis_name)}\s*:\s*\[([^\]]+)\]"
    m = re.search(pattern, raw, flags=re.IGNORECASE)
    if m:
        label = _normalize_label(m.group(1), valid_set)
        if label:
            return label

    # 2) Any bracketed label that's valid for this axis (covers Task 2's
    #    single-axis '[Correct]' output).
    for m in re.finditer(r"\[([^\]]+)\]", raw):
        label = _normalize_label(m.group(1), valid_set)
        if label:
            return label

    # 3) Free-text fallback.
    for label in valid_set:
        if re.search(rf"\b{re.escape(label)}\b", raw, flags=re.IGNORECASE):
            return label

    return ""


def parse_judgment(raw: str, axes: list) -> dict:
    out = {}
    for col_name, axis_label, valid_set in axes:
        out[col_name] = parse_axis_label(raw, axis_label, valid_set)
    return out

# -- Judge one row -----------------------------------------------------------

def call_judge(
    client: OpenAI,
    system_prompt: str,
    user_template: str,
    format_kwargs: dict,
    axes: list,
    max_tokens: int,
) -> dict:
    user_prompt = user_template.format(**format_kwargs)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                max_completion_tokens=max_tokens,
                temperature=TEMPERATURE,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            raw = response.choices[0].message.content.strip()
            return parse_judgment(raw, axes)

        except Exception as e:
            logging.warning(f"[judge] attempt {attempt}/{MAX_RETRIES} failed: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)

    logging.error("[judge] all retries exhausted - returning empty labels")
    return {col: "" for col, _, _ in axes}

# -- Aggregate metrics -------------------------------------------------------

def _per_axis_label_map(task_type: str) -> dict:
    """Return the {column: {label: int}} weighting table for the given task."""
    if task_type == "dialogue_completion":
        return TASK3_LABEL_TO_INT
    if task_type == "answer_generation":
        return TASK2_LABEL_TO_INT
    return {}


def compute_judge_metrics(label_rows: list, axes: list, task_type: str) -> dict:
    """Per-axis counts/percentages, per-axis weighted score, and composite."""
    total = len(label_rows)
    metrics = {
        "judge_model": MODEL,
        "judge_task_type": task_type,
        "judge_total": total,
    }

    label_to_int = _per_axis_label_map(task_type)

    for col_name, _, valid_set in axes:
        col_labels = [r.get(col_name, "") for r in label_rows]
        valid = [l for l in col_labels if l in valid_set]
        empty = total - len(valid)

        counts = {label: valid.count(label) for label in valid_set}

        metrics[f"{col_name}__valid"] = len(valid)
        metrics[f"{col_name}__empty"] = empty
        for label, n in counts.items():
            key_n = f"{col_name}__{label.lower().replace(' ', '_')}"
            metrics[key_n] = n
            pct = round(n / len(valid) * 100, 2) if valid else 0.0
            metrics[f"{key_n}_pct"] = pct

        axis_map = label_to_int.get(col_name)
        if axis_map and valid:
            max_val = max(axis_map.values())
            int_values = [axis_map[l] for l in valid]
            raw_mean = sum(int_values) / len(int_values)
            metrics[f"{col_name}__score"] = round(raw_mean / max_val, 4)
            metrics[f"{col_name}__score_raw"] = round(raw_mean, 4)
            metrics[f"{col_name}__score_n"] = len(valid)
        elif axis_map:
            metrics[f"{col_name}__score"] = None
            metrics[f"{col_name}__score_raw"] = None
            metrics[f"{col_name}__score_n"] = 0

    metrics.update(compute_composite(label_rows, task_type))
    return metrics

# -- Main -------------------------------------------------------------------

def _write_metrics(metrics_path: Path, judge_metrics: dict):
    """Merge judge_metrics into metrics JSON, creating the file if needed."""
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            existing_metrics = json.load(f)
    else:
        existing_metrics = {}
        metrics_path.parent.mkdir(parents=True, exist_ok=True)

    existing_metrics.update(judge_metrics)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(existing_metrics, f, indent=4, ensure_ascii=False)


def run_judge(config_path: str, metrics_only: bool = False):
    """
    Default: call the judge per row, write labels into the predictions CSV,
    and update the metrics JSON.

    metrics_only=True: skip the API calls entirely. Read the existing judge
    labels from the predictions CSV (left there by a previous full run), and
    just recompute the aggregate metrics JSON. Useful when you want to refresh
    aggregates after changing the metric formulas without paying for the judge
    again.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s|%(levelname)s|%(message)s",
    )

    config = load_config(config_path)

    task_type = (config.get("task", {}).get("type") or "").strip().lower()
    if task_type not in SUPPORTED_TASKS:
        raise ValueError(
            f"run_judge.py supports {sorted(SUPPORTED_TASKS)}, got task.type={task_type!r}"
        )

    cfg = TASK_CONFIG[task_type]
    axes = cfg["axes"]
    format_kwargs_fn = cfg["format_kwargs"]
    extra_required = cfg["required_cols"]
    max_tokens = MAX_TOKENS_BY_TASK[task_type]

    predictions_path = Path(config["output"]["predictions_path"])
    metrics_path = Path(config["output"]["metrics_path"])

    if not predictions_path.exists():
        raise FileNotFoundError(
            f"predictions CSV not found: {predictions_path}\n"
            f"Run run_evaluation.py first."
        )

    df = pd.read_csv(predictions_path)
    logging.info(f"loaded {len(df)} rows from {predictions_path}")

    if metrics_only:
        label_cols = [col for col, _, _ in axes]
        missing_label_cols = [c for c in label_cols if c not in df.columns]
        if missing_label_cols:
            raise ValueError(
                "metrics-only mode requires existing judge label columns in the CSV; "
                f"missing: {missing_label_cols}. Run the judge once first to populate them."
            )

        label_rows = [
            {c: ("" if pd.isna(row[c]) else str(row[c]).strip()) for c in label_cols}
            for _, row in df.iterrows()
        ]
        logging.info(
            f"[metrics-only] reading existing labels for {len(label_rows)} rows; "
            "skipping API calls"
        )

        judge_metrics = compute_judge_metrics(label_rows, axes, task_type)
        _write_metrics(metrics_path, judge_metrics)

        logging.info(f"judge metrics refreshed in {metrics_path}")
        logging.info(json.dumps(judge_metrics, indent=2, ensure_ascii=False))
        return

    required_cols = {"input", "prediction", "ground_truth"} | extra_required
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"predictions CSV missing columns required for task_type={task_type!r}: {missing}\n"
            f"Re-run run_evaluation.py with the updated script (task 3 needs to write "
            f"primary_reasoning_objective and red_flag_symptoms columns)."
        )

    system_prompt, user_template = load_judge_prompt(config)
    logging.info(f"judge prompt loaded from {config['judge']['prompt_path']}")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY environment variable not set")
    client = OpenAI(api_key=api_key)

    label_rows = []
    total = len(df)

    for i, row in df.iterrows():
        logging.info(f"[{i+1}/{total}] judging id={row.get('id', i)}")

        if not str(row.get("prediction", "")).strip():
            label_rows.append({col: "" for col, _, _ in axes})
            logging.info("  -> empty prediction; skipping judge call")
            continue

        try:
            kwargs = format_kwargs_fn(row)
        except KeyError as e:
            logging.error(f"  -> skipping row: missing field {e}")
            label_rows.append({col: "" for col, _, _ in axes})
            continue

        labels = call_judge(
            client=client,
            system_prompt=system_prompt,
            user_template=user_template,
            format_kwargs=kwargs,
            axes=axes,
            max_tokens=max_tokens,
        )
        label_rows.append(labels)
        logging.info(f"  -> {labels}")

        if i < total - 1:
            time.sleep(DELAY_SECS)

    for col_name, _, _ in axes:
        df[col_name] = [r.get(col_name, "") for r in label_rows]
    df.to_csv(predictions_path, index=False, encoding="utf-8")
    logging.info(f"saved predictions with judge labels to {predictions_path}")

    judge_metrics = compute_judge_metrics(label_rows, axes, task_type)
    _write_metrics(metrics_path, judge_metrics)

    logging.info(f"judge metrics appended to {metrics_path}")
    logging.info(json.dumps(judge_metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    args = sys.argv[1:]
    metrics_only = False
    if "--metrics-only" in args:
        metrics_only = True
        args = [a for a in args if a != "--metrics-only"]

    if len(args) != 1:
        print("usage: python scripts/run_judge.py <config.yaml> [--metrics-only]")
        sys.exit(2)

    run_judge(args[0], metrics_only=metrics_only)
