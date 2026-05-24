from __future__ import annotations

import argparse
import json
import csv
from pathlib import Path
from typing import Any, Iterable


def count_options_from_example(example: dict[str, Any]) -> int:
    """
    Tries common field names used for MCQ answer choices.

    Supported example formats:
    - {"options": [...], ...}
    - {"choices": [...], ...}
    - {"answers": [...], ...}   # only if this field stores candidate answers
    - {"A": "...", "B": "...", "C": "...", "D": "..."} style
    - {"opa": "...", "opb": "...", "opc": "..."} style
    """

    # # Common list-style fields
    # for key in ["options", "choices", "candidates", "answers"]:
    #     if key in example and isinstance(example[key], list):
    #         return len(example[key])

    # # Lettered choice fields, e.g. A/B/C/D or option_a/.../option_d
    # letter_keys = [k for k in example.keys() if k in ["A", "B", "C", "D", "E", "F", "G", "H"]]
    # if letter_keys:
    #     return len(letter_keys)

    # Compact option keys, e.g. opa/opb/opc/... (case-insensitive)
    op_letter_keys = [
        k for k in example.keys()
        if len(k) == 3 and k[:2].lower() == "op" and k[2].isalpha()
    ]
    if op_letter_keys:
        return len(op_letter_keys)

    # option_prefix_keys = [
    #     k for k in example.keys()
    #     if k.lower().startswith("option_") or k.lower().startswith("choice_")
    # ]
    # if option_prefix_keys:
    #     return len(option_prefix_keys)

    raise ValueError(f"Could not infer options from example keys: {list(example.keys())}")


def random_baseline_from_examples(examples: Iterable[dict[str, Any]]) -> float:
    """
    Computes expected accuracy (%) for uniform random guessing.
    """
    probs = []
    for ex in examples:
        k = count_options_from_example(ex)
        if k <= 0:
            raise ValueError(f"Invalid number of options: {k}")
        probs.append(1.0 / k)

    if not probs:
        raise ValueError("Dataset is empty.")

    return 100.0 * sum(probs) / len(probs)


def load_json(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Supports either a plain list of examples or a dict containing a split
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        # Try common split names
        for split in ["test", "validation", "dev", "train"]:
            if split in data and isinstance(data[split], list):
                return data[split]

    raise ValueError("Unsupported JSON structure.")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_csv(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def load_dataset(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".json":
        return load_json(path)
    if suffix == ".jsonl":
        return load_jsonl(path)
    if suffix == ".csv":
        return load_csv(path)

    raise ValueError(f"Unsupported file format: {suffix}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute expected random-guess baseline for an MCQ dataset."
    )
    parser.add_argument(
        "dataset_path",
        type=Path,
        help="Path to dataset JSON file.",
    )
    args = parser.parse_args()

    dataset_path = args.dataset_path
    if dataset_path.suffix.lower() != ".json":
        raise ValueError("dataset_path must point to a .json file.")

    examples = load_json(dataset_path)
    baseline = random_baseline_from_examples(examples)
    print(f"Random baseline: {baseline:.2f}%")