"""Sample 100 random rows from Task 2 generated answers and save to test.json.

Usage:
    python scripts/sample_random_subset.py
    python scripts/sample_random_subset.py --seed 42
    python scripts/sample_random_subset.py --n 100 --input datasets/answer-gen/task2_complete.json --output test.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Randomly sample records from task2_complete.json into test.json."
    )
    parser.add_argument(
        "--input",
        default="datasets/answer-gen/task2_complete.json",
        help="Path to input JSON list (default: datasets/answer-gen/task2_complete.json).",
    )
    parser.add_argument(
        "--output",
        default="test.json",
        help="Path to output JSON file (default: test.json).",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=100,
        help="Number of samples to draw (default: 100).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Optional random seed for reproducible sampling.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if args.n <= 0:
        raise ValueError(f"--n must be > 0, got {args.n}")

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with input_path.open("r", encoding="utf-8") as f:
        data: Any = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of records.")

    if len(data) < args.n:
        raise ValueError(
            f"Cannot sample {args.n} rows from dataset of size {len(data)}."
        )

    rng = random.Random(args.seed)
    sampled = rng.sample(data, args.n)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(sampled, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(sampled)} samples to {output_path}")

if __name__ == "__main__":
    main()