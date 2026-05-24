"""
Create a fixed few-shot split for Task 1 (MCQ):
- conditioning set (k examples)
- inference set (remaining N-k examples)
- metadata for full reproducibility

Usage example:
  python scripts/create_fewshot_split.py \
      --dataset datasets/qa/test.json \
      --k 5 \
      --seed 42
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple


LETTER_SET = {"A", "B", "C", "D", "E", "F"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create fixed few-shot split for Task1 MCQ datasets.")
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to source MCQ dataset JSON (list of dict records).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of conditioning examples (default: 5).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used to pick conditioning examples (default: 42).",
    )
    parser.add_argument(
        "--output-root",
        default="datasets/qa/fewshot",
        help="Root directory for generated splits (default: datasets/qa/fewshot).",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Optional dataset label in output path (default: source filename stem).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output files if they already exist.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def is_valid_mcq_item(item: Dict[str, Any]) -> bool:
    stem = str(item.get("question") or "").strip()
    if not stem:
        return False

    for key in ("opa", "opb", "opc", "opd"):
        val = str(item.get(key) or "").strip()
        if not val:
            return False

    answer = str(item.get("answer") or "").strip().upper()
    return answer in LETTER_SET


def get_item_id(item: Dict[str, Any], fallback_idx: int) -> str:
    return str(item.get("id", fallback_idx))


def split_dataset(dataset: List[Dict[str, Any]], k: int, seed: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    valid_records: List[Tuple[int, Dict[str, Any]]] = []
    invalid_records: List[Tuple[int, str]] = []

    for idx, item in enumerate(dataset):
        if not isinstance(item, dict):
            invalid_records.append((idx, "item is not a dict"))
            continue
        if is_valid_mcq_item(item):
            valid_records.append((idx, item))
        else:
            invalid_records.append((idx, "missing/invalid MCQ fields"))

    if len(valid_records) < k:
        raise ValueError(
            f"Not enough valid MCQ items for k={k}. "
            f"valid={len(valid_records)} total={len(dataset)}"
        )

    rng = random.Random(seed)
    positions = list(range(len(valid_records)))
    rng.shuffle(positions)
    cond_pos = set(positions[:k])

    conditioning_source_indices: List[int] = []
    for pos, (src_idx, _item) in enumerate(valid_records):
        if pos in cond_pos:
            conditioning_source_indices.append(src_idx)

    conditioning_source_index_set = set(conditioning_source_indices)
    conditioning: List[Dict[str, Any]] = []
    inference: List[Dict[str, Any]] = []

    # Keep ALL non-conditioning rows in inference so inference size is exactly N-k.
    for src_idx, item in enumerate(dataset):
        if src_idx in conditioning_source_index_set:
            conditioning.append(item)
        else:
            inference.append(item)

    duplicate_ids: List[str] = []
    id_seen = set()
    for src_idx, item in valid_records:
        item_id = get_item_id(item, src_idx)
        if item_id in id_seen:
            duplicate_ids.append(item_id)
        id_seen.add(item_id)

    details = {
        "total_items": len(dataset),
        "valid_items": len(valid_records),
        "invalid_items": len(invalid_records),
        "invalid_item_indices": [idx for idx, _ in invalid_records],
        "conditioning_source_indices": conditioning_source_indices,
        "conditioning_ids": [get_item_id(dataset[idx], idx) for idx in conditioning_source_indices],
        "inference_ids": [get_item_id(item, idx) for idx, item in enumerate(inference)],
        "inference_keeps_invalid_items": True,
        "valid_conditioning_pool_size": len(valid_records),
        "duplicate_ids_detected": sorted(set(duplicate_ids)),
    }

    return conditioning, inference, details


def ensure_writable(path: Path, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing file: {path}. Use --force to overwrite.")


def main() -> None:
    args = parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    if args.k <= 0:
        raise ValueError(f"k must be > 0, got {args.k}")

    with dataset_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Dataset JSON must be a list of records.")

    conditioning, inference, details = split_dataset(data, k=args.k, seed=args.seed)

    dataset_name = args.name if args.name else dataset_path.stem
    split_dir = Path(args.output_root) / dataset_name / f"k{args.k}_seed{args.seed}"
    split_dir.mkdir(parents=True, exist_ok=True)

    conditioning_path = split_dir / "conditioning.json"
    inference_path = split_dir / "inference.json"
    metadata_path = split_dir / "metadata.json"

    ensure_writable(conditioning_path, args.force)
    ensure_writable(inference_path, args.force)
    ensure_writable(metadata_path, args.force)

    metadata = {
        "task": "task1_mcq",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_dataset_path": str(dataset_path),
        "source_dataset_sha256": sha256_file(dataset_path),
        "split": {
            "k": args.k,
            "seed": args.seed,
            "conditioning_count": len(conditioning),
            "inference_count": len(inference),
        },
        "stats": details,
        "artifacts": {
            "conditioning_path": str(conditioning_path),
            "inference_path": str(inference_path),
            "metadata_path": str(metadata_path),
        },
    }

    with conditioning_path.open("w", encoding="utf-8") as f:
        json.dump(conditioning, f, ensure_ascii=False, indent=2)

    with inference_path.open("w", encoding="utf-8") as f:
        json.dump(inference, f, ensure_ascii=False, indent=2)

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("Few-shot split created successfully.")
    print(f"  split_dir: {split_dir}")
    print(f"  conditioning: {conditioning_path}")
    print(f"  inference: {inference_path}")
    print(f"  metadata: {metadata_path}")
    print(
        "  counts: "
        f"valid={details['valid_items']} conditioning={len(conditioning)} inference={len(inference)}"
    )


if __name__ == "__main__":
    main()
