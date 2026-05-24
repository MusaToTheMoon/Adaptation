#!/usr/bin/env python3
"""Recompute BERTScore for one or many predictions CSV files.

This script reuses the metric helpers in ``evals.metrics`` and defaults to:
- model_type: aubmindlab/bert-large-arabertv02
- num_layers: 18

By default, it writes per-example BERTScore columns to copied predictions CSV
files and writes aggregate BERTScore values into copied metrics JSON files.
Use --in-place to overwrite original prediction/metrics files.
"""

import argparse
import json
import re
import sys
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Ensure imports like `from evals.metrics import ...` work when invoked as
# `python scripts/recompute_bertscore_arabert.py`.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evals.metrics import (
    DEFAULT_BERT_MODEL,
    DEFAULT_BERT_NUM_LAYERS,
    calculate_bert_score,
    calculate_bert_score_per_example,
)

try:
    import torch
except Exception:
    torch = None

try:
    from transformers import AutoModel, AutoTokenizer
except Exception:
    AutoModel = None
    AutoTokenizer = None


def strip_answer_prefix(text: Any) -> str:
    if text is None:
        return ""
    text = str(text).strip()
    text = re.sub(r"^\s*ANSWER\s*[:=：]\s*", "", text, flags=re.IGNORECASE)
    return text.strip()


def is_nonempty_text(text: str) -> bool:
    return bool(text and text.strip())


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        if torch is None:
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda":
        if torch is None or not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested, but CUDA is not available.")
    return device_arg


def contains_glob_pattern(value: str) -> bool:
    return any(ch in value for ch in ["*", "?", "["])


def resolve_prediction_files(inputs: List[str], recursive: bool) -> List[Path]:
    files: List[Path] = []

    for item in inputs:
        if contains_glob_pattern(item):
            matched = [Path(p) for p in glob(item, recursive=True)]
            files.extend([p for p in matched if p.is_file() and p.suffix.lower() == ".csv"])
            continue

        p = Path(item)
        if p.is_dir():
            pattern = "**/*.csv" if recursive else "*.csv"
            files.extend(sorted(p.glob(pattern)))
            continue

        if p.is_file() and p.suffix.lower() == ".csv":
            files.append(p)
            continue

        raise FileNotFoundError(f"Input is not a CSV file/dir/glob match: {item}")

    # Dedupe while preserving stable order.
    seen = set()
    deduped: List[Path] = []
    for p in files:
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        deduped.append(rp)

    return deduped


def infer_metrics_path(predictions_path: Path, metrics_root: Optional[Path]) -> Path:
    if metrics_root is not None:
        return (metrics_root / predictions_path.name).with_suffix(".json")

    parts = list(predictions_path.parts)
    if "predictions" in parts:
        idx = parts.index("predictions")
        parts[idx] = "metrics"
        return Path(*parts).with_suffix(".json")

    return predictions_path.with_suffix(".json")


def load_existing_metrics(metrics_path: Path) -> Dict[str, Any]:
    if not metrics_path.exists():
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        return {}

    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Existing metrics JSON must be an object: {metrics_path}")
    return data


def update_bertscore_metrics_only(
    metrics: Dict[str, Any],
    bert_precision: float,
    bert_recall: float,
    bert_f1: float,
    model_type: str,
    num_layers: int,
    rescale_with_baseline: bool,
) -> Dict[str, Any]:
    """Update only BERTScore-related fields, preserving all other metrics keys.

    This is intentionally non-destructive for unrelated metrics such as judge_*
    fields that may already exist in the same JSON.
    """
    metrics["bert_precision"] = bert_precision
    metrics["bert_recall"] = bert_recall
    metrics["bert_f1"] = bert_f1
    metrics["bert_model"] = model_type
    metrics["bert_num_layers"] = int(num_layers)
    metrics["bert_rescaled"] = bool(rescale_with_baseline)
    return metrics


def prefetch_hf_model(model_type: str, cache_dir: Optional[str], force_download: bool) -> None:
    if AutoTokenizer is None or AutoModel is None:
        raise RuntimeError(
            "transformers is not available. Install dependencies first: pip install -r requirements.txt"
        )

    print(f"[prefetch] Downloading tokenizer for {model_type} ...")
    AutoTokenizer.from_pretrained(
        model_type,
        cache_dir=cache_dir,
        force_download=force_download,
    )
    print(f"[prefetch] Downloading model weights for {model_type} ...")
    AutoModel.from_pretrained(
        model_type,
        cache_dir=cache_dir,
        force_download=force_download,
    )
    print("[prefetch] Model cached successfully.")


def recompute_for_file(
    predictions_path: Path,
    metrics_path: Path,
    lang: str,
    device: str,
    model_type: str,
    num_layers: int,
    rescale_with_baseline: bool,
    dry_run: bool,
    output_predictions_path: Optional[Path] = None,
    output_metrics_path: Optional[Path] = None,
) -> Dict[str, Any]:
    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions CSV not found: {predictions_path}")

    df = pd.read_csv(predictions_path)

    has_pred_clean = "prediction_clean" in df.columns
    has_gt_clean = "ground_truth_clean" in df.columns
    has_pred_raw = "prediction" in df.columns
    has_gt_raw = "ground_truth" in df.columns

    if not (has_pred_clean or has_pred_raw):
        raise ValueError("CSV must contain 'prediction_clean' or 'prediction' column")
    if not (has_gt_clean or has_gt_raw):
        raise ValueError("CSV must contain 'ground_truth_clean' or 'ground_truth' column")
    if len(df) == 0:
        raise ValueError("Predictions CSV has no rows")

    if has_pred_raw:
        preds = df["prediction"].fillna("").astype(str).tolist()
        preds_clean = [strip_answer_prefix(p) for p in preds]
    else:
        preds_clean = df["prediction_clean"].fillna("").astype(str).tolist()
        preds_clean = [strip_answer_prefix(p) for p in preds_clean]

    if has_gt_raw:
        refs = df["ground_truth"].fillna("").astype(str).tolist()
        refs_clean = [strip_answer_prefix(r) for r in refs]
    else:
        refs_clean = df["ground_truth_clean"].fillna("").astype(str).tolist()
        refs_clean = [strip_answer_prefix(r) for r in refs_clean]

    valid_mask = np.array(
        [is_nonempty_text(p) and is_nonempty_text(r) for p, r in zip(preds_clean, refs_clean)],
        dtype=bool,
    )
    valid_indices = np.where(valid_mask)[0]
    invalid_count = int(len(df) - len(valid_indices))

    p = np.zeros(len(df), dtype=np.float32)
    r = np.zeros(len(df), dtype=np.float32)
    f1 = np.zeros(len(df), dtype=np.float32)

    if len(valid_indices) > 0:
        preds_valid = [preds_clean[i] for i in valid_indices]
        refs_valid = [refs_clean[i] for i in valid_indices]

        rows = calculate_bert_score_per_example(
            predictions=preds_valid,
            references=refs_valid,
            lang=lang,
            model_type=model_type,
            num_layers=num_layers,
            device=device,
            rescale_with_baseline=rescale_with_baseline,
        )

        if rows is None:
            raise RuntimeError(
                "calculate_bert_score_per_example returned None. "
                "Check bert-score/transformers installation and model availability."
            )

        if len(rows) != len(valid_indices):
            raise RuntimeError(
                "Length mismatch after BERTScore on valid rows: "
                f"valid_rows={len(valid_indices)} rows={len(rows)}"
            )

        for out_idx, row in zip(valid_indices, rows):
            p[out_idx] = float(row["bert_precision"])
            r[out_idx] = float(row["bert_recall"])
            f1[out_idx] = float(row["bert_f1"])

    if invalid_count > 0:
        print(
            f"[warn] {predictions_path.name}: {invalid_count} rows have empty prediction/reference; "
            "assigned 0.0 BERTScore for those rows."
        )

    df["prediction_clean"] = preds_clean
    df["ground_truth_clean"] = refs_clean
    df["bert_precision_example"] = p.tolist()
    df["bert_recall_example"] = r.tolist()
    df["bert_f1_example"] = f1.tolist()

    if "judge_label" in df.columns:
        ordered_columns = [col for col in df.columns if col != "judge_label"] + ["judge_label"]
        df = df[ordered_columns]

    bert = calculate_bert_score(
        predictions=preds_clean,
        references=refs_clean,
        lang=lang,
        model_type=model_type,
        num_layers=num_layers,
        device=device,
        rescale_with_baseline=rescale_with_baseline,
    )
    if bert is None:
        raise RuntimeError(
            "calculate_bert_score returned None. "
            "Check bert-score/transformers installation and model availability."
        )

    bert_precision = float(bert["bert_precision"])
    bert_recall = float(bert["bert_recall"])
    bert_f1 = float(bert["bert_f1"])

    metrics = load_existing_metrics(metrics_path)
    metrics = update_bertscore_metrics_only(
        metrics=metrics,
        bert_precision=bert_precision,
        bert_recall=bert_recall,
        bert_f1=bert_f1,
        model_type=model_type,
        num_layers=num_layers,
        rescale_with_baseline=rescale_with_baseline,
    )

    write_predictions_path = output_predictions_path or predictions_path
    write_metrics_path = output_metrics_path or metrics_path

    if not dry_run:
        write_predictions_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(write_predictions_path, index=False, encoding="utf-8")
        write_metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(write_metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4, ensure_ascii=False)

    return {
        "predictions": str(write_predictions_path),
        "predictions_source": str(predictions_path),
        "metrics": str(write_metrics_path),
        "metrics_source": str(metrics_path),
        "rows": int(len(df)),
        "empty_rows": invalid_count,
        "bert_precision": bert_precision,
        "bert_recall": bert_recall,
        "bert_f1": bert_f1,
        "bert_model": model_type,
        "bert_num_layers": int(num_layers),
        "bert_rescaled": bool(rescale_with_baseline),
        "dry_run": bool(dry_run),
    }


def make_copy_path(path: Path, copy_suffix: str) -> Path:
    return path.with_name(f"{path.stem}{copy_suffix}{path.suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Batch recompute BERTScore using evals.metrics helpers. "
            "Defaults to AraBERT large v02 with num_layers=18."
        )
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="CSV files, directories, and/or glob patterns",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="When an input is a directory, include CSV files recursively.",
    )
    parser.add_argument(
        "--metrics-root",
        default=None,
        help=(
            "Optional output folder for metrics JSON files. "
            "When omitted, path is inferred by replacing '/predictions/' with '/metrics/'."
        ),
    )
    parser.add_argument("--lang", default="ar", help="Language passed to BERTScore (default: ar)")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Execution device for BERTScore (default: auto)",
    )
    parser.add_argument(
        "--model-type",
        default=DEFAULT_BERT_MODEL,
        help=f"HF model id for BERTScore (default: {DEFAULT_BERT_MODEL})",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=DEFAULT_BERT_NUM_LAYERS,
        help=f"Model layer to use for BERTScore (default: {DEFAULT_BERT_NUM_LAYERS})",
    )
    parser.add_argument(
        "--rescale-with-baseline",
        action="store_true",
        help="Enable bert-score rescale_with_baseline (default: disabled)",
    )
    parser.add_argument(
        "--prefetch-model",
        action="store_true",
        help="Download/cache the HF tokenizer + model before scoring.",
    )
    parser.add_argument(
        "--prefetch-only",
        action="store_true",
        help="Only download/cache model and exit.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional Hugging Face cache dir for model prefetch.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force fresh download when prefetching model.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and print results without writing CSV/JSON files.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite original predictions CSV and metrics JSON files.",
    )
    parser.add_argument(
        "--copy-suffix",
        default="_recomputed",
        help="Suffix used for copied predictions/metrics file names in copy mode.",
    )

    args = parser.parse_args()

    device = resolve_device(args.device)
    model_type = args.model_type

    if args.prefetch_model or args.prefetch_only:
        prefetch_hf_model(
            model_type=model_type,
            cache_dir=args.cache_dir,
            force_download=args.force_download,
        )
    if args.prefetch_only:
        return

    prediction_files = resolve_prediction_files(args.inputs, recursive=args.recursive)
    if not prediction_files:
        raise RuntimeError("No CSV files found from --inputs")

    metrics_root = Path(args.metrics_root).resolve() if args.metrics_root else None

    results: List[Dict[str, Any]] = []
    for predictions_path in prediction_files:
        metrics_path = infer_metrics_path(predictions_path, metrics_root)
        output_predictions_path = predictions_path
        output_metrics_path = metrics_path
        if not args.in_place:
            output_predictions_path = make_copy_path(predictions_path, args.copy_suffix)
            output_metrics_path = make_copy_path(metrics_path, args.copy_suffix)
        out = recompute_for_file(
            predictions_path=predictions_path,
            metrics_path=metrics_path,
            lang=args.lang,
            device=device,
            model_type=model_type,
            num_layers=args.num_layers,
            rescale_with_baseline=args.rescale_with_baseline,
            dry_run=args.dry_run,
            output_predictions_path=output_predictions_path,
            output_metrics_path=output_metrics_path,
        )
        results.append(out)
        print(json.dumps(out, ensure_ascii=False))

    macro_f1 = float(np.mean([row["bert_f1"] for row in results]))
    print(
        json.dumps(
            {
                "files": len(results),
                "device": device,
                "lang": args.lang,
                "bert_model": model_type,
                "bert_num_layers": int(args.num_layers),
                "bert_rescaled": bool(args.rescale_with_baseline),
                "macro_bert_f1_across_files": macro_f1,
                "dry_run": bool(args.dry_run),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
