#!/usr/bin/env python3
"""One-time setup utility for Arabic BERTScore rescaling.

This script does three things:
1) Warms Hugging Face cache for the target model.
2) Builds an Arabic sentence pool from local files.
3) Computes and installs a BERTScore rescale baseline TSV at the exact
   path expected by bert_score.
"""

import argparse
import csv
import json
import os
import random
from pathlib import Path
from typing import Iterable, List

import pandas as pd
import torch
from tqdm.auto import tqdm

import bert_score
from transformers import AutoModel, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download HF model and generate/install Arabic BERTScore baseline TSV."
    )
    parser.add_argument(
        "--model-type",
        default="bert-base-multilingual-cased",
        help="HF model id for BERTScore (default: bert-base-multilingual-cased)",
    )
    parser.add_argument("--lang", default="ar", help="Language code (default: ar)")
    parser.add_argument(
        "--source-files",
        nargs="+",
        default=["datasets/train/train.json", "datasets/answer-gen/medarabenchv2.json"],
        help="Input files used to build sentence pool (json/jsonl/csv/txt)",
    )
    parser.add_argument(
        "--text-fields",
        default="question,answer_text,opa,opb,opc,opd,ope,opf",
        help="Comma-separated field names to extract from JSON/CSV rows",
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=200000,
        help="Maximum number of candidate lines to keep after filtering/dedup",
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=1,
        help="Minimum whitespace-separated words per line",
    )
    parser.add_argument(
        "--max-words",
        type=int,
        default=32,
        help="Maximum whitespace-separated words per line",
    )
    parser.add_argument(
        "--pairs",
        type=int,
        default=20000,
        help="Target number of random pairs for baseline computation",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Pairs per scorer iteration",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="BERTScore batch size inside each chunk",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Execution device for baseline generation",
    )
    parser.add_argument(
        "--baseline-path",
        default=None,
        help="Explicit output TSV path. If omitted, use bert_score default path.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip model/tokenizer cache warmup step",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing baseline TSV",
    )
    return parser.parse_args()


def normalize_text(value: str) -> str:
    text = "" if value is None else str(value)
    return " ".join(text.strip().split())


def keep_sentence(text: str, min_words: int, max_words: int) -> bool:
    if not text:
        return False
    wc = len(text.split())
    return min_words <= wc <= max_words


def iter_json_rows(path: Path) -> Iterable[dict]:
    if path.suffix.lower() == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if isinstance(item, dict):
                    yield item
        return

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                yield item
    elif isinstance(payload, dict):
        yield payload


def read_sentences_from_file(path: Path, text_fields: List[str]) -> List[str]:
    ext = path.suffix.lower()

    if ext in {".json", ".jsonl"}:
        out: List[str] = []
        for row in iter_json_rows(path):
            for key in text_fields:
                if key in row:
                    out.append(normalize_text(row.get(key)))
        return out

    if ext == ".csv":
        out: List[str] = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                for key in text_fields:
                    if key in row:
                        out.append(normalize_text(row.get(key)))
        return out

    if ext == ".txt":
        with open(path, "r", encoding="utf-8") as f:
            return [normalize_text(line) for line in f]

    raise ValueError(f"Unsupported input file extension: {path}")


def dedup_preserve_order(items: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is not available")
    return device_arg


def default_baseline_path(lang: str, model_type: str) -> Path:
    base_dir = Path(bert_score.__file__).resolve().parent
    return base_dir / "rescale_baseline" / lang / f"{model_type}.tsv"


def warm_model_cache(model_type: str) -> None:
    # Ensure we are online for the first-time download if cache is empty.
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "0"

    print(f"[setup] downloading/checking tokenizer: {model_type}")
    AutoTokenizer.from_pretrained(model_type)
    print(f"[setup] downloading/checking model: {model_type}")
    AutoModel.from_pretrained(model_type)
    print("[setup] Hugging Face cache warmup complete")


def make_random_pairs(lines: List[str], target_pairs: int, seed: int) -> tuple[list[str], list[str]]:
    if len(lines) < 2:
        raise RuntimeError("Need at least 2 usable lines to build random pairs")

    max_pairs = len(lines) // 2
    pair_count = min(target_pairs, max_pairs)
    if pair_count < 1:
        raise RuntimeError("No valid random pairs can be created from the sentence pool")

    rng = random.Random(seed)
    idx = list(range(len(lines)))
    rng.shuffle(idx)
    idx = idx[: pair_count * 2]

    hyp = [lines[i] for i in idx[:pair_count]]
    cand = [lines[i] for i in idx[pair_count : pair_count * 2]]
    return hyp, cand


def compute_baseline(
    hyp: List[str],
    cand: List[str],
    model_type: str,
    lang: str,
    device: str,
    batch_size: int,
    chunk_size: int,
) -> pd.DataFrame:
    print(
        f"[setup] computing baseline with model={model_type}, lang={lang}, "
        f"pairs={len(hyp)}, device={device}, chunk_size={chunk_size}, batch_size={batch_size}"
    )
    scorer = bert_score.BERTScorer(
        model_type=model_type,
        all_layers=True,
        lang=lang,
        device=device,
    )

    score_means = None
    count = 0

    with torch.no_grad():
        for start in tqdm(range(0, len(hyp), chunk_size), desc="baseline chunks"):
            batch_hyp = hyp[start : start + chunk_size]
            batch_cand = cand[start : start + chunk_size]
            scores = scorer.score(batch_hyp, batch_cand, batch_size=batch_size)
            scores = torch.stack(scores, dim=0)

            batch_n = len(batch_hyp)
            batch_mean = scores.mean(dim=-1)

            if score_means is None:
                score_means = batch_mean
            else:
                score_means = (
                    score_means * (count / (count + batch_n))
                    + batch_mean * (batch_n / (count + batch_n))
                )
            count += batch_n

    if score_means is None:
        raise RuntimeError("Failed to compute baseline scores")

    df = pd.DataFrame(score_means.cpu().numpy().transpose(), columns=["P", "R", "F"])
    df.index.name = "LAYER"
    return df


def main() -> None:
    args = parse_args()
    text_fields = [x.strip() for x in args.text_fields.split(",") if x.strip()]
    if not text_fields:
        raise ValueError("--text-fields produced an empty field list")

    device = resolve_device(args.device)

    baseline_path = Path(args.baseline_path) if args.baseline_path else default_baseline_path(args.lang, args.model_type)

    if baseline_path.exists() and not args.overwrite:
        print(f"[setup] baseline already exists: {baseline_path}")
        print("[setup] use --overwrite if you want to regenerate it")
        return

    if not args.skip_download:
        warm_model_cache(args.model_type)

    all_lines: List[str] = []
    for src in args.source_files:
        src_path = Path(src)
        if not src_path.exists():
            raise FileNotFoundError(f"Input source file not found: {src_path}")
        lines = read_sentences_from_file(src_path, text_fields)
        all_lines.extend(lines)
        print(f"[setup] extracted {len(lines)} raw lines from {src_path}")

    all_lines = [x for x in all_lines if keep_sentence(x, args.min_words, args.max_words)]
    all_lines = dedup_preserve_order(all_lines)

    if args.max_lines > 0 and len(all_lines) > args.max_lines:
        all_lines = all_lines[: args.max_lines]

    print(f"[setup] usable deduplicated lines: {len(all_lines)}")

    hyp, cand = make_random_pairs(all_lines, target_pairs=args.pairs, seed=args.seed)

    baseline_df = compute_baseline(
        hyp=hyp,
        cand=cand,
        model_type=args.model_type,
        lang=args.lang,
        device=device,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
    )

    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_df.to_csv(baseline_path)

    print(f"[setup] wrote baseline TSV to: {baseline_path}")
    print(f"[setup] rows={len(baseline_df)} columns={list(baseline_df.columns)}")
    print("[setup] done")


if __name__ == "__main__":
    main()
