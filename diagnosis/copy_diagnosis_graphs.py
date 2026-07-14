#!/usr/bin/env python3
import argparse
from pathlib import Path
import shutil


GRAPH_EXTS = {".png", ".pdf", ".svg"}
MODEL_ALIASES = [
    ("gemma3_27b_it", ("gemma3_27b_it", "gemma-3", "gemma3")),
    ("mistral_7b", ("mistral_7b", "mistral-7b")),
    ("llama31_8b_inst", ("llama31_8b_inst", "llama-3.1", "llama31")),
    ("medgemma", ("medgemma",)),
    ("falcon", ("falcon",)),
    ("fanar", ("fanar",)),
    ("allam", ("allam",)),
    ("llama", ("llama",)),
    ("mistral", ("mistral",)),
]


def model_ids():
    return [model for model, _aliases in MODEL_ALIASES] + ["_shared"]


def model_for_path(rel_path):
    text = str(rel_path).lower()
    if "panel" in text:
        return "_shared"
    for model, aliases in MODEL_ALIASES:
        if any(alias in text for alias in aliases):
            return model
    return "_shared"


def destination_for(src, rel_path, out_dir):
    model = model_for_path(rel_path)
    return out_dir / model / src.name


def graph_sources(diagnosis_dir, model_filter):
    for src in sorted(diagnosis_dir.rglob("*")):
        if not src.is_file() or src.suffix.lower() not in GRAPH_EXTS:
            continue
        rel = src.relative_to(diagnosis_dir)
        if rel.parts and rel.parts[0] == "graphs":
            continue
        model = model_for_path(rel)
        if model_filter and model != model_filter:
            continue
        yield src, rel, model


def clean_output(out_dir, model_filter):
    if model_filter:
        targets = [out_dir / model_filter]
    else:
        targets = [p for p in out_dir.iterdir() if p.is_dir()]

    for target in targets:
        if target.exists():
            shutil.rmtree(target)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=model_ids(),
        help="Copy only graphs for one model id, e.g. mistral_7b, falcon, or _shared.",
    )
    args = parser.parse_args()

    diagnosis_dir = Path(__file__).resolve().parent
    out_dir = diagnosis_dir / "graphs"
    out_dir.mkdir(parents=True, exist_ok=True)

    sources = list(graph_sources(diagnosis_dir, args.model))
    clean_output(out_dir, args.model)

    copied = 0
    for src, rel, _model in sources:
        dst = destination_for(src, rel, out_dir)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1

    scope = args.model if args.model else "all models"
    print(f"Copied {copied} graph files for {scope} under {out_dir}")


if __name__ == "__main__":
    main()
