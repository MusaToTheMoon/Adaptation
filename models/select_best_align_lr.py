#!/usr/bin/env python3
"""Select the best completed LR trial from an alignment-search manifest."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _best_loss(output_dir: Path) -> Optional[float]:
    log_path = output_dir / "val_loss_log.json"
    if not log_path.exists():
        return None
    records = _load_json(log_path)
    losses = [
        record.get("eval_loss")
        for record in records
        if isinstance(record, dict) and record.get("eval_loss") is not None
    ]
    if not losses:
        return None
    return min(float(loss) for loss in losses)


def select_best_trial(manifest_path: Union[str, Path]) -> Dict[str, Any]:
    manifest_path = Path(manifest_path)
    manifest = _load_json(manifest_path)
    candidates: List[Dict[str, Any]] = []

    for trial in manifest.get("trials", []):
        if trial.get("status") != "done":
            continue
        output_dir = Path(trial["output_dir"])
        best_loss = _best_loss(output_dir)
        if best_loss is None:
            continue
        candidates.append(
            {
                "trial": trial["trial"],
                "lr": float(trial["lr"]),
                "best_loss": best_loss,
                "output_dir": str(output_dir),
            }
        )

    if not candidates:
        raise ValueError(f"No completed trials with validation losses found in {manifest_path}")

    return min(candidates, key=lambda item: (item["best_loss"], item["trial"]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", help="Path to search_manifest.json")
    parser.add_argument(
        "--field",
        choices=["lr", "trial", "best_loss", "output_dir", "json"],
        default="json",
        help="Field to print. Use json for the complete selected trial record.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        best = select_best_trial(args.manifest)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if args.field == "json":
        print(json.dumps(best, indent=2))
    elif args.field == "lr":
        print(f"{best['lr']:.6e}")
    else:
        print(best[args.field])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
