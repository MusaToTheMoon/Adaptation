#!/usr/bin/env python3
import argparse
import re
import sys

import numpy as np


BASE_KEYS = {"en_base_prob", "ar_base_prob", "gt_letters"}


def layer_from_key(key):
    match = re.fullmatch(r"patch_L(\d+)", key)
    return int(match.group(1)) if match else None


def recovery(value, ar_mean, en_mean):
    gap = en_mean - ar_mean
    if gap <= 0:
        raise RuntimeError(f"Cannot derive L_patch: En-Ar gap is non-positive ({gap:.6g}).")
    return 100.0 * (value - ar_mean) / gap


def finite_row_subset(arrays):
    n = len(next(iter(arrays.values())))
    keep = np.ones(n, dtype=bool)
    bad_counts = {}
    for name, arr in arrays.items():
        arr = np.asarray(arr)
        if len(arr) != n:
            raise RuntimeError(f"{name} has length {len(arr)}, expected {n}.")
        finite = np.isfinite(arr)
        bad = int((~finite).sum())
        if bad:
            bad_counts[name] = bad
        keep &= finite
    return keep, bad_counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch_npz", required=True)
    parser.add_argument("--threshold", type=float, default=80.0)
    parser.add_argument(
        "--fallback",
        choices=["peak", "last", "error"],
        default="peak",
        help="What to do if no single-layer patch reaches threshold.",
    )
    args = parser.parse_args()

    data = np.load(args.patch_npz, allow_pickle=True)
    single_keys = []
    for key in data.files:
        if key in BASE_KEYS:
            continue
        layer = layer_from_key(key)
        if layer is not None:
            single_keys.append((layer, key))
    single_keys.sort()
    if not single_keys:
        raise RuntimeError(f"No single-layer patch keys found in {args.patch_npz}.")

    arrays = {
        "en_base_prob": data["en_base_prob"],
        "ar_base_prob": data["ar_base_prob"],
        **{key: data[key] for _layer, key in single_keys},
    }
    keep, bad_counts = finite_row_subset(arrays)
    if not keep.any():
        raise RuntimeError(f"No finite rows available in {args.patch_npz}.")

    if bad_counts:
        detail = ", ".join(f"{k}={v}" for k, v in sorted(bad_counts.items()))
        dropped = int((~keep).sum())
        print(
            f"[derive_l_patch] WARNING: dropped {dropped}/{len(keep)} rows with non-finite values: {detail}",
            file=sys.stderr,
            flush=True,
        )

    en_mean = float(np.asarray(data["en_base_prob"])[keep].mean())
    ar_mean = float(np.asarray(data["ar_base_prob"])[keep].mean())

    records = []
    for layer, key in single_keys:
        mean = float(np.asarray(data[key])[keep].mean())
        rec = recovery(mean, ar_mean, en_mean)
        records.append((layer, key, mean, rec))

    for layer, _key, _mean, rec in records:
        if rec >= args.threshold:
            print(layer)
            return

    if args.fallback == "error":
        best = max(records, key=lambda item: item[3])
        raise RuntimeError(
            f"No layer reached {args.threshold:.1f}% recovery; "
            f"best was L{best[0]} at {best[3]:.1f}%."
        )
    if args.fallback == "last":
        print(records[-1][0])
        return

    best = max(records, key=lambda item: item[3])
    print(
        f"[derive_l_patch] WARNING: no layer reached {args.threshold:.1f}% recovery; "
        f"using peak L{best[0]} at {best[3]:.1f}%.",
        file=sys.stderr,
        flush=True,
    )
    print(best[0])


if __name__ == "__main__":
    main()
