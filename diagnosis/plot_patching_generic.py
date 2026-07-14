import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--npz", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--title", default="Activation patching")
parser.add_argument("--reverse", action="store_true")
args = parser.parse_args()

d = np.load(args.npz, allow_pickle=True)
en_base = d["en_base_prob"]
ar_base = d["ar_base_prob"]
patch_keys = [
    k for k in d.files
    if k not in {"en_base_prob", "ar_base_prob", "gt_letters"}
]
patch_keys = sorted(
    patch_keys,
    key=lambda k: [int(x[1:]) for x in k.replace("patch_", "").split("_") if x.startswith("L")]
)

if not patch_keys:
    raise RuntimeError(f"No patch keys found in {args.npz}")

N = len(en_base)
en_mean = float(en_base.mean())
ar_mean = float(ar_base.mean())
gap = en_mean - ar_mean

def recovery(val):
    return 100 * (val - ar_mean) / gap if gap > 0 else 0.0

def degradation(val):
    return 100 * (en_mean - val) / gap if gap > 0 else 0.0

means = [float(d[k].mean()) for k in patch_keys]
sems = [float(d[k].std() / np.sqrt(N)) for k in patch_keys]
metric = [degradation(m) if args.reverse else recovery(m) for m in means]

fig, ax1 = plt.subplots(figsize=(max(9, len(patch_keys) * 0.75), 4.8))
xs = np.arange(len(patch_keys))
ax1.bar(xs, means, yerr=sems, capsize=3, color="#C0392B", alpha=0.85)
ax1.axhline(ar_mean, color="#0D3349", linestyle="--", linewidth=1.4, label=f"Arabic baseline {ar_mean:.3f}")
ax1.axhline(en_mean, color="#2A9D8F", linestyle="--", linewidth=1.4, label=f"English baseline {en_mean:.3f}")
ax1.set_ylabel("Mean P(correct answer letter)")
ax1.set_xticks(xs)
ax1.set_xticklabels([k.replace("patch_", "") for k in patch_keys], rotation=45, ha="right")
ax1.set_title(args.title)
ax1.legend(frameon=False, fontsize=8)
ax1.grid(axis="y", alpha=0.2)

for x, m, pct in zip(xs, means, metric):
    label = f"{pct:.0f}%"
    ax1.text(x, m + max(0.005, max(means) * 0.025), label, ha="center", va="bottom", fontsize=8)

os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
plt.tight_layout()
plt.savefig(args.out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved -> {args.out}")
