"""Render the blog figures from the aggregated bench_landed.py runs.

Reads the same logs aggregate_landed.py consumes so the figures and the tables
in the post cannot drift apart.

Usage: plot_landed.py '/tmp/hb_r*.log' <output_dir>
"""

import glob
import re
import statistics
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LINE = re.compile(
    r"LANDED\s+(?P<dtype>\w+)\s+N=(?P<n>\d+)\s+D=(?P<d>\d+)\s+(?P<mask>\w+)\s+"
    r"stages=(?P<stages>\d+)\s+warps=(?P<warps>\d+)\s+(?P<tflops>[\d.]+)\s+TFLOPS"
)

SHAPES = [(4096, 64, "nomask"), (4096, 64, "causal"), (4096, 128, "nomask"),
          (4096, 128, "causal"), (8192, 128, "nomask"), (8192, 128, "causal")]

BEFORE_COLOR = "#8a8f98"
AFTER_COLOR = "#ED1C24"  # AMD red


def load(pattern):
    samples = defaultdict(list)
    for path in sorted(glob.glob(pattern)):
        with open(path) as fh:
            for line in fh:
                m = LINE.search(line)
                if m:
                    key = (m["dtype"], int(m["n"]), int(m["d"]), m["mask"],
                           int(m["stages"]), int(m["warps"]))
                    samples[key].append(float(m["tflops"]))
    return {k: statistics.median(v) for k, v in samples.items()}


def chart(med, dtype, before, after, before_label, after_label, title, path):
    xs, bs, as_ = [], [], []
    for n, d, mask in SHAPES:
        b = med.get((dtype, n, d, mask) + before)
        a = med.get((dtype, n, d, mask) + after)
        if b is None or a is None:
            continue
        xs.append(f"{n}\nD={d}\n{'causal' if mask == 'causal' else 'no mask'}")
        bs.append(b)
        as_.append(a)

    idx = range(len(xs))
    width = 0.38
    fig, ax = plt.subplots(figsize=(10, 5.2))
    r1 = ax.bar([i - width / 2 for i in idx], bs, width,
                label=before_label, color=BEFORE_COLOR)
    r2 = ax.bar([i + width / 2 for i in idx], as_, width,
                label=after_label, color=AFTER_COLOR)

    ax.bar_label(r1, fmt="%.0f", fontsize=8, padding=2, color="#444")
    ax.bar_label(r2, fmt="%.0f", fontsize=8, padding=2, color="#444")

    for i, (b, a) in enumerate(zip(bs, as_)):
        ax.annotate(f"{a / b:.2f}x", xy=(i, max(b, a)), xytext=(0, 20),
                    textcoords="offset points", ha="center",
                    fontsize=10, fontweight="bold", color=AFTER_COLOR)

    ax.set_xticks(list(idx))
    ax.set_xticklabels(xs, fontsize=9)
    ax.set_ylabel("Throughput (TFLOP/s)")
    ax.set_xlabel("Sequence length / head dimension / masking")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylim(0, max(as_) * 1.22)
    ax.legend(frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    pattern = sys.argv[1] if len(sys.argv) > 1 else "/tmp/hb_r*.log"
    outdir = sys.argv[2].rstrip("/") if len(sys.argv) > 2 else "."
    med = load(pattern)

    chart(med, "float16", (1, 8), (2, 4),
          "Previous defaults (num_stages=1, num_warps=8)",
          "Current defaults (num_stages=2, num_warps=4)",
          "FlexAttention forward, fp16, AMD Instinct MI350X",
          f"{outdir}/01-fp16-speedup.png")

    chart(med, "bfloat16", (1, 4), (2, 4),
          "num_stages=1 (pipelining disabled)",
          "num_stages=2 (pipelining enabled)",
          "FlexAttention forward, bf16, effect of software pipelining on MI350X",
          f"{outdir}/02-bf16-pipelining.png")
