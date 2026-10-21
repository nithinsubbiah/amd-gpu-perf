"""Render the charts from the preserved summary CSV or a complete JSONL campaign.

Usage: plot_landed.py [input_csv_or_jsonl_glob] [output_directory]
Defaults to the official PyTorch 2.14.1 summary and the article's images directory.
No GPU is used. Custom inputs are not given the publication environment label.
"""

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from landed_data import SUMMARY, SHAPES, load_data, medians  # noqa: E402

plt.rcParams["svg.fonttype"] = "none"

BEFORE_COLOR = "#8a8f98"
AFTER_COLOR = "#ED1C24"  # AMD red


def chart(med, dtype, before, after, before_label, after_label, title, path,
          source_label=None):
    xs, bs, as_ = [], [], []
    for n, d, mask in SHAPES:
        b = med[(dtype, n, d, mask) + before]
        a = med[(dtype, n, d, mask) + after]
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
    ax.set_ylim(0, max(bs + as_) * 1.30)
    ax.legend(frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    if source_label:
        fig.subplots_adjust(bottom=0.27)
        fig.text(0.5, 0.025, source_label, ha="center", fontsize=8, color="#444")
    for ext in ("png", "svg"):
        target = Path(path).with_suffix("." + ext)
        fig.savefig(target, dpi=160,
                    metadata={"Date": None} if ext == "svg" else None)
        print("wrote", target)
    plt.close(fig)


if __name__ == "__main__":
    pattern = sys.argv[1] if len(sys.argv) > 1 else str(SUMMARY)
    outdir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parents[1] / "images"
    outdir.mkdir(parents=True, exist_ok=True)
    runs, description = load_data(pattern)
    print("Input:", description)
    med = medians(runs)
    source_label = (
        "PyTorch 2.14.1+rocm7.2 · triton-rocm 3.8.0 · ROCm 7.2.1 · B=1 · H=16 / 32 at S=4096 / 8192"
        if Path(pattern).resolve() == SUMMARY.resolve() else None
    )

    chart(med, "float16", (1, 8), (2, 4),
          "Previous defaults (num_stages=1, num_warps=8)",
          "Updated defaults (num_stages=2, num_warps=4)",
          "FlexAttention forward, fp16, AMD Instinct MI350X",
          f"{outdir}/01-fp16-speedup.png", source_label)

    chart(med, "bfloat16", (1, 4), (2, 4),
          "1 software-pipeline stage",
          "2 software-pipeline stages",
          "FlexAttention forward, bf16, effect of software pipelining on MI350X",
          f"{outdir}/02-bf16-pipelining.png", source_label)
