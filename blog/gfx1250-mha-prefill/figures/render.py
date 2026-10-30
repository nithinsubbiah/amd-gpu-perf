"""Render the article's SVG figures; optionally save PNG review previews.

Requires matplotlib. All computation is on the CPU. Run from any directory:
    python path/to/figures/render.py --preview-dir previews
"""

import argparse
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parent
INK = "#182b42"
MUTED = "#526479"
BLUE = "#2369a8"
PALE = "#e7f0f8"
TEAL = "#087e79"
GREEN = "#d9efeb"
ORANGE = "#b34c17"
PEACH = "#fcebdd"
GRAY = "#eef1f4"
LINE = "#ced7df"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": INK,
        "svg.fonttype": "none",
        "svg.hashsalt": "gfx1250-prefill-article",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


def canvas(title, subtitle, height=58):
    fig, ax = plt.subplots(figsize=(14, height * 0.14))
    fig.subplots_adjust(left=0.015, right=0.985, top=0.99, bottom=0.01)
    ax.set(xlim=(0, 100), ylim=(0, height))
    ax.axis("off")
    label(ax, 3, height - 3, title, size=22, weight="bold")
    label(ax, 3, height - 7, subtitle, size=12, color=MUTED)
    return fig, ax


def label(ax, x, y, text, size=12, color=INK, weight="normal", ha="left", **kw):
    return ax.text(
        x,
        y,
        text,
        fontsize=size,
        color=color,
        weight=weight,
        ha=ha,
        va="center",
        linespacing=1.45,
        **kw,
    )


def box(ax, x, y, w, h, text="", fill=PALE, edge=LINE, size=12):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.2,rounding_size=0.6",
            facecolor=fill,
            edgecolor=edge,
            linewidth=1,
        )
    )
    if text:
        label(ax, x + w / 2, y + h / 2, text, size=size, ha="center")


def arrow(ax, x1, y1, x2, y2, color=MUTED):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=14,
            color=color,
            linewidth=1.5,
        )
    )


def save(fig, name, description, preview_dir):
    path = ROOT / f"{name}.svg"
    fig.savefig(path, format="svg", metadata={"Date": None})
    # Keep searchable SVG text and give the standalone asset an accessible name.
    ns = "http://www.w3.org/2000/svg"
    ET.register_namespace("", ns)
    ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
    tree = ET.parse(path)
    root = tree.getroot()
    root.set("role", "img")
    root.set("aria-labelledby", "figure-title figure-description")
    title = ET.Element(f"{{{ns}}}title", id="figure-title")
    title.text = name.replace("-", " ").capitalize()
    desc = ET.Element(f"{{{ns}}}desc", id="figure-description")
    desc.text = description
    root.insert(0, title)
    root.insert(1, desc)
    tree.write(path, encoding="utf-8", xml_declaration=True)
    if preview_dir:
        preview_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(preview_dir / f"{name}.png", dpi=120)
    plt.close(fig)


def attention_tile(preview_dir):
    fig, ax = canvas(
        "One workgroup, many key tiles",
        "Causal attention in the separate-head layout · schematic",
    )
    cell, x0, top = 2.2, 5, 44
    for q in range(16):
        for k in range(16):
            ax.add_patch(
                Rectangle(
                    (x0 + k * cell, top - (q + 1) * cell),
                    cell,
                    cell,
                    facecolor=PALE if k <= q else GRAY,
                    edgecolor="white",
                    linewidth=0.7,
                )
            )
    # A query tile visits successive key tiles; this highlighted block is full.
    y = top - 12 * cell
    ax.add_patch(
        Rectangle(
            (x0, y),
            16 * cell,
            4 * cell,
            facecolor="none",
            edgecolor=BLUE,
            linewidth=1.5,
            linestyle="--",
        )
    )
    ax.add_patch(
        Rectangle(
            (x0 + 4 * cell, y),
            4 * cell,
            4 * cell,
            facecolor=GREEN,
            edgecolor=TEAL,
            linewidth=2.5,
        )
    )
    label(ax, x0 + 6 * cell, y + 2 * cell, "BM\n× BN", ha="center", weight="bold")
    label(ax, 32, 38, "Masked\nfuture keys", size=12, ha="center", color=MUTED)
    label(ax, 11, 14, "Valid pairs", size=12, color=BLUE)
    arrow(ax, x0, 6, x0 + 16 * cell, 6)
    label(ax, 22, 3, "Key token position →", ha="center")
    arrow(ax, 2.5, top, 2.5, top - 16 * cell)
    label(ax, 0.8, 26, "Query token position →", ha="center", rotation=270)
    label(ax, 5, 47, "One sequence · one head", weight="bold")

    label(ax, 51, 47, "Inside one workgroup", weight="bold")
    box(ax, 51, 38, 18, 6, "Q tile\nBM × D", PALE)
    box(ax, 77, 38, 20, 6, "K / V tiles\nHBM → TDM → LDS", GREEN, size=11)
    arrow(ax, 60, 38, 60, 34)
    arrow(ax, 87, 38, 87, 34)
    box(ax, 51, 27, 46, 7, "QK → mask → softmax\nBM × BN scores / probabilities", PALE)
    arrow(ax, 74, 27, 74, 23)
    box(ax, 51, 16, 46, 7, "PV → accumulate partial output\nBM × D output state", GREEN)
    arrow(ax, 74, 16, 74, 12)
    label(ax, 74, 9, "Repeat for the next key tile", ha="center", weight="bold")
    label(ax, 74, 4, "Intermediate scores stay on chip", ha="center", color=MUTED)
    save(
        fig,
        "attention-tile",
        "A causal matrix highlights one BM by BN tile. A workgroup reuses its query rows while visiting key tiles; QK, softmax, and PV keep intermediate scores on chip.",
        preview_dir,
    )


def packing_and_compaction(preview_dir):
    fig, ax = canvas(
        "Two different ways to remove repeated work",
        "Head packing changes row ownership; ragged compaction changes the launch grid.",
        height=69,
    )
    label(ax, 3, 57, "A. Four query heads sharing one KV head", weight="bold", size=15)
    label(ax, 3, 53, "Separate workgroups", weight="bold")
    for i in range(4):
        y = 47 - 4 * i
        box(ax, 3, y, 12, 3, f"Q head {i}", PALE, size=11)
        arrow(ax, 16, y + 1.5, 19, y + 1.5)
        box(ax, 20, y, 15, 3, "Stage same K/V", GREEN, size=10)
    arrow(ax, 38, 42, 47, 42)
    label(ax, 52, 53, "One packed workgroup", weight="bold")
    # Eight rows = two positions times four heads. Colors encode query heads.
    colors = ["#bdd8ed", "#c8e6df", "#f5d9c6", "#d8d1ee"]
    for row in range(8):
        y = 48.5 - row * 1.9
        box(
            ax,
            52,
            y,
            24,
            1.65,
            f"position {row // 4} · head {row % 4}",
            colors[row % 4],
            size=9,
        )
    box(ax, 82, 39, 14, 7, "Stage K/V\nonce", GREEN)
    arrow(ax, 82, 42.5, 77, 42.5)
    label(
        ax,
        52,
        31.5,
        "Each row keeps its own normalization and output.",
        size=11,
        color=MUTED,
    )
    ax.plot([3, 97], [28, 28], color=LINE, linewidth=1)
    label(
        ax,
        3,
        24.5,
        "B. Uneven lengths [9, 4, 1, 0] · four query positions per tile",
        weight="bold",
        size=15,
    )
    label(ax, 3, 20.5, "Rectangular: 12 slots", weight="bold")
    counts = [3, 1, 1, 0]
    for b, count in enumerate(counts):
        y = 15.5 - b * 3.1
        label(ax, 3, y + 1.2, f"Request {b}", size=11)
        for tile in range(3):
            fill = colors[b] if tile < count else GRAY
            box(
                ax,
                14 + 5.3 * tile,
                y,
                4.7,
                2.4,
                str(tile) if tile < count else "—",
                fill,
                size=10,
            )
    arrow(ax, 34, 12.4, 45, 12.4)
    label(ax, 51, 20.5, "Compact: 8 slots", weight="bold")
    label(ax, 51, 17, "5 live tiles + 3 slots from grid rounding", size=11, color=MUTED)
    requests = [0, 0, 0, 1, 2, None, None, None]
    for i, request in enumerate(requests):
        box(
            ax,
            51 + i * 5.6,
            10.5,
            5,
            4,
            f"R{request}" if request is not None else "—",
            colors[request] if request is not None else GRAY,
            size=11,
        )
    label(ax, 51, 6.8, "Same live query tiles; fewer empty workgroups.", size=11)
    label(
        ax,
        3,
        2.5,
        "Toy examples. Head packing and token packing do not permit attention across sequence boundaries.",
        size=11,
        color=MUTED,
    )
    save(
        fig,
        "packing-and-compaction",
        "Top: four query heads reuse one staged KV tile in a packed workgroup. Bottom: lengths 9, 4, 1, 0 with query span four need five live tiles; compaction reduces twelve launch slots to eight including rounding.",
        preview_dir,
    )


def workgroup_order(preview_dir):
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.subplots_adjust(left=0.10, right=0.96, top=0.79, bottom=0.15, hspace=0.65)
    fig.text(
        0.045,
        0.945,
        "The same work, with a shorter completion tail",
        fontsize=22,
        weight="bold",
    )
    fig.text(
        0.045,
        0.895,
        "Eight toy workgroups · four abstract execution slots · equal total work in both schedules",
        fontsize=12,
        color=MUTED,
    )
    palette = plt.colormaps["Blues"]
    for ax, jobs, title in zip(
        axes,
        [range(1, 9), range(8, 0, -1)],
        ["Short-first order", "Longest-first order"],
    ):
        ends = [0] * 4
        for duration in jobs:
            slot = min(range(4), key=ends.__getitem__)
            start = ends[slot]
            ax.barh(
                slot,
                duration,
                left=start,
                height=0.7,
                color=palette(0.35 + duration * 0.055),
                edgecolor="white",
            )
            ax.text(
                start + duration / 2,
                slot,
                str(duration),
                va="center",
                ha="center",
                color="white" if duration >= 5 else INK,
                weight="bold",
            )
            ends[slot] += duration
        finish = max(ends)
        for slot, end in enumerate(ends):
            ax.barh(slot, finish - end, left=end, height=0.7, color=GRAY)
        ax.axvline(finish, color=TEAL, linestyle="--", linewidth=1.5)
        ax.text(
            finish + 0.1,
            1.5,
            f"Done\nat {finish}",
            color=TEAL,
            fontsize=12,
            va="center",
        )
        ax.set_title(title, loc="left", fontsize=15, weight="bold", pad=12)
        ax.set(
            yticks=range(4),
            yticklabels=[f"Slot {i + 1}" for i in range(4)],
            xlim=(0, 13),
            xticks=range(0, 13, 2),
        )
        ax.invert_yaxis()
        ax.tick_params(axis="both", length=0)
        ax.spines[["top", "left", "right"]].set_visible(False)
        ax.spines["bottom"].set_color(LINE)
    axes[-1].set_xlabel(
        "Arbitrary time units · numbers in blocks are workgroup durations", labelpad=10
    )
    fig.text(
        0.045,
        0.035,
        "Schematic, not a GPU trace. Gray marks idle slots before completion; these toy durations do not reproduce the measured 23% gain.",
        fontsize=11,
        color=MUTED,
    )
    save(
        fig,
        "workgroup-order",
        "Short-first scheduling finishes eight jobs at time twelve. Longest-first scheduling finishes the same jobs at time nine across four abstract execution slots. Durations are illustrative, not measured.",
        preview_dir,
    )


def kv_pipeline(preview_dir):
    fig, ax = canvas(
        "Look ahead in transport, stay narrow in compute",
        "Representative steady-state dependencies · exact instruction interleaving varies by route",
        height=62,
    )
    label(ax, 3, 49, "A. K leads by two tiles; V leads by one", weight="bold", size=15)
    starts = [22, 47, 72]
    for i, x in enumerate(starts):
        suffix = "t" if i == 0 else f"t+{i}"
        label(ax, x + 11.5, 44.5, f"Iteration {suffix}", ha="center", weight="bold")
        box(ax, x, 38.5, 23, 4, f"Transfer K[t+{i + 2}]", GREEN)
        box(ax, x, 33, 23, 4, f"Transfer V[t+{i + 1}]", PEACH)
        current = "t" if i == 0 else f"t+{i}"
        previous = "t−1" if i == 0 else ("t" if i == 1 else "t+1")
        box(
            ax,
            x,
            26,
            23,
            5,
            f"QK[{current}] (new scores)\nFinish softmax[{previous}] → PV[{previous}]",
            PALE,
            size=9.5,
        )
    label(ax, 3, 40.5, "K transport", color=TEAL)
    label(ax, 3, 35, "V transport", color=ORANGE)
    label(ax, 3, 28.5, "Compute")
    label(
        ax,
        22,
        22.5,
        "Two rotating K buffers + two rotating V buffers; reuse requires synchronization.",
        size=11,
        color=MUTED,
    )
    ax.plot([3, 97], [19.5, 19.5], color=LINE, linewidth=1)
    label(
        ax,
        3,
        16,
        "B. A wider transfer need not create a wider live score tile",
        weight="bold",
        size=15,
    )
    box(ax, 3, 5.5, 22, 6.5, "128-key transfer\ninto LDS", GREEN)
    arrow(ax, 26, 8.75, 33, 8.75)
    box(ax, 34, 5.5, 27, 6.5, "First 64 keys\nQK → softmax → PV", PALE)
    arrow(ax, 62, 8.75, 68, 8.75)
    box(ax, 69, 5.5, 27, 6.5, "Next 64 keys\nQK → softmax → PV", PALE)
    label(
        ax,
        3,
        2,
        "Blocks express logical dependencies, not simultaneous matrix execution within a wave.",
        size=11,
        color=MUTED,
    )
    save(
        fig,
        "kv-pipeline",
        "K and V transfers look ahead while earlier attention tiles compute. Two buffers per tensor rotate with synchronization. A 128-key transfer can feed two sequential 64-key compute steps.",
        preview_dir,
    )


def prefill_impact(preview_dir):
    data = json.loads((ROOT.parent / "data/main-comparison.json").read_text())
    rows = {r["dtype"]: r for r in data["rows"] if r["case"] == "mqa_8k"}
    fig, ax = plt.subplots(figsize=(14, 5.1))
    fig.subplots_adjust(left=0.10, right=0.78, top=0.73, bottom=0.28)
    fig.text(0.035, 0.925, "An 8K prompt: about 45% less kernel time", fontsize=23, weight="bold")
    fig.text(0.035, 0.86, "Causal MQA · one sequence · 8 query heads / 1 KV head · head dimension 128",
             fontsize=12, color=MUTED)
    for i, dtype in enumerate(("bfloat16", "float16")):
        row = rows[dtype]
        before, after = row["median_us"]["main"], row["median_us"]["pr"]
        assert math.isclose(row["speedup"], before / after, rel_tol=1e-12)
        for backend, offset, color, legend in (
            ("main", -0.18, "#c7d2df", "Upstream baseline"),
            ("pr", 0.18, TEAL, "Development candidate"),
        ):
            latency = row["median_us"][backend]
            ax.barh(i + offset, latency, height=0.29, color=color, label=legend if i == 0 else None)
            ax.text(latency + 2, i + offset, f"{latency:.1f} μs", va="center", fontsize=12, weight="bold")
            ax.text(3, i + offset, f"{row['tflops'][backend]:,.0f} TFLOP/s", va="center", fontsize=11,
                    color="white" if backend == "pr" else INK)
        ax.text(1.05, i - 0.08, f"{before / after:.2f}×", transform=ax.get_yaxis_transform(),
                fontsize=24, weight="bold", color=TEAL, va="center")
        ax.text(1.05, i + 0.23, f"{100 * (1 - after / before):.0f}% less time",
                transform=ax.get_yaxis_transform(), fontsize=12, color=MUTED, va="center")
    ax.set(yticks=[0, 1], yticklabels=["BF16", "FP16"], ylim=(1.55, -0.6),
           xlim=(0, 140), xticks=[0, 30, 60, 90, 120], xlabel="Kernel latency (μs) · lower is better")
    ax.tick_params(axis="y", length=0, labelsize=13, pad=12)
    ax.spines[["top", "left", "right"]].set_visible(False)
    ax.spines["bottom"].set_color(LINE)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=2, frameon=False, fontsize=11)
    fig.text(0.035, 0.13, "Baseline b5474d8 → candidate 34bf268, before final numerical fixes. "
             "Final merged source has not been retimed on a healthy host.", fontsize=10.5, color=MUTED)
    fig.text(0.035, 0.065, "Source: data/main-comparison.json · warmed graph timing, two runs · "
             "clocks not pinned · useful causal TFLOP/s · kernel results only", fontsize=10.5, color=MUTED)
    save(fig, "prefill-impact",
         "8K causal MQA, B1/Hq8/Hkv1/D128. BF16 kernel latency falls from 118.6 to 65.0 microseconds, "
         "and FP16 from 119.8 to 66.7 microseconds. Values come from the retained data. "
         "The development candidate predates final numerical fixes; this is not an end-to-end model benchmark.",
         preview_dir)


def prefill_speedups(preview_dir):
    data = json.loads((ROOT.parent / "data/main-comparison.json").read_text())
    order = [
        "short_gqa",
        "short_mqa",
        "short_mha",
        "short_partial_gqa2",
        "d64_512",
        "gqa_2k",
        "mqa_4k",
        "batched_4k",
        "mqa_8k",
        "mha_8k",
        "mqa_16k",
        "mqa_32k",
        "ragged_short",
        "ragged_4k",
        "ragged_16k",
        "ragged_skew",
        "wide_transport_control",
        "window_control",
    ]
    rows = {(r["case"], r["dtype"]): r for r in data["rows"]}
    assert len(rows) == 36 and {r["case"] for r in data["rows"]} == set(order)
    fig, ax = plt.subplots(figsize=(15, 10.5))
    fig.subplots_adjust(left=0.405, right=0.82, top=0.79, bottom=0.16)
    fig.text(
        0.035,
        0.955,
        "All 18 shapes, including the regressions",
        fontsize=23,
        weight="bold",
    )
    stats = data["aggregates"]
    fig.text(
        0.035,
        0.912,
        f"Geometric mean: BF16 {stats['bfloat16']['geomean_speedup']:.3f}× · FP16 {stats['float16']['geomean_speedup']:.3f}×"
        "     |     16 / 18 wins in each dtype",
        fontsize=14,
    )
    fig.text(
        0.035,
        0.87,
        "Later candidate vs upstream baseline already containing the earlier optimizations",
        fontsize=12,
        color=MUTED,
    )
    labels = []
    for i, case in enumerate(order):
        lengths, hq, hkv, d, window = rows[case, "bfloat16"]["shape"]
        seq = (
            f"{len(lengths)} × {lengths[0]:,}"
            if len(set(lengths)) == 1
            else "[" + ", ".join(map(str, lengths)) + "]"
        )
        labels.append(
            f"{seq}  ·  {hq}/{hkv}  ·  {d}" + (f"  ·  W{window}" if window >= 0 else "")
        )
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color="#f7f9fb", zorder=0)
    ax.axvspan(0, 1, color="#f8e8e4", alpha=0.55, zorder=1)
    ax.axvline(1, color=MUTED, linewidth=1.3, linestyle="--", zorder=2)
    for dtype, color, marker, offset, col in [
        ("bfloat16", BLUE, "o", -0.13, 1.09),
        ("float16", ORANGE, "D", 0.13, 1.29),
    ]:
        values = [rows[case, dtype]["speedup"] for case in order]
        # Verify that points are derived from latencies, not hand-entered claims.
        for case, value in zip(order, values):
            times = rows[case, dtype]["median_us"]
            assert math.isclose(value, times["main"] / times["pr"], rel_tol=1e-12)
        ax.scatter(
            values,
            [i + offset for i in range(len(order))],
            s=42,
            marker=marker,
            color=color,
            zorder=3,
            label="BF16" if dtype == "bfloat16" else "FP16",
        )
        for i, value in enumerate(values):
            ax.text(
                col,
                i,
                f"{value:.2f}×",
                transform=ax.get_yaxis_transform(),
                ha="center",
                va="center",
                color=color,
                fontsize=11,
            )
        ax.text(
            col,
            -1.2,
            "BF16" if dtype == "bfloat16" else "FP16",
            transform=ax.get_yaxis_transform(),
            ha="center",
            color=color,
            weight="bold",
        )
    ax.set(
        yticks=range(len(order)),
        yticklabels=labels,
        ylim=(17.6, -0.7),
        xlim=(0, 4.6),
        xticks=[0, 1, 2, 3, 4],
        xlabel="Speedup = baseline latency / candidate latency",
    )
    ax.text(
        0,
        -1.2,
        "Batch × length or [lengths] · Hq/Hkv · D",
        transform=ax.get_yaxis_transform(),
        ha="right",
        va="center",
        fontsize=11,
        weight="bold",
    )
    ax.tick_params(axis="y", length=0, labelsize=10.5, pad=10)
    ax.spines[["top", "left", "right"]].set_visible(False)
    ax.spines["bottom"].set_color(LINE)
    ax.legend(
        loc="lower right", bbox_to_anchor=(1, 1.04), ncol=2, fontsize=11, frameon=False
    )
    fig.text(
        0.035,
        0.095,
        "Baseline b5474d8 → candidate 34bf268, before final numerical fixes. Final merged source has not been retimed on a healthy host.",
        fontsize=11,
        color=MUTED,
    )
    fig.text(
        0.035,
        0.060,
        "Two runs × nine alternating timing rounds; each point uses aggregate latency, not a confidence interval. Clocks were not pinned.",
        fontsize=11,
        color=MUTED,
    )
    fig.text(
        0.035,
        0.025,
        "Source: data/main-comparison.json. Causal attention; W512 is the left-window control. Shapes receive equal weight in the geometric mean.",
        fontsize=10.5,
        color=MUTED,
    )
    save(
        fig,
        "prefill-speedups",
        "All eighteen configurations measured in BF16 and FP16, including two regressions per dtype. Geometric mean speedups are 1.468 and 1.472. The measured candidate predates final numerical fixes. All plotted values are read from the retained comparison data.",
        preview_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview-dir", type=Path)
    args = parser.parse_args()
    for render in (
        prefill_impact,
        attention_tile,
        packing_and_compaction,
        workgroup_order,
        kv_pipeline,
        prefill_speedups,
    ):
        render(args.preview_dir)
