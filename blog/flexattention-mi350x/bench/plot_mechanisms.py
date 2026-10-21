"""Draw the article's mechanism diagrams as editable SVG and publication PNG.

Run with a Python environment containing matplotlib. No GPU is used.
"""

from pathlib import Path
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})

INK = "#172333"
MUTED = "#526173"
RED = "#b82029"
BLUE = "#17608a"
GRAY = "#f0f2f5"
LIGHT_BLUE = "#eaf3f8"
LINE = "#cdd5dd"
OUT = Path(__file__).resolve().parents[1] / "images"


def canvas(width, height):
    fig, ax = plt.subplots(figsize=(width / 100, height / 100))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set(xlim=(0, width), ylim=(height, 0))
    ax.axis("off")
    return fig, ax


def text(ax, x, y, label, size=16, color=INK, weight="normal", ha="left"):
    return ax.text(x, y, label, fontsize=size, color=color, weight=weight,
                   va="center", ha=ha, linespacing=1.4)


def box(ax, x, y, w, h, fill="white", edge=LINE, radius=12, **kwargs):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                              boxstyle=f"round,pad=0,rounding_size={radius}",
                              facecolor=fill, edgecolor=edge, linewidth=1.2, **kwargs))


def arrow(ax, x1, y1, x2, y2, color=MUTED):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                mutation_scale=17, linewidth=1.6, color=color))


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    for ext in ("svg", "png"):
        path = OUT / f"{name}.{ext}"
        fig.savefig(path, dpi=160, facecolor="white",
                    metadata={"Date": None} if ext == "svg" else None)
        print(path)
    plt.close(fig)


def wavefronts():
    audit = OUT.parent / "data/publication-2.14.1-2026-10-20/compiler-audit.json"
    counts = {row["num_warps"]: row["publication"]
              for row in json.loads(audit.read_text())["historical_comparison"]}
    fig, ax = canvas(1600, 1040)
    text(ax, 40, 42, "Making every wavefront count", 27, weight="bold")
    text(ax, 40, 82,
         "Inspected case: gfx950 · fp16 · BLOCK_M=128 · BLOCK_N=64 · head dim=128 · 32×32 MFMA", 16, MUTED)

    for left, waves, color, title in (
        (40, 8, RED, "8 wavefronts: replicated QK work"),
        (820, 4, BLUE, "4 wavefronts: matching layouts"),
    ):
        box(ax, left, 120, 740, 820)
        text(ax, left + 25, 157, title, 19, color, "bold")
        text(ax, left + 25, 193, f"num_warps={waves}   |   num_stages=2", 15, MUTED)
        x, w = left + 125, 540
        text(ax, x, 236, "QKᵀ work per 32-row band", 17, weight="bold")
        text(ax, x, 266, f"128 rows × 64 keys · wave layout [{waves}, 1]", 15, MUTED)
        for row in range(4):
            y = 292 + row * 42
            ax.add_patch(Rectangle((x, y), w, 42, facecolor=GRAY if waves == 8 else LIGHT_BLUE,
                                   edgecolor="white", linewidth=2))
            text(ax, x - 14, y + 21, f"{row * 32}–{row * 32 + 31}", 12, MUTED, ha="right")
            if waves == 8:
                ax.add_patch(Rectangle((x + 274, y + 3), 262, 36, facecolor="none",
                                       edgecolor="#bd6b70", hatch="///", linewidth=0.7))
                text(ax, x + 135, y + 21, f"W{row}: all 64 keys", 14, ha="center")
                text(ax, x + 405, y + 21, f"W{row + 4}: same 64 keys", 14, RED, ha="center")
            else:
                text(ax, x + w / 2, y + 21, f"W{row}: all 64 keys", 15, BLUE, "bold", "center")

        cx = x + w / 2
        arrow(ax, cx, 464, cx, 496, color)
        text(ax, cx, 513, "softmax", 16, ha="center")
        arrow(ax, cx, 531, cx, 554, color)
        box(ax, x, 560, w, 45, fill="#fff0f0" if waves == 8 else LIGHT_BLUE, edge=color)
        text(ax, cx, 582,
             "Layout conversion through LDS" if waves == 8 else "Matching wave layouts",
             16, color, "bold", "center")
        arrow(ax, cx, 610, cx, 642, color)
        text(ax, x, 666, "PV: 128 query rows × 128 output columns", 17, weight="bold")
        text(ax, x, 696, f"32-column tiles · wave layout [4, {2 if waves == 8 else 1}]", 15, MUTED)
        for row in range(4):
            y = 720 + row * 32
            ax.add_patch(Rectangle((x, y), w, 32, facecolor=GRAY if waves == 8 else LIGHT_BLUE,
                                   edgecolor="white", linewidth=2))
            if waves == 8:
                for col in range(4):
                    if col:
                        ax.plot([x + w * col / 4] * 2, [y, y + 32], color="white", linewidth=2)
                    text(ax, x + w * (col + 0.5) / 4, y + 16,
                         f"W{2 * row + col % 2}", 13, color, ha="center")
            else:
                text(ax, cx, y + 16, f"W{row}: all 128 columns", 14, color, ha="center")

        text(ax, left + 25, 877, "Static evidence · PyTorch 2.14.1 · triton-rocm 3.8.0", 14, MUTED, "bold")
        text(ax, left + 25, 910,
             f"MFMA sites × waves: {counts[waves]['wave_weighted_mfma_sites']}"
             f"     ·     ds_write sites: {counts[waves]['instructions']['ds_write']}",
             16, color, "bold")

    text(ax, 40, 977, "Each horizontal band represents 32 query rows. Hatching marks replicated computation.", 15, MUTED)
    text(ax, 40, 1008, "Static counts describe the compiled kernels; runtime speedups are measured separately.", 15, MUTED)
    save(fig, "03-wavefront-distribution")


def addressing():
    fig, ax = canvas(1600, 820)
    text(ax, 40, 42, "Streamlining memory addressing", 27, weight="bold")
    text(ax, 40, 82, "Known bounds let the compiler retain one base address and use smaller offsets.", 17, MUTED)

    steps = [(40, "Storage provably < 2 GiB"),
             (570, "tt.pointer_range = 32"),
             (1100, "Separate base and offsets")]
    for x, label in steps:
        box(ax, x, 120, 460, 70, LIGHT_BLUE)
        text(ax, x + 230, 155, label, 17, BLUE, "bold", "center")
    arrow(ax, 506, 155, 561, 155)
    arrow(ax, 1036, 155, 1091, 155)

    box(ax, 40, 230, 740, 490)
    box(ax, 820, 230, 740, 490)
    text(ax, 70, 268, "General address representation", 21, weight="bold")
    text(ax, 850, 268, "Base plus bounded offsets", 21, BLUE, "bold")

    for lane in range(4):
        y = 332 + lane * 65
        text(ax, 72, y + 23, f"lane {lane}", 15, MUTED)
        box(ax, 160, y, 550, 46, GRAY)
        text(ax, 435, y + 23, f"64-bit address = base + offset{lane}", 17, ha="center")

    box(ax, 930, 310, 560, 52, LIGHT_BLUE, BLUE)
    text(ax, 1210, 336, "Shared 64-bit base", 18, BLUE, "bold", "center")
    ax.plot([920, 900, 900, 900], [336, 336, 603, 603], color=BLUE, linewidth=1.6)
    ax.plot([930, 920], [336, 336], color=BLUE, linewidth=1.6)
    for lane in range(4):
        y = 393 + lane * 55
        arrow(ax, 900, y + 20, 1015, y + 20, BLUE)
        box(ax, 1020, y, 470, 40, LIGHT_BLUE)
        text(ax, 1255, y + 20, f"lane {lane}: bounded 32-bit offset{lane}", 16, ha="center")

    text(ax, 410, 670, "General global loads and stores", 18, MUTED, "bold", "center")
    text(ax, 1190, 670, "Eligible buffer loads and stores", 18, BLUE, "bold", "center")
    text(ax, 40, 757, "Addresses remain 64-bit. The compiler narrows offsets only when it can prove the transformation safe.", 16, MUTED)
    text(ax, 40, 790, "This diagram covers loads and stores; atomic-containing kernels need the separate annotation guard.", 15, MUTED)
    save(fig, "04-memory-addressing")


if __name__ == "__main__":
    wavefronts()
    addressing()
