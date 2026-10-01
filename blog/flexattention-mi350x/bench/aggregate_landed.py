"""Aggregate repeated bench_landed.py runs into medians + observed spread.

Publication-grade reporting: a single run is not enough to distinguish a real
effect from run-to-run variation, so every figure quoted in the write-up comes
from the median of N independent runs, with the spread reported alongside.
"""

import glob
import re
import statistics
import sys
from collections import defaultdict

LINE = re.compile(
    r"LANDED\s+(?P<dtype>\w+)\s+N=(?P<n>\d+)\s+D=(?P<d>\d+)\s+(?P<mask>\w+)\s+"
    r"stages=(?P<stages>\d+)\s+warps=(?P<warps>\d+)\s+(?P<tflops>[\d.]+)\s+TFLOPS"
)

samples: dict[tuple, list[float]] = defaultdict(list)

logs = sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1 else "/tmp/bl_r*.log"))
for path in logs:
    with open(path) as fh:
        for line in fh:
            m = LINE.search(line)
            if m:
                key = (
                    m["dtype"],
                    int(m["n"]),
                    int(m["d"]),
                    m["mask"],
                    int(m["stages"]),
                    int(m["warps"]),
                )
                samples[key].append(float(m["tflops"]))

print(f"aggregated {len(logs)} runs: {', '.join(logs)}\n")

med = {k: statistics.median(v) for k, v in samples.items()}
spread = {k: (max(v) - min(v)) / statistics.median(v) * 100 for k, v in samples.items()}

print(f"{'config':<44} {'median':>8} {'min':>8} {'max':>8} {'spread%':>8} {'n':>3}")
for k in sorted(samples):
    v = samples[k]
    label = f"{k[0]:<9} N={k[1]:<5} D={k[2]:<4} {k[3]:<7} s={k[4]} w={k[5]}"
    print(
        f"{label:<44} {med[k]:>8.1f} {min(v):>8.1f} {max(v):>8.1f} "
        f"{spread[k]:>8.1f} {len(v):>3}"
    )

print(f"\nworst-case run-to-run spread: {max(spread.values()):.1f}%")
print(f"median run-to-run spread    : {statistics.median(spread.values()):.1f}%")

SHAPES = [(4096, 64, "nomask"), (4096, 64, "causal"), (4096, 128, "nomask"),
          (4096, 128, "causal"), (8192, 128, "nomask"), (8192, 128, "causal")]


def table(title, dtype, before, after):
    print(f"\n### {title}\n")
    print("| Sequence length | Head dim | Mask | before | after | Speedup |")
    print("|---|---|---|---|---|---|")
    ratios = []
    for n, d, mask in SHAPES:
        b = med.get((dtype, n, d, mask) + before)
        a = med.get((dtype, n, d, mask) + after)
        if b is None or a is None:
            continue
        ratios.append(a / b)
        label = mask.replace("nomask", "none")
        print(f"| {n} | {d} | {label} | {b:.1f} | {a:.1f} | {a / b:.2f}x |")
    if ratios:
        print(f"\nrange: {min(ratios):.2f}x - {max(ratios):.2f}x")


table("bf16 pipelining (stages 1 -> 2, warps=4)", "bfloat16", (1, 4), (2, 4))
table("fp16 warps (8 -> 4, stages=2)", "float16", (2, 8), (2, 4))
table("fp16 combined (pre: s1/w8 -> now: s2/w4)", "float16", (1, 8), (2, 4))
