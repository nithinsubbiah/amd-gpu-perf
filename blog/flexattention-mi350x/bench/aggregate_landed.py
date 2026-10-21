"""Print article tables from the aggregate CSV or complete JSONL process runs.

Default: the official PyTorch 2.14.1 summary. For a new campaign, pass a quoted
JSONL glob. Ratio ranges describe the retained runs; they are not confidence intervals.
"""

import argparse
import statistics

from landed_data import SUMMARY, SHAPES, load_data, medians


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", default=str(SUMMARY))
    args = parser.parse_args()
    runs, description = load_data(args.input)
    med = medians(runs)
    print(f"Input: {description}\n")

    comparisons = [
        ("bf16 pipelining (1 -> 2 stages, 4 wavefronts)", "bfloat16", (1, 4), (2, 4)),
        ("fp16 wavefronts (8 -> 4, 2 stages)", "float16", (2, 8), (2, 4)),
        ("fp16 combined (1 stage / 8 waves -> 2 stages / 4 waves)", "float16", (1, 8), (2, 4)),
    ]
    for title, dtype, before, after in comparisons:
        print(f"### {title}\n")
        extra = " Median paired ratio [min, max] |" if len(runs) > 1 else ""
        print("| Sequence length | Head dim | Mask | Before | After | Speedup |" + extra)
        print("|---|---|---|---|---|---|" + ("---|" if len(runs) > 1 else ""))
        for n, d, mask in SHAPES:
            bkey = (dtype, n, d, mask) + before
            akey = (dtype, n, d, mask) + after
            before_tf, after_tf = med[bkey], med[akey]
            label = "none" if mask == "nomask" else mask
            row = (f"| {n} | {d} | {label} | {before_tf:.1f} | {after_tf:.1f} "
                   f"| {after_tf / before_tf:.2f}x |")
            if len(runs) > 1:
                paired = [run[akey] / run[bkey] for run in runs]
                row += f" {statistics.median(paired):.3f}x [{min(paired):.3f}, {max(paired):.3f}] |"
            print(row)
        print()


if __name__ == "__main__":
    main()
