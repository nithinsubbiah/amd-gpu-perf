"""Audit the three publication processes and compare them with the draft data."""

import csv
import gzip
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1] / "bench"))
from landed_data import SUMMARY_KEYS, key_of, load_data, medians  # noqa: E402


def main():
    runs, description = load_data(str(ROOT / "run[123].jsonl"))
    med = medians(runs)
    historical_runs, _ = load_data(ROOT.parents[0] / "forward-summary.csv")
    historical = historical_runs[0]
    records = []
    for path in sorted(ROOT.glob("run[123].jsonl")):
        records.extend(json.loads(line) for line in path.read_text().splitlines())
    metadata = [row for row in records if row["type"] == "environment"]
    measurements = [row for row in records if row["type"] == "measurement"]
    if (len(metadata) != 3 or len(measurements) != 108
            or any(not row["environment"]["torch"].startswith("2.14.1+rocm7.2")
                   for row in metadata)
            or any(row["protocol"]["grad_mode"] != "enabled" for row in metadata)):
        raise RuntimeError("Unexpected publication campaign")
    hashes = json.loads((ROOT / "source/sha256.json").read_text())
    for row in metadata:
        if any(value != hashes[name] for name, value in row["script_sha256"].items()):
            raise RuntimeError("Measured harness differs from the frozen source")
    for name, digest in hashes.items():
        if hashlib.sha256(gzip.decompress((ROOT / f"source/{name}.gz").read_bytes())).hexdigest() != digest:
            raise RuntimeError(f"Archived source checksum mismatch: {name}")
    defaults = json.loads((ROOT / "defaults.json").read_text())
    if not defaults["complete"] or len(defaults["checks"]) != 12:
        raise RuntimeError("Incomplete unforced-default checks")
    before = {"combined": ("float16", (1, 8), (2, 4)),
              "wavefronts": ("float16", (2, 8), (2, 4)),
              "pipelining": ("bfloat16", (1, 4), (2, 4))}
    comparisons = {}
    for title, (dtype, old, new) in before.items():
        values = []
        for key in sorted(med):
            if key[0] != dtype or key[-2:] != old:
                continue
            updated = key[:-2] + new
            paired = [run[updated] / run[key] for run in runs]
            values.append({"sequence_length": key[1], "head_dim": key[2], "mask": key[3],
                           "before_tflops": med[key], "after_tflops": med[updated],
                           "ratio_of_medians": med[updated] / med[key],
                           "paired_ratios": paired})
        comparisons[title] = values
    historical_comparison = [
        {"key": list(key), "historical_tflops": historical[key], "publication_tflops": med[key],
         "change_pct": (med[key] / historical[key] - 1) * 100}
        for key in sorted(historical)]
    process_spreads = [(max(run[k] for run in runs) - min(run[k] for run in runs)) / med[k] * 100
                       for k in med]
    summary = {
        "description": description, "processes": 3, "measurements": len(measurements),
        "timing_trials": sum(len(row["trials_ms"]) for row in measurements),
        "environment": metadata[0]["environment"], "protocol": metadata[0]["protocol"],
        "output_logsumexp": sorted({row["compiler_settings"]["output_logsumexp"]
                                    for row in measurements}),
        "max_within_point_trial_spread_pct": max(
            (max(row["trials_ms"]) - min(row["trials_ms"])) / row["median_ms"] * 100
            for row in measurements),
        "median_across_process_spread_pct": statistics.median(process_spreads),
        "max_across_process_spread_pct": max(process_spreads),
        "max_abs_diff_by_dtype": {dtype: max(row["max_abs_diff"] for row in measurements
                                             if row["dtype"] == dtype)
                                  for dtype in ("float16", "bfloat16")},
        "loaded_sclk_min_mhz": min(row["clock_validation"]["min_mhz"] for row in measurements),
        "loaded_sclk_max_mhz": max(row["clock_validation"]["max_mhz"] for row in measurements),
        "comparisons": comparisons, "historical_comparison": historical_comparison,
        "unforced_default_settings": sorted({tuple(setting) for row in defaults["checks"]
                                              for setting in row["compiler_settings"]}),
    }
    (ROOT / "statistics.json").write_text(json.dumps(summary, indent=2) + "\n")
    fields = ["dtype", "batch", "heads", "sequence_length", "head_dim", "mask", "num_stages",
              "num_warps", "tflops"]
    first_run = [row for row in records if row.get("type") == "measurement"][:36]
    rows_by_key = {key_of(row): row for row in first_run}
    with (ROOT / "forward-summary.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for key in sorted(SUMMARY_KEYS):
            writer.writerow({**{name: rows_by_key[key][name] for name in fields}, "tflops": med[key]})
    print(json.dumps({k: summary[k] for k in ("description", "measurements", "timing_trials",
                     "max_within_point_trial_spread_pct", "max_across_process_spread_pct",
                     "comparisons", "unforced_default_settings")}, indent=2))


if __name__ == "__main__":
    main()
