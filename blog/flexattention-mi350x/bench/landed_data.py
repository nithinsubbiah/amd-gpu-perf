"""Load the preserved aggregate CSV or complete, compatible JSONL campaigns."""

import csv
import glob
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

from gpu_monitor import validate_clocks

SUMMARY = (Path(__file__).resolve().parents[1]
           / "data/publication-2.14.1-2026-10-20/forward-summary.csv")
SHAPES = [(4096, 64, "nomask"), (4096, 64, "causal"),
          (4096, 128, "nomask"), (4096, 128, "causal"),
          (8192, 128, "nomask"), (8192, 128, "causal")]
CONFIGS = {"float16": [(1, 8), (2, 8), (2, 4), (1, 4)],
           "bfloat16": [(1, 4), (2, 4)]}
EXPECTED = {(dtype, n, d, mask, s, w) for dtype, configs in CONFIGS.items()
            for n, d, mask in SHAPES for s, w in configs}
SUMMARY_KEYS = {key for key in EXPECTED if not (key[0] == "float16" and key[-2:] == (1, 4))}


def key_of(row):
    return (row["dtype"], int(row["sequence_length"]), int(row["head_dim"]),
            row["mask"], int(row["num_stages"]), int(row["num_warps"]))


def add_row(run, row):
    key = key_of(row)
    if key not in EXPECTED or key in run:
        raise ValueError(f"Unexpected or duplicate configuration: {key}")
    if int(row["batch"]) != 1 or int(row["heads"]) != (16 if key[1] == 4096 else 32):
        raise ValueError(f"Unexpected batch or head count: {key}")
    value = float(row["tflops"])
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"Invalid throughput: {key}")
    run[key] = value


def validate_monitoring(row, meta, protocol):
    idle = row.get("pre_timing_idle_samples", [])
    samples = row.get("monitor_samples", [])
    required_idle = protocol["idle_samples_required"]
    if len(idle) < required_idle or not samples:
        raise ValueError("Missing GPU idleness or monitoring evidence")
    for sample in idle + samples:
        if (sample["pci_bus"] != meta["environment"]["pci_bus"]
                or sample["pids"] != [meta["process_pid"]]
                or not 0 <= sample["gpu_use_pct"] <= 100):
            raise ValueError("GPU identity, process isolation, or utilization check failed")
    if any(s["phase"] != "idle_check" for s in idle):
        raise ValueError("Invalid idle-check samples")
    if any(s["gpu_use_pct"] != 0 for s in idle[-required_idle:]):
        raise ValueError("GPU was not idle before timing")
    if (samples[0]["phase"] != "before" or samples[-1]["phase"] != "after"
            or any(s["phase"] != "timing" for s in samples[1:-1])):
        raise ValueError("Invalid monitor sample phases")
    try:
        expected = validate_clocks(samples, protocol["min_sclk_mhz"],
                                   protocol["max_clock_spread_pct"],
                                   protocol["clock_min_busy_pct"], protocol["clock_min_samples"])
    except RuntimeError as error:
        raise ValueError(str(error)) from error
    if row.get("clock_validation") != expected:
        raise ValueError("Recorded clock validation disagrees with observed samples")


def load_data(spec=SUMMARY):
    paths = sorted(Path(p) for p in glob.glob(str(spec)))
    if not paths:
        raise ValueError(f"No data matched {spec}")
    if len(paths) == 1 and paths[0].suffix == ".csv":
        run = {}
        with paths[0].open(newline="") as fh:
            for row in csv.DictReader(fh):
                add_row(run, row)
        if run.keys() != SUMMARY_KEYS:
            raise ValueError("Incomplete published summary CSV")
        return [run], "published aggregate summary"

    runs, signatures, ids = [], [], set()
    for path in paths:
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if (not records or records[0].get("type") != "environment"
                or records[0].get("schema") != 2 or records[-1].get("type") != "complete"):
            raise ValueError(f"Incomplete or unsupported run: {path}")
        meta = records[0]
        if meta["run_id"] in ids:
            raise ValueError("The same process run was included more than once")
        ids.add(meta["run_id"])
        protocol = {k: v for k, v in meta["protocol"].items() if k != "reverse_order"}
        if protocol["reps"] < 3 or protocol["idle_samples_required"] < 3:
            raise ValueError("Insufficient timing repeats or idle observations")
        environment = dict(meta["environment"])
        if "environment_overrides" in environment:
            # Fresh process-private cache directories are artifact locations,
            # while every compiler/backend override must still match exactly.
            environment["environment_overrides"] = {
                key: value for key, value in environment["environment_overrides"].items()
                if key not in {"TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR"}}
        signatures.append({"environment": environment, "protocol": protocol,
                           "script_sha256": meta["script_sha256"]})
        run = {}
        for row in records[1:-1]:
            if row.get("type") != "measurement" or row.get("correctness_passed") is not True:
                raise ValueError(f"Failed or unexpected record in {path}")
            if row.get("BLOCK_M") != 128 or row.get("BLOCK_N") != 64:
                raise ValueError(f"Unexpected tile size in {path}")
            validate_monitoring(row, meta, protocol)
            source_path = (path.parent / row["generated_source"]).resolve()
            if not source_path.is_relative_to(path.parent.resolve()):
                raise ValueError("Generated source path escapes the campaign directory")
            source = gzip.decompress(source_path.read_bytes())
            if hashlib.sha256(source).hexdigest() != row["generated_source_sha256"]:
                raise ValueError("Generated source checksum mismatch")
            headers = re.findall(r"@triton_heuristics\.template\((.*?)\)\s*@triton\.jit", source.decode(), re.S)
            settings = {(int(re.search(r"num_stages\s*=\s*(\d+)", h)[1]),
                         int(re.search(r"num_warps\s*=\s*(\d+)", h)[1])) for h in headers}
            if settings != {(row["num_stages"], row["num_warps"])}:
                raise ValueError("Generated code does not use the requested compiler settings")
            if "grad_mode" in protocol:
                if protocol["grad_mode"] not in {"enabled", "disabled"}:
                    raise ValueError("Invalid grad-mode protocol")
                if row.get("grad_enabled") is not (protocol["grad_mode"] == "enabled"):
                    raise ValueError("Recorded grad mode differs from the requested protocol")
                lse = {v == "True" for v in re.findall(
                    r"OUTPUT_LOGSUMEXP\s*:\s*tl.constexpr\s*=\s*(True|False)", source.decode())}
                if lse != {row.get("compiler_settings", {}).get("output_logsumexp")}:
                    raise ValueError("Generated OUTPUT_LOGSUMEXP setting disagrees with the record")
            trials = row["trials_ms"]
            if len(trials) != protocol["reps"] or not all(math.isfinite(t) and t > 0 for t in trials):
                raise ValueError(f"Invalid timing trials in {path}")
            median = statistics.median(trials)
            if (max(trials) - min(trials)) / median * 100 > protocol["max_trial_spread_pct"]:
                raise ValueError(f"Unstable timing trials in {path}")
            n, d, heads = row["sequence_length"], row["head_dim"], row["heads"]
            valid = n * (n + 1) // 2 if row["mask"] == "causal" else n * n
            tf = 4 * row["batch"] * heads * d * valid / median * 1e-9
            if not math.isclose(row["median_ms"], median) or not math.isclose(row["tflops"], tf):
                raise ValueError(f"Timing and throughput disagree in {path}")
            add_row(run, row)
        if run.keys() != EXPECTED or records[-1]["measurements"] != len(EXPECTED):
            raise ValueError(f"Missing configurations in {path}")
        runs.append(run)
    if any(s != signatures[0] for s in signatures[1:]):
        raise ValueError("Runs have different hardware, software, or measurement protocols")
    if len(runs) < 3:
        raise ValueError("Retain at least three complete process runs per campaign")
    grad_mode = signatures[0]["protocol"].get("grad_mode", "unrecorded")
    return runs, f"{len(runs)} complete process runs; grad mode: {grad_mode}"


def medians(runs):
    return {key: statistics.median(run[key] for run in runs) for key in runs[0]}
