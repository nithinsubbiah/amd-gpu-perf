"""Audit and aggregate this retained six-process addressing experiment; no GPU."""

import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys

ROOT = Path(__file__).resolve().parent
BENCH = ROOT.parents[1] / "bench"
sys.path.insert(0, str(BENCH))
from landed_data import validate_monitoring  # noqa: E402


def main():
    runs = {mode: [] for mode in ("on", "off")}
    signatures, ids = [], set()
    for mode in runs:
        for index in range(1, 4):
            path = ROOT / f"address-{mode}{index}.jsonl"
            records = [json.loads(line) for line in path.read_text().splitlines()]
            assert records[-1] == {"type": "complete", "measurements": 2}
            assert len(records) == 4
            meta = records[0]
            assert meta["schema"] == 2 and meta["run_id"] not in ids
            ids.add(meta["run_id"])
            environment = dict(meta["environment"])
            overrides = dict(environment["environment_overrides"])
            assert overrides.pop("TORCHINDUCTOR_EMIT_POINTER_RANGE_32") == ("1" if mode == "on" else "0")
            for key in ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR"):
                overrides.pop(key)
            environment["environment_overrides"] = overrides
            signatures.append((environment, meta["protocol"], meta["script_sha256"]))
            for name, expected in meta["script_sha256"].items():
                archived = gzip.decompress((ROOT / "source" / f"{name}.gz").read_bytes())
                assert hashlib.sha256(archived).hexdigest() == expected
            run = {}
            for row in records[1:-1]:
                assert row["type"] == "measurement" and row["correctness_passed"] is True
                assert (row["dtype"], row["batch"], row["heads"], row["sequence_length"],
                        row["head_dim"], row["BLOCK_M"], row["BLOCK_N"], row["num_stages"],
                        row["num_warps"]) == ("float16", 1, 16, 4096, 128, 128, 64, 2, 4)
                validate_monitoring(row, meta, meta["protocol"])
                source = gzip.decompress((ROOT / row["generated_source"]).read_bytes())
                assert hashlib.sha256(source).hexdigest() == row["generated_source_sha256"]
                assert (b"tt.pointer_range" in source) == (mode == "on")
                trials = row["trials_ms"]
                assert len(trials) == meta["protocol"]["reps"]
                assert all(math.isfinite(t) and t > 0 for t in trials)
                ms = statistics.median(trials)
                assert (max(trials) - min(trials)) / ms * 100 <= meta["protocol"]["max_trial_spread_pct"]
                valid = 4096 * 4097 // 2 if row["mask"] == "causal" else 4096**2
                expected = 4 * 16 * 128 * valid / ms * 1e-9
                assert math.isclose(row["median_ms"], ms) and math.isclose(row["tflops"], expected)
                assert row["mask"] not in run
                run[row["mask"]] = row["tflops"]
            assert run.keys() == {"nomask", "causal"}
            manifest = json.loads(path.with_suffix("").joinpath("ablation.json").read_text())
            assert manifest["pointer_range"] == mode and manifest["memory_path_verified"] is True
            archived = gzip.decompress((ROOT / "source/bench_addressing.py.gz").read_bytes())
            assert manifest["driver_sha256"] == hashlib.sha256(archived).hexdigest()
            assert len(manifest["assembly"]) == 2
            for artifact in manifest["assembly"]:
                assembly = gzip.decompress((path.with_suffix("") / artifact["path"]).read_bytes())
                assert hashlib.sha256(assembly).hexdigest() == artifact["sha256"]
                for op in ("buffer_load", "buffer_store", "global_load", "global_store"):
                    assert len(re.findall(rf"^\s*{op}\w*\s", assembly.decode(), re.M)) == artifact[op]
                assert artifact["buffer_load"] > 0 if mode == "on" else artifact["buffer_load"] == 0
                if mode == "off":
                    assert artifact["global_load"] > 0 and artifact["buffer_store"] == 0
            runs[mode].append(run)
    assert all(signature == signatures[0] for signature in signatures)
    results = {}
    for mask in ("nomask", "causal"):
        on = statistics.median(run[mask] for run in runs["on"])
        off = statistics.median(run[mask] for run in runs["off"])
        paired = [a[mask] / b[mask] for a, b in zip(runs["on"], runs["off"])]
        results[mask] = {"buffer_on_tflops": on, "buffer_off_tflops": off,
                         "ratio_of_medians": on / off,
                         "paired_ratios": paired,
                         "paired_median": statistics.median(paired),
                         "paired_min": min(paired), "paired_max": max(paired)}
    (ROOT / "addressing-comparisons.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
