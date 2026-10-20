"""Audit and summarize the matched four-configuration rechecks; no GPU work."""

import gzip
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[2] / "bench"))
from landed_data import validate_monitoring  # noqa: E402


def main():
    runs, compilation_keys = [], []
    for index in (1, 2, 3):
        label = f"current-matched-grad{index}"
        rows = [json.loads(line) for line in (ROOT / f"{label}.jsonl").read_text().splitlines()]
        assert rows[-1] == {"type": "smoke_complete", "measurements": 4}
        meta = rows[0]
        assert meta["protocol"]["grad_mode"] == "enabled"
        run = {}
        for row in rows[1:-1]:
            assert row["correctness_passed"] is True and row["grad_enabled"] is True
            assert row["compiler_settings"]["output_logsumexp"] is True
            validate_monitoring(row, meta, meta["protocol"])
            source = gzip.decompress((ROOT / row["generated_source"]).read_bytes())
            assert hashlib.sha256(source).hexdigest() == row["generated_source_sha256"]
            assert set(re.findall(rb"OUTPUT_LOGSUMEXP\s*:\s*tl.constexpr\s*=\s*(True|False)", source)) == {b"True"}
            key = (row["num_stages"], row["num_warps"])
            assert key not in run
            run[key] = row["tflops"]
        assert run.keys() == {(1, 8), (2, 8), (2, 4), (1, 4)}
        for name, expected in meta["script_sha256"].items():
            assert hashlib.sha256(gzip.decompress((ROOT / "source" / f"{name}.gz").read_bytes())).hexdigest() == expected
        assembly = json.loads((ROOT / label / "assembly.json").read_text())
        target = next(a for a in assembly if (a["metadata"]["num_stages"], a["metadata"]["num_warps"]) == (1, 8))
        text = gzip.decompress((ROOT / target["path"]).read_bytes())
        assert hashlib.sha256(text).hexdigest() == target["sha256"]
        assert re.search(rb"; NumVgprs: 128\b", text) and re.search(rb"; Occupancy: 4\b", text)
        compilation_keys.append(target["metadata"]["hash"])
        runs.append(run)
    original = json.loads((ROOT / "original-fresh-monitored.json").read_text())
    target = next(a for a in original["assembly"] if (a["metadata"]["num_stages"], a["metadata"]["num_warps"]) == (1, 8))
    assert set(compilation_keys) == {target["metadata"]["hash"]}
    summary = {f"s{s}-w{w}": {"process_tflops": [run[s, w] for run in runs],
                               "median_tflops": statistics.median(run[s, w] for run in runs)}
               for s, w in ((1, 8), (2, 8), (2, 4), (1, 4))}
    summary["combined_ratio_of_medians"] = summary["s2-w4"]["median_tflops"] / summary["s1-w8"]["median_tflops"]
    summary["baseline_compilation_key"] = compilation_keys[0]
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
