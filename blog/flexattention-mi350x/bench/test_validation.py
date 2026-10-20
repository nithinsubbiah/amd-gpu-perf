"""CPU checks for isolation evidence, idle clocks, and complete campaigns."""

import copy
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from gpu_monitor import Monitor, parse_pid_gpus, validate_clocks
from landed_data import EXPECTED, load_data


def sample(phase, mhz=2200, busy=100, pid=10):
    return {"phase": phase, "sclk_mhz": mhz, "gpu_use_pct": busy,
            "pci_bus": "0000:e5:00.0", "pids": [pid], "time_ns": 1, "smi_device": 6}


def observations(pid=10):
    return [sample("before", 145, 0, pid), sample("timing", 2200, 100, pid),
            sample("timing", 2205, 100, pid), sample("after", 94, 0, pid)]


class ClockTests(unittest.TestCase):
    def test_normal_idle_clocks_at_boundaries_are_allowed(self):
        self.assertEqual(validate_clocks(observations(), 1000, 5)["samples"], 2)

    def test_cold_loaded_clocks_are_rejected(self):
        rows = observations()
        rows[1]["sclk_mhz"] = 145
        with self.assertRaises(RuntimeError):
            validate_clocks(rows, 1000, 5)

    def test_clock_spread_under_load_is_rejected(self):
        rows = observations()
        rows[1]["sclk_mhz"] = 1500
        with self.assertRaises(RuntimeError):
            validate_clocks(rows, 1000, 5)

    def test_missing_loaded_observations_are_rejected(self):
        with self.assertRaises(RuntimeError):
            validate_clocks(observations()[::3], 1000, 5)

    def test_dvfs_spread_is_retained_without_an_optional_limit(self):
        rows = observations()
        rows[1]["sclk_mhz"] = 1643
        rows[2]["sclk_mhz"] = 1836
        self.assertGreater(validate_clocks(rows, 1000, None)["spread_pct"], 11)

    def test_idle_checks_need_consecutive_zero_readings(self):
        monitor = object.__new__(Monitor)
        rows = [sample("idle_check", busy=n) for n in [0, 3, 0, 0, 0]]
        with patch.object(monitor, "snapshot", side_effect=rows), patch("gpu_monitor.time.sleep"):
            self.assertEqual(len(monitor.wait_idle()), 5)

    def test_pid_mapping_and_truncated_reports(self):
        self.assertEqual(parse_pid_gpus("PID 10 is using 2 DRM device(s):\n0 6 \n"), {10: [0, 6]})
        with self.assertRaises(RuntimeError):
            parse_pid_gpus("PID 10 is using 2 DRM device(s):\n6 \n")


class CampaignTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="flex-validation-test-")
        self.addCleanup(self.tmp.cleanup)
        self.runs = []
        for i in range(3):
            pid = 10 + i
            protocol = {"reps": 3, "reverse_order": i == 1, "idle_samples_required": 3,
                        "min_sclk_mhz": 1000, "max_clock_spread_pct": 5,
                        "clock_min_busy_pct": 90, "clock_min_samples": 2,
                        "max_trial_spread_pct": 5, "grad_mode": "enabled"}
            records = [{"type": "environment", "schema": 2, "run_id": str(i), "process_pid": pid,
                        "environment": {"pci_bus": "0000:e5:00.0"}, "protocol": protocol,
                        "script_sha256": {"benchmark": "fixture", "monitor": "fixture"}}]
            for dtype, n, dim, mask, stages, warps in sorted(EXPECTED):
                heads = 16 if n == 4096 else 32
                valid = n * (n + 1) // 2 if mask == "causal" else n * n
                samples = observations(pid)
                source = (f"@triton_heuristics.template(num_stages={stages},num_warps={warps})\n"
                          "@triton.jit\ndef fixture():\n    OUTPUT_LOGSUMEXP: tl.constexpr = True\n").encode()
                source_name = f"s{stages}-w{warps}.py.gz"
                Path(self.tmp.name, source_name).write_bytes(gzip.compress(source, mtime=0))
                records.append({"type": "measurement", "dtype": dtype, "batch": 1, "heads": heads,
                                "sequence_length": n, "head_dim": dim, "mask": mask,
                                "num_stages": stages, "num_warps": warps, "BLOCK_M": 128, "BLOCK_N": 64,
                                "trials_ms": [1.0] * 3, "median_ms": 1.0,
                                "tflops": 4 * heads * dim * valid * 1e-9, "correctness_passed": True,
                                "grad_enabled": True, "compiler_settings": {"output_logsumexp": True},
                                "pre_timing_idle_samples": [sample("idle_check", 145, 0, pid)] * 3,
                                "monitor_samples": samples,
                                "clock_validation": validate_clocks(samples, 1000, 5),
                                "generated_source": source_name,
                                "generated_source_sha256": hashlib.sha256(source).hexdigest()})
            records.append({"type": "complete", "measurements": len(EXPECTED)})
            self.runs.append(records)

    def load(self):
        for i, records in enumerate(self.runs):
            Path(self.tmp.name, f"run{i}.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in records))
        return load_data(str(Path(self.tmp.name, "*.jsonl")))

    def test_accepts_complete_compatible_runs(self):
        self.assertEqual(len(self.load()[0]), 3)

    def test_private_cache_paths_can_differ_but_compiler_overrides_cannot(self):
        for i, records in enumerate(self.runs):
            records[0]["environment"]["environment_overrides"] = {
                "TRITON_CACHE_DIR": f"/tmp/run{i}/triton",
                "TORCHINDUCTOR_CACHE_DIR": f"/tmp/run{i}/inductor",
                "TRITON_HIP_USE_BLOCK_PINGPONG": "0"}
        self.assertEqual(len(self.load()[0]), 3)
        self.runs[1][0]["environment"]["environment_overrides"]["TRITON_HIP_USE_BLOCK_PINGPONG"] = "1"
        with self.assertRaises(ValueError):
            self.load()

    def test_rejects_missing_or_contaminated_monitoring(self):
        good = copy.deepcopy(self.runs)
        for mutation in [
            lambda row: row.update(monitor_samples=[]),
            lambda row: row.update(pre_timing_idle_samples=[]),
            lambda row: row["monitor_samples"][1].update(pids=[999]),
            lambda row: row["pre_timing_idle_samples"][-1].update(gpu_use_pct=2),
            lambda row: row["monitor_samples"][1].update(pci_bus="other"),
            lambda row: row["clock_validation"].update(min_mhz=9999),
        ]:
            self.runs = copy.deepcopy(good)
            mutation(self.runs[0][1])
            with self.assertRaises(ValueError):
                self.load()

    def test_rejects_incomplete_or_inconsistent_campaigns(self):
        good = copy.deepcopy(self.runs)
        for mutation in [
            lambda r: r[0].pop(),
            lambda r: r[0].insert(2, copy.deepcopy(r[0][1])),
            lambda r: r[0][1].update(BLOCK_M=64),
            lambda r: r[0][1].update(correctness_passed=False),
            lambda r: r[0][1].update(grad_enabled=False),
            lambda r: r[0][1].update(compiler_settings={"output_logsumexp": False}),
            lambda r: r[0][1].update(trials_ms=[float("nan")] * 3),
            lambda r: r[0][1].update(trials_ms=[1.0, 1.0, 1.1]),
            lambda r: r[0][0].update(run_id="1"),
            lambda r: r[0][0].update(environment={"different": True, "pci_bus": "0000:e5:00.0"}),
        ]:
            self.runs = copy.deepcopy(good)
            mutation(self.runs)
            with self.assertRaises(ValueError):
                self.load()

    def test_rejects_modified_generated_code(self):
        self.load()
        Path(self.tmp.name, self.runs[0][1]["generated_source"]).write_bytes(gzip.compress(b"different source"))
        with self.assertRaises(ValueError):
            load_data(str(Path(self.tmp.name, "*.jsonl")))


if __name__ == "__main__":
    unittest.main()
