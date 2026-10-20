"""Run release default checks, a smoke check, and three audited benchmark processes."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=6)
    args = parser.parse_args()
    if args.device < 0:
        parser.error("GPU ordinal must be nonnegative")
    os.environ["HIP_VISIBLE_DEVICES"] = str(args.device)
    root = Path(__file__).resolve().parent
    # Keep GPU runtime initialization in a short-lived process so it cannot
    # appear as a competing KFD PID while the benchmark children are running.
    probe = subprocess.check_output([
        sys.executable, "-c",
        "import json,torch; print(json.dumps({'torch':torch.__version__,"
        "'torch_git':torch.version.git_version,'gpu_available':torch.cuda.is_available(),"
        "'device_count':torch.cuda.device_count()}))"], text=True)
    device_state = json.loads(probe)
    preflight = {"recorded_utc": datetime.now(timezone.utc).isoformat(),
                 **device_state,
                 "kfd_exposed": Path("/dev/kfd").exists(),
                 "selected_ordinal": args.device,
                 "rocm_smi": subprocess.check_output(
                     ["rocm-smi", "--showuse", "--showpidgpus"], text=True)}
    (root / "environment/preflight.json").write_text(json.dumps(preflight, indent=2) + "\n")
    if not preflight["gpu_available"] or preflight["device_count"] != 1:
        raise RuntimeError("GPU preflight failed: run in an execution context exposing /dev/kfd "
                           "and /dev/dri. No benchmark was started.")
    if preflight["torch"] != "2.14.1+rocm7.2":
        raise RuntimeError("Run with the isolated PyTorch 2.14.1 ROCm 7.2 interpreter")
    stages = [("snapshot_environment.py", []), ("audit_release.py", []),
              ("check_defaults.py", ["--device", str(args.device)]),
              ("run_campaign.py", ["--device", str(args.device), "--label", "smoke", "--smoke"])]
    stages += [("run_campaign.py", ["--device", str(args.device), "--label", f"run{i}"]
                + (["--reverse-order"] if i == 2 else [])) for i in (1, 2, 3)]
    for script, options in stages:
        print(f"Running {script} {' '.join(options)}", flush=True)
        subprocess.run([sys.executable, str(root / script), *options], check=True)
    subprocess.run([sys.executable, str(root / "analyze.py")], check=True)
    bench = root.parents[1] / "bench"
    with (root / "comparisons.md").open("w") as output:
        subprocess.run([sys.executable, str(bench / "aggregate_landed.py"),
                        str(root / "run[123].jsonl")], stdout=output, check=True)
    subprocess.run([sys.executable, str(bench / "plot_landed.py"),
                    str(root / "run[123].jsonl"), str(root / "figures")], check=True)
    (root / "complete.json").write_text(json.dumps({
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "processes": 3, "measurements": 108, "correctness_and_monitoring_passed": True,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
