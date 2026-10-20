"""Measure the forward configuration changes; retain complete JSONL runs.

Use an exclusively allocated gfx950 GPU with one HIP_VISIBLE_DEVICES entry.
Compilation, reference checks, clock ramps and monitoring run outside the timer.
See README.md for the historical campaign and the stricter reproduction protocol.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import gzip
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import statistics
import time
import uuid

from gpu_monitor import Monitor


SHAPES = [(1, 16, 4096, 64), (1, 16, 4096, 128), (1, 32, 8192, 128)]
LADDER = {"float16": [(1, 8), (2, 8), (2, 4), (1, 4)],
          "bfloat16": [(1, 4), (2, 4)]}
TOLERANCES = {"float16": {"atol": 0.003, "rtol": 0.01},
              "bfloat16": {"atol": 0.02, "rtol": 0.02}}


def flops(batch, heads, sequence, dim, causal):
    valid = sequence * (sequence + 1) // 2 if causal else sequence * sequence
    return 4 * batch * heads * dim * valid


def ramp_clocks(torch, seconds):
    a = torch.randn(8192, 8192, device="cuda", dtype=torch.float16)
    b = torch.randn_like(a)
    out = torch.empty_like(a)
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        for _ in range(20):
            torch.mm(a, b, out=out)
        torch.cuda.synchronize()


def warm_target(torch, fn, q, k, v, seconds):
    """Let DVFS settle on the measured kernel after the dense-matmul ramp."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        for _ in range(100):
            fn(q, k, v)
        torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New JSONL file; never overwritten")
    parser.add_argument("--device", type=int, help="Select a HIP GPU ordinal before importing PyTorch")
    parser.add_argument("--smoke", action="store_true", help="Validate only the first fp16 workload; cannot be aggregated")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--ramp-seconds", type=float, default=8.0)
    parser.add_argument("--target-warmup-seconds", type=float, default=3.0)
    parser.add_argument("--reverse-order", action="store_true")
    parser.add_argument("--grad-mode", choices=("enabled", "disabled"), default="enabled",
                        help="Match the original script's enabled grad context by default; inputs never require gradients")
    parser.add_argument("--min-sclk-mhz", type=int, default=1000)
    parser.add_argument("--max-clock-spread-pct", type=float, default=None,
                        help="Optional strict SCLK range limit; default records DVFS variation")
    parser.add_argument("--max-trial-spread-pct", type=float, default=5.0)
    args = parser.parse_args()
    if args.device is not None:
        if args.device < 0:
            parser.error("GPU ordinal must be nonnegative")
        os.environ["HIP_VISIBLE_DEVICES"] = str(args.device)
    if (args.reps < 3 or args.min_sclk_mhz <= 0
            or not math.isfinite(args.ramp_seconds) or args.ramp_seconds <= 0
            or not math.isfinite(args.target_warmup_seconds) or args.target_warmup_seconds <= 0
            or not math.isfinite(args.max_trial_spread_pct) or args.max_trial_spread_pct <= 0
            or (args.max_clock_spread_pct is not None and
                (not math.isfinite(args.max_clock_spread_pct) or args.max_clock_spread_pct <= 0))):
        parser.error("Use at least three repeats and positive clock/ramp limits")
    visible = os.environ.get("HIP_VISIBLE_DEVICES", "").split(",")
    if len(visible) != 1 or not visible[0].strip():
        parser.error("Set HIP_VISIBLE_DEVICES to exactly one exclusively allocated GPU")

    import torch
    import triton
    from torch._inductor import config
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    if not torch.version.hip or torch.cuda.device_count() != 1:
        raise RuntimeError("Expected exactly one visible ROCm GPU")
    prop = torch.cuda.get_device_properties(0)
    if prop.gcnArchName.split(":")[0] != "gfx950":
        raise RuntimeError("This measurement campaign targets gfx950")
    # Establish our KFD context so the monitor can verify its physical-device mapping.
    context = torch.ones(1, device="cuda")
    torch.cuda.synchronize()
    monitor = Monitor(args.min_sclk_mhz, args.max_clock_spread_pct)
    initial = monitor.wait_idle()
    patches = {"max_autotune": False, "max_autotune_gemm": False, "fx_graph_cache": False}
    if hasattr(config, "gluon_flex_attention"):
        patches["gluon_flex_attention"] = False
    torch_root = Path(torch.__file__).parent
    flex_root = torch_root / "_inductor/kernel/flex"
    source_files = [torch_root / path for path in ["_inductor/codegen/triton.py",
                    "_inductor/codegen/triton_utils.py", "_inductor/select_algorithm.py",
                    "_inductor/config.py", "_inductor/heuristics/template/triton.py",
                    "_inductor/runtime/triton_heuristics.py", "_inductor/utils.py",
                    "nn/attention/flex_attention.py"]]
    source_files += ([p for p in flex_root.rglob("*") if p.suffix in {".py", ".jinja"}]
                     if flex_root.is_dir() else [torch_root / "_inductor/kernel/flex_attention.py"])
    template_root = torch_root / "_inductor/kernel/templates"
    if template_root.is_dir():
        source_files += [p for p in template_root.rglob("*") if p.suffix in {".py", ".jinja"}]
    triton_distributions = {}
    for name in ("triton", "triton-rocm", "pytorch-triton-rocm", "pytorch-triton"):
        try:
            triton_distributions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    if not triton_distributions:
        raise RuntimeError("Cannot identify the installed Triton distribution")
    environment = {
        "torch": torch.__version__, "torch_git": torch.version.git_version,
        "hip": torch.version.hip, "triton": triton.__version__,
        "triton_distribution": ", ".join(f"{name}=={version}" for name, version in triton_distributions.items()),
        "triton_distributions": triton_distributions,
        "python": platform.python_version(), "platform": platform.platform(),
        "gpu": prop.name, "arch": prop.gcnArchName, "pci_bus": monitor.bus,
        "HIP_VISIBLE_DEVICES": os.environ["HIP_VISIBLE_DEVICES"],
        "inductor_overrides": patches,
        "environment_overrides": {k: v for k, v in os.environ.items()
                                  if k.startswith(("TORCHINDUCTOR_", "TRITON_"))},
        "torch_source_sha256": {
            str(path.relative_to(torch_root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(source_files)},
    }
    with args.output.open("x") as output:
        def emit(record):
            output.write(json.dumps(record, allow_nan=False) + "\n")
            output.flush()

        emit({"type": "environment", "schema": 2, "run_id": str(uuid.uuid4()),
              "process_pid": os.getpid(),
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "environment": environment, "initial_gpu_state": initial,
              "protocol": {"reps": args.reps, "ramp_seconds": args.ramp_seconds,
                           "target_warmup_seconds": args.target_warmup_seconds,
                           "warmup_ms": 200, "rep_ms": 500,
                           "reverse_order": args.reverse_order,
                           "grad_mode": args.grad_mode,
                           "min_sclk_mhz": args.min_sclk_mhz,
                           "max_clock_spread_pct": args.max_clock_spread_pct,
                           "max_trial_spread_pct": args.max_trial_spread_pct,
                           "clock_min_busy_pct": 90, "clock_min_samples": 2,
                           "idle_samples_required": 3,
                           "poll_interval_seconds": 0.5, "tolerances": TOLERANCES},
              "script_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                for name in ["bench_landed.py", "gpu_monitor.py"]}})
        count = 0
        attempt = {}
        try:
            with config.patch(patches), torch.set_grad_enabled(args.grad_mode == "enabled"):
                for dtype_name, configs in LADDER.items():
                    if args.smoke and dtype_name != "float16":
                        continue
                    dtype = getattr(torch, dtype_name)
                    ordered = list(reversed(configs)) if args.reverse_order else configs
                    for batch, heads, sequence, dim in SHAPES:
                        if args.smoke and (sequence, dim) != (4096, 64):
                            continue
                        for mask in ("nomask", "causal"):
                            if args.smoke and mask != "nomask":
                                continue
                            monitor.wait_idle()
                            torch.manual_seed(0)
                            q, k, v = (torch.randn(batch, heads, sequence, dim, device="cuda", dtype=dtype)
                                       for _ in range(3))
                            bm = (create_block_mask(lambda b, h, m, n: m >= n, batch, heads,
                                                    sequence, sequence, device="cuda")
                                  if mask == "causal" else None)
                            ref = flex_attention(q, k, v, block_mask=bm, scale=dim**-0.5)
                            if not torch.isfinite(ref).all().item():
                                raise RuntimeError("Reference contains nonfinite values")
                            for stages, warps in ordered:
                                monitor.snapshot()
                                opts = {"BLOCK_M": 128, "BLOCK_N": 64,
                                        "num_stages": stages, "num_warps": warps}

                                def fn(q, k, v, opts=opts):
                                    return flex_attention(q, k, v, block_mask=bm,
                                                          scale=dim**-0.5, kernel_options=opts)

                                torch._dynamo.reset()
                                compiled = torch.compile(fn, mode="default", fullgraph=True)
                                from torch._inductor.utils import run_and_get_code

                                out, generated = run_and_get_code(compiled, q, k, v)
                                source = "\n".join(generated)
                                headers = re.findall(r"@triton_heuristics\.template\((.*?)\)\s*@triton\.jit", source, re.S)
                                actual = {(int(re.search(r"num_stages\s*=\s*(\d+)", h)[1]),
                                           int(re.search(r"num_warps\s*=\s*(\d+)", h)[1])) for h in headers}
                                if actual != {(stages, warps)}:
                                    raise RuntimeError(f"Generated compiler settings {actual} do not match {(stages, warps)}")
                                lse_values = {value == "True" for value in re.findall(
                                    r"OUTPUT_LOGSUMEXP\s*:\s*tl.constexpr\s*=\s*(True|False)", source)}
                                if len(lse_values) != 1:
                                    raise RuntimeError("Cannot establish the generated OUTPUT_LOGSUMEXP setting")
                                output_logsumexp = lse_values.pop()
                                source_dir = args.output.with_suffix("") / "code"
                                source_dir.mkdir(parents=True, exist_ok=True)
                                source_path = source_dir / f"{dtype_name}-{sequence}-{dim}-{mask}-s{stages}-w{warps}.py.gz"
                                source_path.write_bytes(gzip.compress(source.encode(), mtime=0))
                                torch.cuda.synchronize()
                                if not torch.isfinite(out).all().item():
                                    raise RuntimeError("Compiled output contains nonfinite values")
                                torch.testing.assert_close(out, ref, **TOLERANCES[dtype_name])
                                diff = (out.float() - ref.float()).abs().max().item()
                                idle_samples = monitor.wait_idle()
                                ramp_clocks(torch, args.ramp_seconds)
                                warm_target(torch, compiled, q, k, v, args.target_warmup_seconds)
                                attempt = {"dtype": dtype_name, "sequence_length": sequence,
                                           "head_dim": dim, "mask": mask, **opts}
                                with monitor:
                                    trials = [float(triton.testing.do_bench(
                                        lambda: compiled(q, k, v), warmup=200, rep=500,
                                        return_mode="median")) for _ in range(args.reps)]
                                    attempt["trials_ms"] = trials
                                    torch.cuda.synchronize()
                                if not all(math.isfinite(t) and t > 0 for t in trials):
                                    raise RuntimeError("Timing contains invalid samples")
                                ms = statistics.median(trials)
                                spread = (max(trials) - min(trials)) / ms * 100
                                if spread > args.max_trial_spread_pct:
                                    raise RuntimeError(f"Timing spread {spread:.1f}% exceeds "
                                                       f"{args.max_trial_spread_pct}%")
                                tf = flops(batch, heads, sequence, dim, mask == "causal") / ms * 1e-9
                                emit({"type": "measurement", "dtype": dtype_name,
                                      "batch": batch, "heads": heads, "sequence_length": sequence,
                                      "head_dim": dim, "mask": mask, **opts,
                                      "trials_ms": trials, "median_ms": ms, "tflops": tf,
                                      "max_abs_diff": diff, "correctness_passed": True,
                                      "grad_enabled": torch.is_grad_enabled(),
                                      "pre_timing_idle_samples": idle_samples,
                                      "monitor_samples": monitor.samples,
                                      "clock_validation": monitor.clock_validation, "order": count,
                                      "generated_source": str(source_path.relative_to(args.output.parent)),
                                      "generated_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                                      "compiler_settings": {"num_stages": stages, "num_warps": warps,
                                                            "output_logsumexp": output_logsumexp}})
                                print(f"LANDED {dtype_name:8s} N={sequence} D={dim} {mask:7s} "
                                      f"stages={stages} warps={warps} {tf:6.1f} TFLOPS "
                                      f"maxd={diff:.1e} spread={spread:.1f}%", flush=True)
                                count += 1
            emit({"type": "smoke_complete" if args.smoke else "complete", "measurements": count})
        except Exception as error:
            emit({"type": "aborted", "error": f"{type(error).__name__}: {error}",
                  "measurements": count, "last_monitor_samples": monitor.samples,
                  "last_attempt": attempt})
            raise
    del context


if __name__ == "__main__":
    main()
