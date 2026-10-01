"""Quantify the landed upstream FlexAttention default-config work on MI350X.

Two merged PRs changed the ROCm default forward config on gfx950, and both are
measurable by pinning the old values through kernel_options and comparing against
today's default. This runs the *default* (non-autotuned) path, which is what the
PRs affect and what a user gets out of the box.

  pre-#176676   fp16 (128, 64, 1, 8)   bf16 (128, 64, 1, 4)
  +#176676      fp16 (128, 64, 2, 8)   bf16 (128, 64, 2, 4)   pipelining: stages 1->2
  +#180720      fp16 (128, 64, 2, 4)                          fp16 warps 8->4  [current]

(#181283 renamed the existing table to gfx950_* and added a separate gfx942 one, so
on MI350X its values are unchanged -- it is an MI300X improvement and is not
measurable here.)

These GPUs idle at ~145 MHz SCLK and this is a shared machine, so a short warmup
or a transiently busy device produces measurements that are low by anything from
10% to 5x. Both are guarded against here: clocks are ramped before any timing, each
point is measured several times and reduced by median, and the device is checked for
other users up front. Pin to a known-idle device with HIP_VISIBLE_DEVICES.

Usage: bench_landed.py [--reps N]
"""

import argparse
import statistics
import subprocess
import sys
import time

import torch
import triton
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

parser = argparse.ArgumentParser()
parser.add_argument("--reps", type=int, default=3,
                    help="timed repeats per point, reduced by median")
args = parser.parse_args()


def warn_if_device_busy():
    """This is a shared box; a co-tenant on our device invalidates the numbers."""
    try:
        out = subprocess.run(["rocm-smi", "--showpids"], capture_output=True,
                             text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return
    others = [ln for ln in out.splitlines()
              if ln.strip() and ln.split()[0].isdigit()]
    if others:
        print("WARNING: other processes hold GPUs; pin to an idle device via "
              "HIP_VISIBLE_DEVICES and re-check:", file=sys.stderr)
        for ln in others:
            print("  " + ln.strip(), file=sys.stderr)


def ramp_clocks(seconds=8.0):
    """Drag SCLK off its idle floor so the first point isn't measured cold."""
    a = torch.randn(8192, 8192, device="cuda", dtype=torch.float16)
    b = torch.randn(8192, 8192, device="cuda", dtype=torch.float16)
    deadline = time.time() + seconds
    while time.time() < deadline:
        for _ in range(20):
            torch.mm(a, b)
        torch.cuda.synchronize()
    del a, b
    torch.cuda.empty_cache()

SHAPES = [(1, 16, 4096, 64), (1, 16, 4096, 128), (1, 32, 8192, 128)]
MASKS = ["nomask", "causal"]

# dtype -> [(label, num_stages, num_warps), ...] in chronological order
LADDER = {
    torch.float16: [
        ("pre-#176676", 1, 8),
        ("+#176676 pipelining", 2, 8),
        ("+#180720 warps (current)", 2, 4),
        # Fourth cell of the (num_stages, num_warps) factorial. Not on the
        # chronological ladder, but needed to show the two knobs interact.
        ("stages=1 warps=4 (factorial cell)", 1, 4),
    ],
    torch.bfloat16: [
        ("pre-#176676", 1, 4),
        ("+#176676 pipelining (current)", 2, 4),
    ],
}


def flops(B, H, N, D, causal):
    valid = (N * N + N) / 2 if causal else N * N
    return 2 * (2.0 * B * H * valid * D)


warn_if_device_busy()
ramp_clocks()

for dtype in (torch.float16, torch.bfloat16):
    for (B, H, N, D) in SHAPES:
        for mask in MASKS:
            torch.manual_seed(0)
            q, k, v = (
                torch.randn(B, H, N, D, device="cuda", dtype=dtype) for _ in range(3)
            )
            bm = (
                create_block_mask(lambda b, h, m, n: m >= n, B, H, N, N, device="cuda")
                if mask == "causal"
                else None
            )
            ref = flex_attention(q, k, v, block_mask=bm, scale=D**-0.5)
            base = None
            for (label, stages, warps) in LADDER[dtype]:
                opts = {
                    "BLOCK_M": 128,
                    "BLOCK_N": 64,
                    "num_stages": stages,
                    "num_warps": warps,
                }

                def fn(q, k, v, opts=opts):
                    return flex_attention(
                        q, k, v, block_mask=bm, scale=D**-0.5, kernel_options=opts
                    )

                try:
                    torch._dynamo.reset()
                    compiled = torch.compile(fn, fullgraph=True)
                    out = compiled(q, k, v)
                    torch.cuda.synchronize()
                    trials = [
                        triton.testing.do_bench(
                            lambda: compiled(q, k, v), warmup=200, rep=500,
                            return_mode="median",
                        )
                        for _ in range(args.reps)
                    ]
                    ms = statistics.median(trials)
                    spread = (max(trials) - min(trials)) / ms * 100
                    tf = flops(B, H, N, D, mask == "causal") / ms * 1e-9
                    if base is None:
                        base = tf
                    diff = (out.float() - ref.float()).abs().max().item()
                    print(
                        f"LANDED {str(dtype).split('.')[-1]:8s} N={N} D={D} {mask:7s} "
                        f"stages={stages} warps={warps} {tf:6.1f} TFLOPS "
                        f"({tf / base:.3f}x vs pre) maxd={diff:.1e} "
                        f"spread={spread:.1f}%  {label}",
                        flush=True,
                    )
                except Exception as e:  # noqa: BLE001
                    print(f"LANDED {dtype} N={N} D={D} {mask} stages={stages} "
                          f"warps={warps}: ERR {type(e).__name__} {str(e)[:60]}",
                          flush=True)
