"""Independent backward validation. Artifacts stay outside the publication repo.

Complete backward is timed: delta preprocessing, every gradient kernel and any
reduction. Forward/compilation/reference calculations are outside the timer.
Numerical reference uses chunked FP64 PyTorch matmuls and explicit derivatives,
with a separate small CPU autograd check of that reference implementation.
"""

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "support"))
from gpu_monitor import Monitor

TOLERANCES = {
    "bfloat16": {"atol": 0.02, "rtol": 0.02, "relative_l2": 0.01},
    "float16": {"atol": 0.001, "rtol": 0.01, "relative_l2": 0.002},
    "float32": {"atol": 0.0001, "rtol": 0.001, "relative_l2": 0.0001},
}


def merged_cases():
    rows = []
    for d, claimed in [(64, [1.80, 2.09, 2.31]), (128, [1.18, 1.17, 1.14])]:
        rows += [("bfloat16", 1, 16, 16, n, d, "causal", ratio)
                 for n, ratio in zip([2048, 4096, 8192], claimed)]
    rows += [("bfloat16", 1, 16, 4, 4096, 64, "causal", 1.55),
             ("bfloat16", 1, 16, 4, 4096, 128, "causal", 1.41),
             ("bfloat16", 1, 16, 16, 4096, 128, "window1024", 1.36),
             ("float16", 1, 16, 16, 4096, 64, "causal", 2.09),
             ("float16", 1, 16, 16, 4096, 128, "causal", 1.16)]
    return rows


def open_cases():
    return [
        ("float16", 1, 8, 2, 4096, 64, "dense", 2.4062),
        ("bfloat16", 4, 64, 8, 4096, 64, "dense", 1.1823),
        ("bfloat16", 4, 21, 21, 2048, 96, "dense", 2.8014),
        ("bfloat16", 4, 21, 21, 4096, 96, "dense", 2.9070),
        ("bfloat16", 2, 8, 2, 2048, 96, "dense", 1.3046),
        ("bfloat16", 2, 8, 2, 4096, 96, "dense", 1.7766),
        ("bfloat16", 4, 64, 8, 4096, 128, "causal", 1.5739),
        ("bfloat16", 4, 10, 10, 2048, 192, "dense", 2.7682),
        ("bfloat16", 4, 10, 10, 4096, 192, "dense", 2.9178),
        ("bfloat16", 2, 8, 2, 2048, 192, "dense", 2.6170),
        ("bfloat16", 4, 32, 32, 4096, 256, "dense", 2.6562),
        ("bfloat16", 4, 32, 8, 4096, 256, "dense", 1.7618),
        ("float16", 4, 64, 8, 2048, 128, "causal", 1.5915),
        ("float16", 4, 64, 8, 4096, 128, "causal", 1.6374),
        ("float16", 2, 64, 8, 2048, 128, "causal", 1.8598),
        ("float16", 2, 64, 8, 4096, 128, "causal", 1.7983),
        ("float16", 1, 64, 8, 4096, 128, "causal", 1.6471),
        ("float32", 4, 64, 8, 2048, 128, "causal", 2.8652),
        ("float32", 4, 64, 8, 4096, 128, "causal", 2.6843),
        ("float32", 2, 64, 8, 2048, 128, "causal", 2.8568),
        ("float32", 2, 64, 8, 4096, 128, "causal", 2.7388),
        ("float32", 1, 64, 8, 2048, 128, "causal", 2.2070),
        ("float32", 1, 64, 8, 4096, 128, "causal", 2.9024),
    ]


def tile_options(values):
    return {"bwd_" + k: v for k, v in zip(
        ["BLOCK_M1", "BLOCK_N1", "BLOCK_M2", "BLOCK_N2", "num_stages", "num_warps"], values)}


def variants(campaign, row, ablations=False):
    dtype, b, hq, hkv, n, d, mask, claimed = row
    if campaign == "merged":
        old = (64, 64, 64, 64, 1, 4) if d == 64 else (64, 128, 128, 64, 1, 4)
        result = {"old": tile_options(old), "tuned": tile_options((32, 128, 128, 32, 2, 4))}
        if dtype == "bfloat16" and d == 128 and hkv == 16 and mask == "causal":
            result["tiles_only"] = tile_options((32, 128, 128, 32, 1, 4))
            result["stages_only"] = tile_options((64, 128, 128, 64, 2, 4))
        return result
    if campaign == "parent":
        return {"parent": {"BACKEND": "TRITON"}}
    result = {"stock": {"BACKEND": "TRITON", "bwd_ROCM_SEPARATE_DQ": False},
              "automatic": {"BACKEND": "TRITON"}}
    if (dtype, b, n, d) in [("float16", 2, 4096, 128), ("float32", 4, 2048, 128)]:
        opts = {"BACKEND": "TRITON", "bwd_ROCM_SEPARATE_DQ": True,
                "bwd_ROCM_EARLY_DP": True, "bwd_XCD_REMAP": True,
                "bwd_ROCM_D128_REVERSE_DQ_ROWS": True,
                "bwd_ROCM_D128_DV_VIEW_CARRY_SPLIT64": True,
                "bwd_BLOCKS_ARE_CONTIGUOUS": True, "bwd_USE_TMA": False,
                "bwd_PRESCALE_QK": False, "bwd_matrix_instr_nonkdim": 0,
                "bwd_waves_per_eu": 0, "bwd_kpack": 1}
        if dtype == "float16":
            opts.update(tile_options((32, 128, 128, 32, 2, 4)), bwd_ROCM_D128_FP16_COVERAGE=True)
        else:
            opts.update(tile_options((16, 64, 64, 16, 1, 4)), bwd_ROCM_D128_FP32_BALANCED64_COVERAGE=True)
        result["explicit_contiguous"] = opts
        result["explicit_indexed"] = dict(opts, bwd_BLOCKS_ARE_CONTIGUOUS=False)
        if ablations:
            result["no_reverse_or_views"] = dict(opts, bwd_ROCM_D128_REVERSE_DQ_ROWS=False,
                                                  bwd_ROCM_D128_DV_VIEW_CARRY_SPLIT64=False)
            result["no_views"] = dict(opts, bwd_ROCM_D128_DV_VIEW_CARRY_SPLIT64=False)
    return result


def reference(torch, q, k, v, grad, mask, qchunk=512, hchunk=8):
    """Explicit FP64 attention derivatives, with bounded score-matrix storage."""
    with torch.no_grad():
        q, k, v, grad = [x.detach().to(torch.float64) for x in (q, k, v, grad)]
        b, hq, n, d = q.shape
        hkv = k.shape[1]
        group = hq // hkv
        ke, ve = [x.repeat_interleave(group, dim=1) for x in (k, v)]
        out, dq = torch.empty_like(q), torch.empty_like(q)
        dk, dv = torch.zeros_like(ke), torch.zeros_like(ve)
        ni = torch.arange(n, device=q.device)[None, :]
        for bi in range(b):
            for hi in range(0, hq, hchunk):
                kk, vv = ke[bi, hi:hi+hchunk], ve[bi, hi:hi+hchunk]
                for qi in range(0, n, qchunk):
                    qq, gg = q[bi, hi:hi+hchunk, qi:qi+qchunk], grad[bi, hi:hi+hchunk, qi:qi+qchunk]
                    logits = (qq @ kk.transpose(-1, -2)) * d**-0.5
                    if mask != "dense":
                        mi = torch.arange(qi, min(qi+qchunk, n), device=q.device)[:, None]
                        allowed = mi >= ni
                        if mask == "window1024":
                            allowed = allowed & (ni > mi - 1024)
                        logits.masked_fill_(~allowed, -float("inf"))
                    p = logits.softmax(-1)
                    out[bi, hi:hi+hchunk, qi:qi+qchunk] = p @ vv
                    dp = gg @ vv.transpose(-1, -2)
                    ds = p * (dp - (p * dp).sum(dim=-1, keepdim=True))
                    dq[bi, hi:hi+hchunk, qi:qi+qchunk] = (ds @ kk) * d**-0.5
                    dk[bi, hi:hi+hchunk] += (ds.transpose(-1, -2) @ qq) * d**-0.5
                    dv[bi, hi:hi+hchunk] += p.transpose(-1, -2) @ gg
        return out, dq, dk.reshape(b, hkv, group, n, d).sum(2), dv.reshape(b, hkv, group, n, d).sum(2)


def check_reference(torch):
    torch.manual_seed(479)
    for mask in ["dense", "causal", "window1024"]:
        b, h, n, d = (1, 2, 1033, 4) if mask == "window1024" else (2, 4, 17, 8)
        q = torch.randn(b, h, n, d, dtype=torch.float64, requires_grad=True)
        k = torch.randn(b, h//2, n, d, dtype=torch.float64, requires_grad=True)
        v = torch.randn_like(k, requires_grad=True)
        g = torch.randn_like(q)
        ke, ve = [x.repeat_interleave(2, dim=1) for x in (k, v)]
        score = q @ ke.transpose(-1, -2) / d**0.5
        if mask != "dense":
            mi, ni = torch.arange(n)[:, None], torch.arange(n)[None, :]
            allowed = (mi >= ni) & ((ni > mi - 1024) if mask == "window1024" else True)
            score = score.masked_fill(~allowed, -float("inf"))
        out = score.softmax(-1) @ ve
        gradients = torch.autograd.grad(out, (q, k, v), g)
        actual = reference(torch, q, k, v, g, mask, qchunk=73 if n > 17 else 5, hchunk=3)
        for a, e in zip(actual, (out, *gradients)):
            torch.testing.assert_close(a, e, atol=1e-12, rtol=1e-12)


def compare(torch, actual, expected, dtype):
    limits = TOLERANCES[dtype]
    metrics = {}
    for name, a, e in zip(["out", "dq", "dk", "dv"], actual, expected):
        if not torch.isfinite(a).all().item():
            raise AssertionError(f"{name} contains nonfinite values")
        diff = a.detach().double() - e
        norm = float(e.norm())
        err = float(diff.norm()) / max(norm, 1e-30)
        ratio = diff.abs() / (limits["atol"] + limits["rtol"] * e.abs())
        metrics[name] = {"max_abs": float(diff.abs().max()), "relative_l2": err,
                         "max_tolerance_ratio": float(ratio.max()),
                         "reference_max_abs": float(e.abs().max())}
        if err > limits["relative_l2"] or metrics[name]["max_tolerance_ratio"] > 1.0:
            raise AssertionError(f"{name} fails independent reference: {metrics[name]}")
    return metrics


def source_audit(source):
    flags = {}
    for name, value in re.findall(r"^\s*([A-Za-z0-9_]+)\s*:\s*tl.constexpr\s*=\s*([^\n#]+)", source, re.M):
        if name.startswith(("ROCM_", "XCD_", "BLOCK_", "BLOCKS_", "GQA_")):
            flags.setdefault(name, []).append(value.strip())
    headers = re.findall(r"@triton_heuristics\.template\((.*?)\)\s*@triton\.jit", source, re.S)
    return {"flags": flags, "template_headers": headers,
            "kernel_definitions": re.findall(r"^def (triton_[^(]+)", source, re.M)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", choices=["merged", "open", "parent"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cases", default="all", help="Comma-separated zero-based row indices")
    parser.add_argument("--device", type=int, default=6)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--gradient", choices=["ones", "random"], default="random")
    parser.add_argument("--reverse-order", action="store_true")
    parser.add_argument("--ablations", action="store_true")
    parser.add_argument("--correctness-only", action="store_true")
    args = parser.parse_args()
    os.environ["HIP_VISIBLE_DEVICES"] = str(args.device)
    import torch
    import triton
    import triton.testing
    from torch._inductor import config
    from torch._inductor.utils import run_and_get_code
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    torch.set_num_threads(4)
    check_reference(torch)
    context = torch.ones(1, device="cuda")
    torch.cuda.synchronize()
    monitor = Monitor()
    initial = monitor.wait_idle()
    prop = torch.cuda.get_device_properties(0)
    if torch.cuda.device_count() != 1 or prop.gcnArchName.split(":")[0] != "gfx950":
        raise RuntimeError("Expected exactly one gfx950 GPU")
    patches = {"max_autotune": False, "max_autotune_gemm": False, "fx_graph_cache": False}
    if hasattr(config, "gluon_flex_attention"):
        patches["gluon_flex_attention"] = False
    rows = merged_cases() if args.campaign == "merged" else open_cases()
    chosen = list(range(len(rows))) if args.cases == "all" else [int(x) for x in args.cases.split(",")]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    codedir = args.output.with_suffix("") / "code"
    codedir.mkdir(parents=True, exist_ok=True)
    torchroot = Path(torch.__file__).parent
    sourcefiles = [torchroot / p for p in ["_inductor/kernel/flex/flex_attention.py",
                    "_inductor/kernel/flex/templates/flex_backwards.py.jinja", "_inductor/choices.py",
                    "_inductor/runtime/triton_heuristics.py", "nn/attention/flex_attention.py"]]
    failures = []
    with args.output.open("x") as file:
        def emit(rec):
            file.write(json.dumps(rec, allow_nan=False) + "\n")
            file.flush()
        emit({"type": "environment", "started": datetime.now(timezone.utc).isoformat(),
              "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "torch": torch.__version__, "torch_git": torch.version.git_version,
              "torch_root": str(torchroot), "hip": torch.version.hip,
              "source_checkout": subprocess.run(["git", "-C", str(torchroot), "rev-parse", "HEAD"],
                                                capture_output=True, text=True).stdout.strip(),
              "triton": triton.__version__, "triton_distribution": importlib.metadata.version("triton-rocm"),
              "python": platform.python_version(), "gpu": prop.name, "arch": prop.gcnArchName,
              "pci_bus": monitor.bus, "initial_idle": initial, "patches": patches,
              "source_sha256": {str(p.relative_to(torchroot)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sourcefiles},
              "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "tolerances": TOLERANCES, "reference_self_check": "FP64 CPU autograd passed",
              "timing": "triton.testing.do_bench; complete backward; warmup=200ms; rep=500ms; median; alternating variant order",
              "cache_env": {k: v for k, v in os.environ.items() if k.startswith(("TORCHINDUCTOR_", "TRITON_"))}})
        with config.patch(patches):
            for index in chosen:
                row = rows[index]
                dtype, b, hq, hkv, n, d, mask, claimed = row
                print(f"CASE {index}: {row}", flush=True)
                functions = {}
                try:
                    monitor.wait_idle()
                    torch.manual_seed(args.seed)
                    q = torch.randn(b, hq, n, d, device="cuda", dtype=getattr(torch, dtype), requires_grad=True)
                    k, v = [torch.randn(b, hkv, n, d, device="cuda", dtype=getattr(torch, dtype), requires_grad=True) for _ in range(2)]
                    g = torch.ones_like(q) if args.gradient == "ones" else torch.randn_like(q)
                    if mask == "dense":
                        bm = None
                    else:
                        def causal(b, h, m, n): return m >= n
                        def window(b, h, m, n): return (m >= n) & (n > m - 1024)
                        bm = create_block_mask(window if mask == "window1024" else causal,
                                               b if args.campaign == "merged" else None,
                                               None, n, n, device="cuda", BLOCK_SIZE=128)
                    refs = reference(torch, q, k, v, g, mask)
                    scale = d**-0.5 if args.campaign == "merged" else 1 / math.sqrt(d)
                    baseline_metrics = None
                    for name, opts in variants(args.campaign, row, args.ablations).items():
                        torch._dynamo.reset()
                        def fn(q, k, v, opts=opts):
                            return flex_attention(q, k, v, block_mask=bm, scale=scale,
                                                  enable_gqa=hq != hkv, kernel_options=opts)
                        compiled = torch.compile(fn, fullgraph=True)
                        out, fwd_code = run_and_get_code(compiled, q, k, v)
                        def backward(out=out):
                            q.grad = k.grad = v.grad = None
                            out.backward(g, retain_graph=True)
                        _, bwd_code = run_and_get_code(backward)
                        torch.cuda.synchronize()
                        code = "\n".join(bwd_code)
                        path = codedir / f"case{index}-{name}.py.gz"
                        path.write_bytes(gzip.compress(code.encode(), mtime=0))
                        metrics = compare(torch, [out, q.grad, k.grad, v.grad], refs, dtype)
                        if baseline_metrics is None:
                            baseline_metrics = metrics
                        else:
                            for field in ("dq", "dk", "dv"):
                                if metrics[field]["relative_l2"] > 2 * baseline_metrics[field]["relative_l2"] + 1e-6:
                                    raise AssertionError(f"{field} relative L2 error exceeds twice the stock error: {metrics}")
                        audit = source_audit(code)
                        if args.campaign == "open":
                            phases = audit["flags"].get("ROCM_BWD_PHASE", [])
                            if name == "stock" and any(p != "0" for p in phases):
                                raise AssertionError(f"Stock selected split phases: {phases}")
                            if name != "stock" and not {"1", "2"}.issubset(phases):
                                raise AssertionError(f"Requested optimized path did not activate: {phases}")
                        functions[name] = backward
                        emit({"type": "correctness", "case": index, "row": row, "variant": name,
                              "options": opts, "metrics": metrics, "audit": audit,
                              "generated_code": str(path.relative_to(args.output.parent)),
                              "code_sha256": hashlib.sha256(code.encode()).hexdigest()})
                        print(f"  {name}: independent gradients passed", flush=True)
                    del refs
                    if args.correctness_only:
                        continue
                    idle = monitor.wait_idle()
                    from bench_landed import ramp_clocks
                    ramp_clocks(torch, 8.0)
                    order = list(functions)
                    if args.reverse_order:
                        order.reverse()
                    times = {name: [] for name in functions}
                    with monitor:
                        deadline = time.monotonic() + 3.0
                        while time.monotonic() < deadline:
                            for fun in functions.values():
                                for _ in range(10): fun()
                            torch.cuda.synchronize()
                        for repeat in range(args.rounds):
                            for name in (order if repeat % 2 == 0 else order[::-1]):
                                ms = float(triton.testing.do_bench(functions[name], warmup=200, rep=500, return_mode="median"))
                                times[name].append(ms)
                    medians = {name: statistics.median(vals) for name, vals in times.items()}
                    base = "old" if args.campaign == "merged" else "stock" if args.campaign == "open" else "parent"
                    ratios = {name: statistics.median(a/b for a, b in zip(times[base], vals)) for name, vals in times.items()}
                    emit({"type": "timing", "case": index, "row": row, "ms": times,
                          "median_ms": medians, "paired_median_speedup": ratios,
                          "spread_pct": {name: (max(vals)-min(vals))/statistics.median(vals)*100 for name, vals in times.items()},
                          "idle": idle, "monitor": monitor.samples, "clocks": monitor.clock_validation})
                    print(f"  ms={medians}; speedup={ratios}", flush=True)
                except Exception as exc:
                    failures.append(index)
                    emit({"type": "failure", "case": index, "row": row, "error": repr(exc), "traceback": traceback.format_exc()})
                    traceback.print_exc()
                finally:
                    functions.clear()
                    torch.cuda.synchronize()
            emit({"type": "complete", "cases": chosen, "failures": failures})
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
