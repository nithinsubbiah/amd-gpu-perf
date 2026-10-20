"""Check unforced release defaults, generated flags, and reference correctness."""

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=6)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    artifacts = root / "defaults"
    artifacts.mkdir()
    (root / "environment/rocm-smi-hardware.json").write_text(subprocess.check_output(
        ["rocm-smi", "--showdriverversion", "--showproductname", "--showmeminfo", "vram",
         "--showbus", "--json"], text=True))
    os.environ["HIP_VISIBLE_DEVICES"] = str(args.device)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(artifacts / "inductor-cache")
    os.environ["TRITON_CACHE_DIR"] = str(artifacts / "triton-cache")
    sys.path.insert(0, str(root.parents[1] / "bench"))
    import torch
    from torch._inductor import config
    from torch._inductor.utils import run_and_get_code
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention
    from bench_landed import SHAPES, TOLERANCES
    from gpu_monitor import Monitor

    if not torch.version.hip or torch.cuda.device_count() != 1:
        raise RuntimeError("Expected one visible ROCm GPU")
    context = torch.ones(1, device="cuda")
    torch.cuda.synchronize()
    monitor = Monitor()
    initial = monitor.wait_idle()
    patches = {"max_autotune": False, "max_autotune_gemm": False, "fx_graph_cache": False}
    if hasattr(config, "gluon_flex_attention"):
        patches["gluon_flex_attention"] = False
    records = []
    with config.patch(patches), torch.enable_grad():
        for dtype_name in ("float16", "bfloat16"):
            for batch, heads, sequence, dim in SHAPES:
                for mask in ("nomask", "causal"):
                    idle = monitor.wait_idle()
                    torch.manual_seed(0)
                    q, k, v = (torch.randn(batch, heads, sequence, dim, device="cuda",
                                           dtype=getattr(torch, dtype_name)) for _ in range(3))
                    bm = (create_block_mask(lambda b, h, m, n: m >= n, batch, heads,
                                            sequence, sequence, device="cuda")
                          if mask == "causal" else None)

                    def fn(q, k, v):
                        return flex_attention(q, k, v, block_mask=bm, scale=dim**-0.5)

                    reference = fn(q, k, v)
                    torch._dynamo.reset()
                    compiled = torch.compile(fn, mode="default", fullgraph=True)
                    out, generated = run_and_get_code(compiled, q, k, v)
                    torch.cuda.synchronize()
                    if not torch.isfinite(out).all().item():
                        raise RuntimeError("Nonfinite output")
                    torch.testing.assert_close(out, reference, **TOLERANCES[dtype_name])
                    source = "\n".join(generated)
                    headers = re.findall(r"@triton_heuristics\.template\((.*?)\)\s*@triton\.jit",
                                         source, re.S)
                    settings = sorted({(int(re.search(r"num_stages\s*=\s*(\d+)", h)[1]),
                                        int(re.search(r"num_warps\s*=\s*(\d+)", h)[1]))
                                       for h in headers})
                    lse = {value == "True" for value in re.findall(
                        r"OUTPUT_LOGSUMEXP\s*:\s*tl.constexpr\s*=\s*(True|False)", source)}
                    if not settings or len(lse) != 1:
                        raise RuntimeError("Cannot establish generated settings")
                    filename = f"{dtype_name}-{sequence}-{dim}-{mask}.py.gz"
                    (artifacts / filename).write_bytes(gzip.compress(source.encode(), mtime=0))
                    row = {"dtype": dtype_name, "sequence_length": sequence, "head_dim": dim,
                           "mask": mask, "compiler_settings": settings, "grad_enabled": True,
                           "output_logsumexp": lse.pop(), "correctness_passed": True,
                           "max_abs_diff": (out.float() - reference.float()).abs().max().item(),
                           "pre_compile_idle_samples": idle, "post_compile_state": monitor.snapshot(),
                           "generated_source": filename,
                           "generated_source_sha256": hashlib.sha256(source.encode()).hexdigest()}
                    records.append(row)
                    print(f"DEFAULT {dtype_name} S={sequence} D={dim} {mask}: "
                          f"stages/warps={settings}, OUTPUT_LOGSUMEXP={row['output_logsumexp']}, "
                          f"max_abs_diff={row['max_abs_diff']:.6g}", flush=True)
    result = {"torch": torch.__version__, "torch_git": torch.version.git_version,
              "initial_gpu_state": initial, "inductor_overrides": patches,
              "kernel_options": {}, "checks": records, "complete": True}
    (root / "defaults.json").write_text(json.dumps(result, indent=2) + "\n")
    del context


if __name__ == "__main__":
    main()
