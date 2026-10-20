"""Run the original script's first workload without changing its timing loop.

Only the shape/mask/dtype sweep is restricted. Monitoring encloses the whole
experiment, and the original do_bench calls are intercepted to retain trials.
"""

import argparse
import ast
import gzip
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--device", type=int, default=6)
parser.add_argument("--label", required=True)
parser.add_argument("--cache", choices=("existing", "fresh"), default="fresh")
parser.add_argument("--no-monitor", action="store_true")
parser.add_argument("--no-grad", action="store_true")
args = parser.parse_args()
root = Path(__file__).resolve().parent
output = root / f"{args.label}.json"
if output.exists():
    parser.error("Use a new output label")
os.environ["HIP_VISIBLE_DEVICES"] = str(args.device)
cache_root = root / args.label
cache_root.mkdir()
if args.cache == "fresh":
    os.environ["TRITON_CACHE_DIR"] = str(cache_root / "triton-cache")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache_root / "inductor-cache")
sys.path.insert(0, str(root.parents[2] / "bench"))
from gpu_monitor import Monitor  # noqa: E402

import torch  # noqa: E402
import triton  # noqa: E402

torch.ones(1, device="cuda")
torch.cuda.synchronize()
monitor = Monitor()
idle = monitor.wait_idle()
original = gzip.decompress((root / "source/historical_bench_landed.py.gz").read_bytes()).decode()
tree = ast.parse(original)
for node in tree.body:
    if not isinstance(node, ast.Assign):
        continue
    names = [t.id for t in node.targets if isinstance(t, ast.Name)]
    if "SHAPES" in names:
        node.value = ast.parse("[(1,16,4096,64)]", mode="eval").body
    elif "MASKS" in names:
        node.value = ast.parse("['nomask']", mode="eval").body
# Restrict the outer dtype loop, retaining all four fp16 configurations.
for node in tree.body:
    if isinstance(node, ast.For) and isinstance(node.target, ast.Name) and node.target.id == "dtype":
        node.iter = ast.parse("(torch.float16,)", mode="eval").body
ast.fix_missing_locations(tree)
scope = {"__name__": "__main__"}
records = []
bench = triton.testing.do_bench


def retain(fn, *a, **kw):
    value = float(bench(fn, *a, **kw))
    records.append({"options": dict(scope["opts"]), "median_ms": value})
    return value


triton.testing.do_bench = retain
sys.argv = ["original_bench_landed.py", "--reps", "5"]
torch.set_grad_enabled(not args.no_grad)
try:
    if args.no_monitor:
        before = monitor.snapshot()
        exec(compile(tree, "original_bench_landed.py", "exec"), scope)
        samples = [before, monitor.snapshot()]
    else:
        with monitor:
            exec(compile(tree, "original_bench_landed.py", "exec"), scope)
        samples = monitor.samples
    torch.testing.assert_close(scope["out"], scope["ref"], atol=0.003, rtol=0.01)
finally:
    triton.testing.do_bench = bench
artifacts = []
if args.cache == "fresh":
    for path in sorted((cache_root / "triton-cache").rglob("*flex_attention*.amdgcn")):
        data = path.read_bytes()
        target = cache_root / f"{path.parent.name}-{path.name}.gz"
        target.write_bytes(gzip.compress(data, mtime=0))
        meta = json.loads(path.with_suffix(".json").read_text())
        artifacts.append({"path": str(target.relative_to(root)),
                          "sha256": hashlib.sha256(data).hexdigest(),
                          "metadata": meta})
summary = []
for stages, warps in ((1, 8), (2, 8), (2, 4), (1, 4)):
    trials = [r["median_ms"] for r in records if
              (r["options"]["num_stages"], r["options"]["num_warps"]) == (stages, warps)]
    assert len(trials) == 5
    ms = statistics.median(trials)
    summary.append({"stages": stages, "warps": warps, "trials_ms": trials,
                    "tflops": 4 * 16 * 4096**2 * 64 / ms * 1e-9})
output.write_text(json.dumps({"args": vars(args), "torch": torch.__version__,
                             "triton": triton.__version__, "idle": idle,
                             "monitor_samples": samples, "results": summary,
                             "assembly": artifacts,
                             "original_script_sha256": hashlib.sha256(original.encode()).hexdigest()}, indent=2) + "\n")
print("RECHECK", json.dumps(summary), flush=True)
