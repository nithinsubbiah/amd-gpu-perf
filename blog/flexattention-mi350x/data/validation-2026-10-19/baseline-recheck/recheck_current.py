"""Run the revised four-configuration smoke check with a private compiler cache."""

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--device", type=int, default=6)
parser.add_argument("--label", required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parent
artifacts = root / args.label
artifacts.mkdir()
os.environ["HIP_VISIBLE_DEVICES"] = str(args.device)
os.environ["TRITON_CACHE_DIR"] = str(artifacts / "triton-cache")
os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(artifacts / "inductor-cache")
sys.path.insert(0, str(root.parents[2] / "bench"))
import bench_landed  # noqa: E402

sys.argv = [sys.argv[0], "--smoke", "--reps", "5", "--output", str(root / f"{args.label}.jsonl")]
bench_landed.main()
assembly = []
for path in sorted((artifacts / "triton-cache").rglob("*flex_attention*.amdgcn")):
    data = path.read_bytes()
    target = artifacts / f"{path.parent.name}-{path.name}.gz"
    target.write_bytes(gzip.compress(data, mtime=0))
    assembly.append({"path": str(target.relative_to(root)),
                     "sha256": hashlib.sha256(data).hexdigest(),
                     "metadata": json.loads(path.with_suffix(".json").read_text())})
(artifacts / "assembly.json").write_text(json.dumps(assembly, indent=2) + "\n")
