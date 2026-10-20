"""Controlled pointer-range ablation using the forward harness's timing checks.

Run each on/off case in a fresh process. The flag controls Inductor's annotation;
the saved assembly must also demonstrate a change in the resulting memory path.
"""

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import sys

import bench_landed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pointer-range", choices=("on", "off"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    if args.output.exists() or args.output.with_suffix("").exists():
        parser.error("Use a new output file and artifact directory")
    root = args.output.with_suffix("").resolve()
    root.mkdir(parents=True)
    os.environ["TORCHINDUCTOR_EMIT_POINTER_RANGE_32"] = "1" if args.pointer_range == "on" else "0"
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(root / "inductor-cache")
    os.environ["TRITON_CACHE_DIR"] = str(root / "triton-cache")
    bench_landed.SHAPES = [(1, 16, 4096, 128)]
    bench_landed.LADDER = {"float16": [(2, 4)]}
    sys.argv = [sys.argv[0], "--output", str(args.output), *remaining]
    bench_landed.main()

    records = [json.loads(line) for line in args.output.read_text().splitlines()]
    if records[-1] != {"type": "complete", "measurements": 2}:
        raise RuntimeError("Incomplete addressing ablation")
    measurements = records[1:-1]
    for row in measurements:
        source = gzip.decompress((args.output.parent / row["generated_source"]).read_bytes()).decode()
        annotated = "tt.pointer_range" in source
        if annotated != (args.pointer_range == "on"):
            raise RuntimeError("Generated pointer-range annotation does not match the requested ablation")

    artifacts = []
    assembly_dir = root / "assembly"
    assembly_dir.mkdir()
    for path in sorted((root / "triton-cache").rglob("*flex_attention*.amdgcn")):
        source = path.read_bytes()
        counts = {op: len(re.findall(rf"^\s*{op}\w*\s", source.decode(), re.M))
                  for op in ("buffer_load", "buffer_store", "global_load", "global_store")}
        target = assembly_dir / f"{path.parent.name}-{path.name}.gz"
        target.write_bytes(gzip.compress(source, mtime=0))
        artifacts.append({"path": str(target.relative_to(root)),
                          "sha256": hashlib.sha256(source).hexdigest(), **counts})
    if len(artifacts) != 2:
        raise RuntimeError(f"Expected two attention assembly artifacts; found {len(artifacts)}")
    for artifact in artifacts:
        if args.pointer_range == "on" and artifact["buffer_load"] == 0:
            raise RuntimeError("Annotated attention kernel did not use buffer loads")
        if args.pointer_range == "off" and (artifact["buffer_load"] != 0 or artifact["global_load"] == 0):
            raise RuntimeError("Unannotated attention kernel did not switch to global loads")
    manifest = {"pointer_range": args.pointer_range,
                "scope": "B=1 H=16 S=4096 D=128 fp16, unmasked and causal, stages=2 warps=4",
                "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "assembly": artifacts, "memory_path_verified": True}
    (root / "ablation.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Addressing ablation {args.pointer_range}: annotation and assembly verified", flush=True)


if __name__ == "__main__":
    main()
