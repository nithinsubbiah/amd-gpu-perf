"""Run one publication process with fresh caches and retain compiler artifacts."""

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import sys


def archive_codegen(root):
    records = []
    target = root / "codegen"
    target.mkdir(exist_ok=True)
    for path in sorted((root / "triton-cache").rglob("*")):
        if path.suffix not in {".amdgcn", ".ttgir", ".ttir", ".llir", ".json"}:
            continue
        data = path.read_bytes()
        name = f"{path.parent.name}-{path.name}.gz"
        (target / name).write_bytes(gzip.compress(data, mtime=0))
        record = {"path": f"codegen/{name}", "cache_key": path.parent.name,
                  "sha256": hashlib.sha256(data).hexdigest()}
        if path.suffix == ".amdgcn":
            source = data.decode()
            record["instructions"] = {
                op: len(re.findall(rf"^\s*{op}\w*\s", source, re.M))
                for op in ("buffer_load", "buffer_store", "global_load", "global_store",
                           "v_mfma", "ds_write")}
            record["vgprs"] = re.findall(r"\.amdhsa_next_free_vgpr\s+(\d+)", source)
        records.append(record)
    (root / "codegen.json").write_text(json.dumps(records, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--device", type=int, default=6)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--reverse-order", action="store_true")
    parser.add_argument("--reps", type=int, default=6,
                        help="Use a longer monitoring window while retaining every trial")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", args.label):
        parser.error("Use a simple unique run label")
    campaign = Path(__file__).resolve().parent
    output = campaign / f"{args.label}.jsonl"
    if output.exists():
        parser.error("The output file already exists")
    root = campaign / args.label
    root.mkdir()
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(root / "inductor-cache")
    os.environ["TRITON_CACHE_DIR"] = str(root / "triton-cache")
    # Cache locations differ between processes; the harness records their actual paths.
    sys.path.insert(0, str(campaign.parents[1] / "bench"))
    import bench_landed

    sys.argv = [sys.argv[0], "--device", str(args.device), "--output", str(output),
                "--reps", str(args.reps)]
    if args.smoke:
        sys.argv.append("--smoke")
    if args.reverse_order:
        sys.argv.append("--reverse-order")
    try:
        bench_landed.main()
    finally:
        archive_codegen(root)


if __name__ == "__main__":
    main()
