"""Inventory retained Triton native binaries and compiler settings without executing them."""
import argparse
import hashlib
import json
from pathlib import Path
import re

p = argparse.ArgumentParser()
p.add_argument("cache", type=Path)
p.add_argument("output", type=Path)
args = p.parse_args()
records = []
for path in sorted(args.cache.rglob("*.json")):
    meta = json.loads(path.read_text())
    if not isinstance(meta, dict) or "name" not in meta or "target" not in meta:
        continue
    files = {}
    asm_text = ""
    for artifact in sorted(path.parent.iterdir()):
        if artifact.suffix in (".json", ".amdgcn", ".hsaco", ".llir", ".ttgir", ".ttir", ".source"):
            files[artifact.name] = {"sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(), "bytes": artifact.stat().st_size}
        if artifact.suffix == ".amdgcn":
            asm_text = artifact.read_text()
    resources = {}
    for name in ("next_free_vgpr", "next_free_sgpr", "private_segment_fixed_size", "group_segment_fixed_size", "accum_offset"):
        resources[name] = [int(v) for v in re.findall(r"\.amdhsa_" + name + r"\s+(\d+)", asm_text)]
    records.append({"path": str(path), "metadata": meta, "resources": resources, "artifacts": files})
args.output.write_text(json.dumps(records, indent=2) + "\n")
print(f"Recorded {len(records)} compiled kernels")
