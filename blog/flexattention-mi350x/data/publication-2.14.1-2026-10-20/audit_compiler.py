"""Associate measured workloads with retained compiler IR and assembly."""

import gzip
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parent


def unpack(path, digest):
    data = gzip.decompress(path.read_bytes())
    if hashlib.sha256(data).hexdigest() != digest:
        raise RuntimeError(f"Artifact checksum mismatch: {path}")
    return data.decode()


def inspect_run(label):
    run = ROOT / label
    artifacts = json.loads((run / "codegen.json").read_text())
    by_path = {item["path"]: item for item in artifacts}
    kernels = []
    for item in artifacts:
        if not item["path"].endswith(".ttgir.gz"):
            continue
        stem = item["path"][:-len(".ttgir.gz")]
        metadata_item = by_path[stem + ".json.gz"]
        metadata = json.loads(unpack(run / metadata_item["path"], metadata_item["sha256"]))
        if not metadata.get("name", "").startswith("triton_tem_"):
            continue
        if "flex_attention" not in metadata["name"]:
            continue
        ir = unpack(run / item["path"], item["sha256"])
        assembly_item = by_path[stem + ".amdgcn.gz"]
        assembly = unpack(run / assembly_item["path"], assembly_item["sha256"])
        instructions = {
            op: len(re.findall(rf"^\s*{op}\w*\s", assembly, re.M))
            for op in ("v_mfma", "ds_write", "buffer_load", "buffer_store",
                       "global_load", "global_store")
        }
        kernels.append({
            "source_paths": set(re.findall(r'loc\("([^\"]+\.py)":', ir)),
            "metadata": metadata,
            "evidence": {
                "cache_key": item["cache_key"],
                "ttgir": f"{label}/{item['path']}",
                "ttgir_sha256": item["sha256"],
                "amdgcn": f"{label}/{assembly_item['path']}",
                "amdgcn_sha256": assembly_item["sha256"],
                "instructions": instructions,
                "wave_weighted_mfma_sites": instructions["v_mfma"] * metadata["num_warps"],
                "mfma_opcodes": sorted(set(re.findall(r"^\s*(v_mfma\w*)\s", assembly, re.M))),
                "mfma_layouts": [line for line in ir.splitlines() if "#ttg.amd_mfma<" in line],
                "scf_for_sites": len(re.findall(r"\bscf\.for\b", ir)),
                "vgprs": re.findall(r"\.amdhsa_next_free_vgpr\s+(\d+)", assembly),
                "shared_bytes": metadata["shared"],
            },
        })
    records = [json.loads(line) for line in (ROOT / f"{label}.jsonl").read_text().splitlines()]
    if records[-1] != {"type": "complete", "measurements": 36}:
        raise RuntimeError(f"Incomplete run: {label}")
    measured = [item for item in records if item["type"] == "measurement"]
    result = []
    for row in measured:
        generated = unpack(ROOT / row["generated_source"], row["generated_source_sha256"])
        matches = [kernel for kernel in kernels
                   if any(f"# kernel path: {p}\n" in generated for p in kernel["source_paths"])]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one attention kernel for {row['generated_source']}: {len(matches)}")
        kernel = matches[0]
        if any(kernel["metadata"][key] != row[key] for key in ("num_stages", "num_warps")):
            raise RuntimeError("Measured and compiled settings differ")
        result.append({
            "run": label,
            **{key: row[key] for key in ("dtype", "sequence_length", "head_dim", "mask",
                                       "num_stages", "num_warps", "generated_source")},
            **kernel["evidence"],
        })
    if len(result) != 36:
        raise RuntimeError(f"Expected 36 measurements: {label}")
    return result


def main():
    records = [row for label in ("run1", "run2", "run3") for row in inspect_run(label)]
    historical = json.loads((ROOT.parent / "codegen/counts.json").read_text())
    cases = []
    for warps in (4, 8):
        selected = [row for row in records
                    if (row["dtype"], row["sequence_length"], row["head_dim"], row["mask"],
                        row["num_stages"], row["num_warps"]) ==
                    ("float16", 4096, 128, "nomask", 2, warps)]
        if len(selected) != 3:
            raise RuntimeError("Missing historical-comparison kernel")
        keys = ("instructions", "wave_weighted_mfma_sites", "mfma_opcodes", "mfma_layouts",
                "scf_for_sites")
        if any(any(row[key] != selected[0][key] for key in keys) for row in selected[1:]):
            raise RuntimeError("Compiler mechanism differs between processes")
        previous = historical[str(warps)]
        historical_ir = unpack(ROOT.parent / f"codegen/warps{warps}.ttgir.gz",
                               previous["ttgir_sha256"])
        historical_assembly = unpack(ROOT.parent / f"codegen/warps{warps}.amdgcn.gz",
                                     previous["amdgcn_sha256"])
        mfma = len(re.findall(r"^\s*v_mfma\w*\s", historical_assembly, re.M))
        writes = len(re.findall(r"^\s*ds_write\w*\s", historical_assembly, re.M))
        if (mfma != previous["mfma_sites"] or mfma * warps != previous["wave_weighted_mfma_sites"]
                or writes != previous["ds_write_sites"]):
            raise RuntimeError("Historical instruction counts differ from retained assembly")
        cases.append({"num_warps": warps,
                      "historical": {**previous,
                                     "mfma_layouts": [line for line in historical_ir.splitlines()
                                                      if "#ttg.amd_mfma<" in line],
                                     "scf_for_sites": len(re.findall(r"\bscf\.for\b", historical_ir))},
                      "publication": {key: selected[0][key] for key in keys},
                      "artifacts": [{"run": row["run"], "ttgir": row["ttgir"],
                                     "amdgcn": row["amdgcn"]} for row in selected]})
    output = {"measurements_audited": len(records), "historical_comparison": cases,
              "scope": "Static instruction sites; dynamic counts depend on loops and masks.",
              "measurements": records}
    with (ROOT / "compiler-audit.json").open("x") as stream:
        stream.write(json.dumps(output, indent=2) + "\n")
    print(json.dumps({key: output[key] for key in ("measurements_audited", "historical_comparison")}, indent=2))


if __name__ == "__main__":
    main()
