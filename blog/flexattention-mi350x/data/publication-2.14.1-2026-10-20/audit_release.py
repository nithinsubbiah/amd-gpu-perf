"""Inspect the installed release's actual default tables and template controls."""

import ast
import gzip
import hashlib
import json
from pathlib import Path

import torch


def main():
    root = Path(__file__).resolve().parent
    torch_root = Path(torch.__file__).parent
    paths = {
        "heuristics": torch_root / "_inductor/heuristics/template/triton.py",
        "template_metadata": torch_root / "_inductor/select_algorithm.py",
        "pointer_metadata": torch_root / "_inductor/codegen/triton_utils.py",
        "frontend": torch_root / "nn/attention/flex_attention.py",
    }
    sources = {name: path.read_text() for name, path in paths.items()}
    tree = ast.parse(sources["heuristics"])
    tables = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Attribute) or target.attr not in {
            "gfx950_default_flex_config", "gfx942_default_flex_config"}:
            continue
        entries = []
        for key, value in zip(node.value.keys, node.value.values):
            dtype, dim = key.elts
            entries.append({"dtype": dtype.attr, "head_dim": ast.literal_eval(dim),
                            "BLOCK_M": ast.literal_eval(value.args[0]),
                            "BLOCK_N": ast.literal_eval(value.args[1]),
                            "num_stages": ast.literal_eval(value.args[2]),
                            "num_warps": ast.literal_eval(value.args[3])})
        tables[target.attr] = entries
    if len(tables) != 2:
        raise RuntimeError("Cannot identify both architecture-specific default tables")
    for row in tables["gfx950_default_flex_config"]:
        if row["dtype"] in {"float16", "bfloat16"} and row["head_dim"] in {64, 128}:
            if [row[key] for key in ("BLOCK_M", "BLOCK_N", "num_stages", "num_warps")] != [128, 64, 2, 4]:
                raise RuntimeError("The release default differs from the article's updated configuration")
    metadata_tree = ast.parse(sources["template_metadata"])
    jit_lines = [node for node in ast.walk(metadata_tree)
                 if isinstance(node, ast.FunctionDef) and node.name == "jit_lines"]
    if len(jit_lines) != 1:
        raise RuntimeError("Cannot identify template metadata construction")
    jit_source = ast.get_source_segment(sources["template_metadata"], jit_lines[0])
    result = {
        "torch": torch.__version__, "torch_git": torch.version.git_version,
        "source_audit_only": True, "gpu_compilation_verified": False,
        "tables": tables,
        "template_config_call": "config_of(signature)" in jit_source,
        "template_shared_pointer_helper": "pointer_range_kwargs" in jit_source,
        "template_pointer_override": "pointer_range_override" in jit_source,
        "frontend_grad_controls_lse": 'kernel_options["OUTPUT_LOGSUMEXP"] = torch.is_grad_enabled()'
        in sources["frontend"],
        "source_sha256": {name: hashlib.sha256(source.encode()).hexdigest()
                          for name, source in sources.items()},
    }
    for name, source in sources.items():
        (root / f"source/release-{name}.py.gz").write_bytes(gzip.compress(source.encode(), mtime=0))
    (root / "environment/release-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
