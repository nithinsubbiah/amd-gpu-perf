"""Associate generated backward programs with retained Triton compiler artifacts.

Uses the source filename embedded in Triton IR, not a potentially reused kernel
name. This is a read-only audit and does not import torch or initialize a GPU.
"""
import argparse
import ast
import gzip
import hashlib
import json
from pathlib import Path
import re

p = argparse.ArgumentParser()
p.add_argument('--cache', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('records', nargs='+', type=Path)
args = p.parse_args()
by_source = {}
for path in args.cache.rglob('*.source'):
    text = path.read_text()
    match = re.search(r'^#loc = loc\("([^"]+\.py)"', text, re.M)
    meta_path = path.with_suffix('.json')
    if not match or not meta_path.exists():
        continue
    meta = json.loads(meta_path.read_text())
    assembly = path.with_suffix('.amdgcn')
    asm = assembly.read_text()
    resource = {key: [int(v) for v in re.findall(r'\.amdhsa_' + key + r'\s+(\d+)', asm)]
                for key in ('next_free_vgpr', 'next_free_sgpr',
                            'private_segment_fixed_size', 'group_segment_fixed_size')}
    source_path = Path(match[1])
    if not source_path.exists():
        raise AssertionError(f'Missing compiled source {source_path}')
    source_key = hashlib.sha256(source_path.read_text().strip().encode()).hexdigest()
    by_source.setdefault(source_key, []).append({
        'compiled_source': str(source_path),
        'metadata_file': str(meta_path),
        'options': {key: meta[key] for key in ('num_warps', 'num_stages',
                    'matrix_instr_nonkdim', 'kpack', 'waves_per_eu')},
        'resources': resource,
        'assembly_sha256': hashlib.sha256(assembly.read_bytes()).hexdigest(),
        'mfma_instructions': sorted(set(re.findall(r'\bv_mfma_\w+', asm))),
    })
out = []
for recordfile in args.records:
    for line in recordfile.read_text().splitlines():
        record = json.loads(line)
        if record['type'] != 'correctness':
            continue
        generated = recordfile.parent / record['generated_code']
        code = gzip.decompress(generated.read_bytes()).decode()
        kernels = []
        for node in ast.walk(ast.parse(code)):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == 'triton' and len(node.args) >= 2):
                continue
            section = ast.literal_eval(node.args[1])
            path = ast.literal_eval(node.args[0])
            if '@triton_heuristics.template(' not in section:
                continue
            phase = re.search(r'ROCM_BWD_PHASE\s*:\s*tl.constexpr\s*=\s*(\d+)', section)
            constants = dict(re.findall(r'^\s*(BLOCK_[MN][12]|ROCM_[A-Z0-9_]+|XCD_REMAP)\s*:\s*tl.constexpr\s*=\s*([^\n#]+)', section, re.M))
            declared = {}
            for key in ('num_warps', 'num_stages'):
                m = re.search(r'\b' + key + r'=(\d+)', section)
                if m:
                    declared[key] = int(m[1])
            for key in ('matrix_instr_nonkdim', 'kpack', 'waves_per_eu'):
                m = re.search(r"'" + key + r"': (\d+)", section)
                if m:
                    declared[key] = int(m[1])
            source_key = hashlib.sha256(section.strip().encode()).hexdigest()
            native = by_source.get(source_key, [])
            if not native:
                raise AssertionError(f'No native compiler artifacts for {path}')
            kernels.append({'source': path, 'phase': int(phase[1]) if phase else None,
                            'constants': constants, 'declared_options': declared,
                            'native': native,
                            'option_mismatches': [{key: [value, n['options'][key]]
                                                  for key, value in declared.items()
                                                  if value != n['options'][key]}
                                                 for n in native]})
        if not kernels:
            raise AssertionError(f'No template kernels in {generated}')
        out.append({'record_file': str(recordfile), 'case': record['case'],
                    'variant': record['variant'], 'kernels': kernels})
args.output.write_text(json.dumps(out, indent=2) + '\n')
print(f'Mapped {len(out)} backward programs to native artifacts')
