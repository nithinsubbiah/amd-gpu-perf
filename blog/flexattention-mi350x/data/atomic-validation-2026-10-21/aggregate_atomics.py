"""Audit retained atomic-guard evidence and aggregate complete process pairs."""
import ast
import gzip
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'support'))
from gpu_monitor import validate_clocks


def templates(code):
    result = {}
    for node in ast.walk(ast.parse(code)):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'triton' and len(node.args) > 1
                and isinstance(node.args[0], ast.Constant) and isinstance(node.args[1], ast.Constant)):
            name, source = node.args[0].value, node.args[1].value
            if '_tem_' in name:
                result[name] = source[source.index('def '):]
    if len(result) != 2:
        raise RuntimeError(f'Expected forward and backward templates, got {list(result)}')
    return result


def load(name, timed=True):
    directory = ROOT/name
    complete = json.loads((directory/'complete.json').read_text())
    env = json.loads((directory/'environment.json').read_text())
    rows = [json.loads(line) for line in (directory/'results.jsonl').read_text().splitlines()]
    if complete['measurements'] != 2 or {r['kind'] for r in rows} != {'head','query'}:
        raise RuntimeError(f'Incomplete process {name}')
    output = {}
    for row in rows:
        code = gzip.decompress((directory/f'{row["kind"]}-code.py.gz').read_bytes())
        if hashlib.sha256(code).hexdigest() != row['code_sha256']:
            raise RuntimeError('Generated-code hash mismatch')
        if not all(row['metrics'][key]['assert_close_passed'] for key in ('out','dq','dk','dv')):
            raise RuntimeError('Q/K/V reference checks failed')
        if not timed and not row['metrics']['dbias']['assert_close_passed']:
            raise RuntimeError('Nondegenerate captured-gradient reference check failed')
        if timed:
            trial = row['trial_ms']
            if len(trial)!=6 or statistics.median(trial)!=row['median_ms'] or row['spread_pct']>5:
                raise RuntimeError('Invalid timing trial record')
            idle = row['pre_timing_idle']
            if len(idle)<3 or any(s['gpu_use_pct']!=0 for s in idle[-3:]):
                raise RuntimeError('Missing idle evidence')
            expected = validate_clocks(row['monitor'],1000,None)
            if expected!=row['clocks']:
                raise RuntimeError('Clock audit differs from timing record')
            for sample in idle+row['monitor']:
                if sample['pci_bus']!=env['pci'] or len(sample['pids'])!=1:
                    raise RuntimeError('GPU identity or isolation mismatch')
        output[row['kind']] = row
    native = json.loads((directory/'native.json').read_text())
    found = 0
    for entry in native:
        for item in entry['files'].values():
            payload = gzip.decompress((directory/item['path']).read_bytes())
            if hashlib.sha256(payload).hexdigest()!=item['sha256']:
                raise RuntimeError('Compiler-artifact hash mismatch')
        text = gzip.decompress((directory/entry['files']['.amdgcn']['path']).read_bytes()).decode()
        counts={key:len(re.findall(r'^\s*'+key+r'\w*\b',text,re.M)) for key in entry['instructions']}
        if counts!=entry['instructions']:
            raise RuntimeError('Native instruction recount mismatch')
        if counts['buffer_atomic'] or counts['global_atomic']:
            found+=1
            expected = 'buffer_atomic' if name.endswith('before') else 'global_atomic'
            unexpected = 'global_atomic' if name.endswith('before') else 'buffer_atomic'
            if counts[expected]==0 or counts[unexpected]!=0:
                raise RuntimeError('Expected native atomic lowering not observed')
    if found!=2:
        raise RuntimeError('Expected two captured-gradient atomic kernels')
    return env, output


def main():
    names=[f'run{i}-{v}' for i in (1,2,3) for v in ('before','after')]
    loaded={name:load(name) for name in names}
    baseline=loaded[names[0]][0]
    for env, _ in loaded.values():
        for key in ('torch','torch_commit','triton','triton_distribution','hip','python','pci','source_sha256','harness_sha256'):
            if env[key]!=baseline[key]:
                raise RuntimeError(f'Process environments differ: {key}')
    body_matches=[]
    for i in (1,2,3):
        for kind in ('head','query'):
            codes=[gzip.decompress((ROOT/f'run{i}-{variant}'/f'{kind}-code.py.gz').read_bytes()).decode()
                   for variant in ('before','after')]
            if templates(codes[0])!=templates(codes[1]):
                raise RuntimeError('Kernel bodies differ beyond annotation metadata')
            body_matches.append(f'run{i}-{kind}')
    rows={}
    for kind in ('head','query'):
        before=[loaded[f'run{i}-before'][1][kind]['median_ms'] for i in (1,2,3)]
        after=[loaded[f'run{i}-after'][1][kind]['median_ms'] for i in (1,2,3)]
        rows[kind]={'before_ms':before,'after_ms':after,'median_before_ms':statistics.median(before),
                    'median_after_ms':statistics.median(after),
                    'speedup':statistics.median(before)/statistics.median(after),
                    'paired_speedups':[a/b for a,b in zip(before,after)],
                    'bias_max_abs_errors':{variant:[loaded[f'run{i}-{variant}'][1][kind]['metrics']['dbias']['max_abs_error']
                                                  for i in (1,2,3)] for variant in ('before','after')}}
    controls={variant:load('control-'+variant,timed=False)[1] for variant in ('before','after')}
    result={'environment':{k:baseline[k] for k in ('torch','torch_commit','triton_distribution','hip','python','gpu','arch','pci')},
            'workload':{'shape':[1,8,2048,64],'dtype':'float16','mask':'dense','seed':0,'incoming_gradient':'ones'},
            'results':rows,'kernel_body_matches':body_matches,
            'measurements':12,'timing_trials':72,'controls':controls,
            'numerical_scope':'Outputs and Q/K/V pass FP64 tolerances. Additive per-head bias fails strict 0.003 absolute tolerance before and after; true derivative is zero. Nonconstant key-linear controls must pass all gradients.'}
    (ROOT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(rows,indent=2))


if __name__=='__main__':
    main()
