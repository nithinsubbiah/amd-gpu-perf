"""Independently measure the template atomic guard on one fixed release build.

Only the after process restores atomic tracking and template annotation suppression.
The installed package files remain untouched. All source transformations, generated
code, native artifacts, reference errors, raw times and GPU monitoring are retained.
"""
import argparse
from datetime import datetime, timezone
import difflib
import gzip
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import platform
import re
import statistics
import sys
import textwrap
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'support'))
from gpu_monitor import Monitor
from bench_landed import ramp_clocks


def sha(data):
    return hashlib.sha256(data).hexdigest()


def restore_guard(module, directory):
    edits = [
        (module.ModificationWrapper, 'store',
         '        buf_name = self._add_kernel_input(name)',
         '        self.kernel.atomic_add_found = True\n        buf_name = self._add_kernel_input(name)'),
        (module.TritonTemplateKernel, 'jit_lines',
         '        triton_meta["configs"] = [config_of(signature)]',
         '        triton_meta["configs"] = [config_of(signature, pointer_range_override=() if self.atomic_add_found else None)]'),
    ]
    patch = []
    for cls, name, old, new in edits:
        original = inspect.getsource(getattr(cls, name))
        if original.count(old) != 1:
            raise RuntimeError(f'Unexpected official release source: {cls.__name__}.{name}')
        updated = original.replace(old, new)
        label = f'{cls.__name__}.{name}'
        patch.extend(difflib.unified_diff(original.splitlines(True), updated.splitlines(True),
                                        fromfile=label + '.before', tofile=label + '.after'))
        space = {}
        exec(compile(textwrap.dedent(updated), f'<restored-guard:{label}>', 'exec'), vars(module), space)
        setattr(cls, name, space[name])
    (directory / 'guard.patch').write_text(''.join(patch))


def reference(torch, q, k, v, bias, grad, kind, modifier, chunk=256):
    with torch.no_grad():
        q, k, v, bias, grad = [t.detach().double() for t in (q, k, v, bias, grad)]
        n, d = q.shape[-2:]
        output, dq = torch.empty_like(q), torch.empty_like(q)
        dk, dv, db = torch.zeros_like(k), torch.zeros_like(v), torch.zeros_like(bias)
        weights = (torch.arange(n, device=q.device, dtype=torch.float64)/n)[None, None, None, :]
        for start in range(0, n, chunk):
            qq, gg = q[:, :, start:start+chunk], grad[:, :, start:start+chunk]
            bb = bias[None, :, None, None] if kind == 'head' else bias[None, :, start:start+chunk, None]
            scores = (qq @ k.transpose(-1, -2)) * d**-0.5
            scores += bb if modifier == 'additive' else bb * weights
            p = scores.softmax(-1)
            output[:, :, start:start+chunk] = p @ v
            dp = gg @ v.transpose(-1, -2)
            ds = p * (dp - (p * dp).sum(-1, keepdim=True))
            dq[:, :, start:start+chunk] = (ds @ k) * d**-0.5
            dk += (ds.transpose(-1, -2) @ qq) * d**-0.5
            dv += p.transpose(-1, -2) @ gg
            db_tile = (ds if modifier == 'additive' else ds * weights).sum(-1)
            if kind == 'head':
                db += db_tile.sum((0, 2))
            else:
                db[:, start:start+chunk] += db_tile.sum(0)
        return output, dq, dk, dv, db


def reference_self_check(torch):
    torch.manual_seed(11)
    for kind in ('head', 'query'):
        for modifier in ('additive', 'key_linear'):
            q, k, v = [torch.randn(1, 2, 17, 8, dtype=torch.float64, requires_grad=True) for _ in range(3)]
            bias = torch.randn((2,) if kind == 'head' else (2, 17), dtype=torch.float64, requires_grad=True)
            grad = torch.randn_like(q)
            weights = torch.arange(17, dtype=torch.float64)[None, None, None, :] / 17
            bb = bias[None, :, None, None] if kind == 'head' else bias[None, :, :, None]
            scores = (q @ k.transpose(-1, -2)) * 8**-0.5
            scores = scores + (bb if modifier == 'additive' else bb * weights)
            out = scores.softmax(-1) @ v
            expected = (out, *torch.autograd.grad(out, (q, k, v, bias), grad))
            actual = reference(torch, q, k, v, bias, grad, kind, modifier, chunk=7)
            for a, e in zip(actual, expected):
                torch.testing.assert_close(a, e, atol=1e-12, rtol=1e-12)


def metrics(torch, actual, expected):
    result = {}
    for name, a, e in zip(('out', 'dq', 'dk', 'dv', 'dbias'), actual, expected):
        a = a.detach().double()
        diff = a - e
        finite = bool(torch.isfinite(a).all())
        tolerance_ratio = float((diff.abs() / (0.003 + 0.01 * e.abs())).max())
        result[name] = {'finite': finite, 'max_abs_error': float(diff.abs().max()),
                        'reference_max_abs': float(e.abs().max()), 'actual_max_abs': float(a.abs().max()),
                        'relative_l2': float(diff.norm()) / max(float(e.norm()), 1e-30),
                        'max_tolerance_ratio': tolerance_ratio, 'assert_close_passed': finite and tolerance_ratio <= 1.0}
    return result


def native_audit(cache, output):
    records = []
    directory = output / 'native'
    directory.mkdir()
    for path in sorted(cache.glob('**/*.amdgcn')):
        text = path.read_text()
        key = path.parent.name + '-' + path.stem
        counts = {name: len(re.findall(r'^\s*'+name+r'\w*\b', text, re.M))
                  for name in ('buffer_atomic', 'global_atomic', 'buffer_load', 'global_load')}
        files = {}
        for suffix in ('.amdgcn', '.ttir', '.ttgir', '.json'):
            source = path.with_suffix(suffix)
            if source.exists():
                payload = source.read_bytes()
                target = directory / (key + suffix + '.gz')
                target.write_bytes(gzip.compress(payload, mtime=0))
                files[suffix] = {'path': str(target.relative_to(output)), 'sha256': sha(payload)}
        records.append({'kernel': path.stem, 'cache_key': path.parent.name, 'instructions': counts, 'files': files})
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', choices=('before', 'after'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', type=int, default=6)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gradient', choices=('ones', 'random'), default='ones')
    parser.add_argument('--modifier', choices=('additive', 'key_linear'), default='additive')
    parser.add_argument('--correctness-only', action='store_true')
    parser.add_argument('--kinds', default='head,query')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ['HIP_VISIBLE_DEVICES'] = str(args.device)
    os.environ['TORCHINDUCTOR_CACHE_DIR'] = str(args.output / 'inductor-cache')
    os.environ['TRITON_CACHE_DIR'] = str(args.output / 'triton-cache')
    import torch
    import triton
    import triton.testing
    from torch._inductor import config, select_algorithm
    from torch._inductor.utils import run_and_get_code
    from torch.nn.attention.flex_attention import flex_attention
    torch.set_num_threads(4)
    if torch.__version__ != '2.14.1+rocm7.2':
        raise RuntimeError('Expected pinned official release environment')
    reference_self_check(torch)
    if args.variant == 'after':
        restore_guard(select_algorithm, args.output)
    context = torch.ones(1, device='cuda')
    torch.cuda.synchronize()
    monitor = Monitor()
    initial_idle = monitor.wait_idle()
    prop = torch.cuda.get_device_properties(0)
    if torch.cuda.device_count() != 1 or prop.gcnArchName.split(':')[0] != 'gfx950':
        raise RuntimeError('Expected one visible gfx950 GPU')
    sources = [Path(torch.__file__).parent / rel for rel in ('_inductor/select_algorithm.py',
               '_inductor/codegen/triton.py', '_inductor/codegen/triton_utils.py',
               '_inductor/kernel/flex/flex_attention.py', '_inductor/kernel/flex/templates/flex_backwards.py.jinja',
               '_inductor/heuristics/template/triton.py', 'nn/attention/flex_attention.py')]
    environment = {'torch': torch.__version__, 'torch_commit': torch.version.git_version,
                   'hip': torch.version.hip, 'triton': triton.__version__, 'python': platform.python_version(),
                   'triton_distribution': importlib.metadata.version('triton-rocm'),
                   'gpu': prop.name, 'arch': prop.gcnArchName, 'pci': monitor.bus,
                   'initial_idle': initial_idle, 'harness_sha256': sha(Path(__file__).read_bytes()),
                   'source_sha256': {str(p.relative_to(Path(torch.__file__).parent)): sha(p.read_bytes()) for p in sources},
                   'args': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                   'started': datetime.now(timezone.utc).isoformat(), 'reference_cpu_autograd_check': 'passed',
                   'tolerances': {'atol': 0.003, 'rtol': 0.01},
                   'bias_note': 'Additive row-constant bias has true gradient zero; record strict errors separately, without relative-error acceptance.'}
    (args.output / 'environment.json').write_text(json.dumps(environment, indent=2)+'\n')
    results = []
    with (args.output / 'results.jsonl').open('x') as log:
        def emit(row):
            log.write(json.dumps(row, allow_nan=False)+'\n'); log.flush()
        with config.patch({'max_autotune': False, 'max_autotune_gemm': False, 'fx_graph_cache': False}):
            for kind in args.kinds.split(','):
                monitor.wait_idle()
                torch.manual_seed(args.seed)
                q, k, v = [torch.randn(1,8,2048,64,device='cuda',dtype=torch.float16,requires_grad=True) for _ in range(3)]
                bias = torch.randn((8,) if kind=='head' else (8,2048),device='cuda',dtype=torch.float16,requires_grad=True)
                grad = torch.ones_like(q) if args.gradient=='ones' else torch.randn_like(q)
                refs = reference(torch, q,k,v,bias,grad,kind,args.modifier)
                if kind == 'head':
                    def score_mod(score,b,h,m,n):
                        return score + bias[h] if args.modifier=='additive' else score + bias[h]*(n/2048)
                else:
                    def score_mod(score,b,h,m,n):
                        return score + bias[h,m] if args.modifier=='additive' else score + bias[h,m]*(n/2048)
                torch._dynamo.reset()
                compiled = torch.compile(flex_attention, fullgraph=True)
                out, fwd = run_and_get_code(compiled,q,k,v,score_mod=score_mod,scale=64**-0.5,
                                             kernel_options={'BACKEND':'TRITON'})
                def backward():
                    q.grad=k.grad=v.grad=bias.grad=None
                    out.backward(grad,retain_graph=True)
                _, bwd = run_and_get_code(backward)
                torch.cuda.synchronize()
                all_metrics = metrics(torch, (out,q.grad,k.grad,v.grad,bias.grad), refs)
                for part in ('out','dq','dk','dv'):
                    if not all_metrics[part]['assert_close_passed']:
                        raise RuntimeError(f'{part} failed FP64 reference: {all_metrics[part]}')
                if not all_metrics['dbias']['finite']:
                    raise RuntimeError('Nonfinite bias gradient')
                if args.modifier=='key_linear' and not all_metrics['dbias']['assert_close_passed']:
                    emit({'type':'numerical_failure','kind':kind,'metrics':all_metrics})
                    raise RuntimeError(f'Nondegenerate bias gradient failed FP64 reference: {all_metrics["dbias"]}')
                code='\n'.join(fwd+bwd)
                (args.output/f'{kind}-code.py.gz').write_bytes(gzip.compress(code.encode(),mtime=0))
                for name,tensor in zip(('out','dq','dk','dv','dbias'),(out,q.grad,k.grad,v.grad,bias.grad)):
                    if name=='dbias':
                        torch.save({'actual':tensor.detach().cpu(),'reference':refs[-1].cpu()},args.output/f'{kind}-bias-grad.pt')
                row={'type':'measurement','kind':kind,'metrics':all_metrics,'code_sha256':sha(code.encode()),
                     'options':{'BACKEND':'TRITON'},'mask':'dense','shape':[1,8,2048,64]}
                print(f'{args.variant} {kind}: FP64 Q/K/V checks passed; dbias={all_metrics["dbias"]}',flush=True)
                if not args.correctness_only:
                    idle=monitor.wait_idle()
                    ramp_clocks(torch,8.0)
                    deadline=time.monotonic()+3.0
                    while time.monotonic()<deadline:
                        backward();torch.cuda.synchronize()
                    with monitor:
                        timings=[float(triton.testing.do_bench(backward,warmup=200,rep=500,return_mode='median')) for _ in range(6)]
                    median=statistics.median(timings)
                    spread=(max(timings)-min(timings))/median*100
                    row.update({'trial_ms':timings,'median_ms':median,'spread_pct':spread,
                                'pre_timing_idle':idle,'monitor':monitor.samples,'clocks':monitor.clock_validation})
                    emit(row)
                    if spread>5.0:
                        raise RuntimeError(f'Timing spread exceeds preset 5% limit: {spread}')
                    print(f'{args.variant} {kind}: {median:.6f} ms; trial spread {spread:.2f}%',flush=True)
                else:
                    emit(row)
                results.append(row)
    natives=native_audit(Path(os.environ['TRITON_CACHE_DIR']),args.output)
    (args.output/'native.json').write_text(json.dumps(natives,indent=2)+'\n')
    atomic=[r for r in natives if sum(r['instructions'][x] for x in ('buffer_atomic','global_atomic'))]
    expected='buffer_atomic' if args.variant=='before' else 'global_atomic'
    if not atomic or any(not r['instructions'][expected] for r in atomic):
        raise RuntimeError(f'Expected {expected} in captured-gradient native kernels')
    if args.variant=='after' and any(r['instructions']['buffer_atomic'] for r in atomic):
        raise RuntimeError('Guarded kernels still use buffer atomics')
    postflight=monitor.wait_idle()
    (args.output/'complete.json').write_text(json.dumps({'completed':datetime.now(timezone.utc).isoformat(),
       'measurements':len(results),'atomic_kernels':len(atomic),'postflight':postflight},indent=2)+'\n')
    print('COMPLETE',args.output,flush=True)


if __name__=='__main__':
    main()
