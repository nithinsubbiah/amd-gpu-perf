"""Aggregate the completed independent campaign, retaining failed attempts separately."""
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'

def records(names):
    out = []
    for name in names:
        for line in (RESULTS / name).read_text().splitlines():
            r = json.loads(line)
            r['record_file'] = name
            out.append(r)
    return out

def timings(names, expected):
    selected = {}
    for r in records(names):
        if r['type'] != 'timing':
            continue
        assert r['case'] not in selected, (names, r['case'])
        assert r['clocks']['min_mhz'] >= 1000 and r['clocks']['samples'] >= 2
        assert len(r['idle']) >= 3 and all(s['gpu_use_pct'] == 0 for s in r['idle'][-3:])
        assert all(len(s['pids']) == 1 for s in r['monitor'])
        assert all(len(v) == 6 for v in r['ms'].values())
        selected[r['case']] = r
    assert set(selected) == set(expected), (names, set(selected), set(expected))
    return selected

def correctness(names, expected):
    out = records(names)
    assert not [r for r in out if r['type'] == 'failure'], names
    assert set(r['case'] for r in out if r['type'] == 'correctness') == set(expected)
    assert out[-1]['type'] == 'complete' and not out[-1]['failures']
    return out

merged0 = timings(['merged-0.jsonl', 'merged-0-case2.jsonl'], range(11))
merged1 = timings(['merged-1.jsonl'], range(11))
correctness(['merged-reference-0.jsonl'], range(11))
correctness(['merged-reference-1.jsonl'], range(11))
parent0 = timings(['parent-0.jsonl', 'parent-0-fp16-rerun.jsonl'], range(23))
open0 = timings(['open-0.jsonl', 'open-0-resumed.jsonl'], range(23))
correctness(['open-reference-1.jsonl'], range(23))
repeat = [0, 1, 3, 8, 10, 15, 17]
parent1 = timings(['parent-repeat-1.jsonl'], repeat)
open1 = timings(['open-repeat-1.jsonl'], repeat)

summary = {'merged': [], 'open': []}
for i in range(11):
    runs = [merged0[i], merged1[i]]
    medians = {key: statistics.median(r['median_ms'][key] for r in runs)
               for key in runs[0]['median_ms']}
    ratios = {key: statistics.median(r['paired_median_speedup'][key] for r in runs)
              for key in runs[0]['paired_median_speedup']}
    summary['merged'].append({'case': i, 'row': runs[0]['row'], 'median_ms': medians,
        'speedup': ratios, 'per_process_speedups': [r['paired_median_speedup'] for r in runs],
        'record_files': [r['record_file'] for r in runs]})
for i in range(23):
    p, o = parent0[i], open0[i]
    item = {'case': i, 'row': o['row'], 'parent_ms': p['median_ms']['parent'],
            'pr_ms': o['median_ms'],
            'speedup_vs_parent': {key: p['median_ms']['parent'] / ms
                                  for key, ms in o['median_ms'].items()},
            'paired_speedup_vs_pr_optout': o['paired_median_speedup'],
            'record_files': [p['record_file'], o['record_file']]}
    if i in parent1:
        p1, o1 = parent1[i], open1[i]
        item['repeat'] = {'parent_ms': p1['median_ms']['parent'], 'pr_ms': o1['median_ms'],
            'speedup_vs_parent': {key: p1['median_ms']['parent'] / ms
                                  for key, ms in o1['median_ms'].items()},
            'record_files': [p1['record_file'], o1['record_file']]}
    summary['open'].append(item)

all_timings = [*merged0.values(), *merged1.values(), *parent0.values(), *open0.values(),
               *parent1.values(), *open1.values()]
summary['quality'] = {
    'timed_workload_runs': len(all_timings),
    'min_loaded_sclk_mhz': min(r['clocks']['min_mhz'] for r in all_timings),
    'max_loaded_sclk_mhz': max(r['clocks']['max_mhz'] for r in all_timings),
    'max_within_process_latency_spread_pct': max(v for r in all_timings for v in r['spread_pct'].values()),
    'failed_attempts': [{k: r[k] for k in ('case', 'row', 'error', 'record_file')}
                        for r in records(['merged-0.jsonl', 'parent-0.jsonl']) if r['type'] == 'failure'],
}
all_correct = records(['merged-0.jsonl', 'merged-0-case2.jsonl', 'merged-1.jsonl',
                      'merged-reference-0.jsonl', 'merged-reference-1.jsonl',
                      'parent-0.jsonl', 'parent-0-fp16-rerun.jsonl', 'parent-repeat-1.jsonl',
                      'open-0.jsonl', 'open-0-resumed.jsonl', 'open-reference-1.jsonl', 'open-repeat-1.jsonl'])
summary['quality']['max_relative_l2_by_dtype'] = {
    dtype: max(m['relative_l2'] for r in all_correct if r['type'] == 'correctness' and r['row'][0] == dtype
               for m in r['metrics'].values())
    for dtype in ('bfloat16', 'float16', 'float32')}
for campaign in ('parent', 'open'):
    mapped = json.loads((RESULTS / f'{campaign}-0-native-map.json').read_text())
    assert {r['case'] for r in mapped} == set(range(23))
    assert not any(m for r in mapped for k in r['kernels'] for m in k['option_mismatches'])
    if campaign == 'open':
        for r in mapped:
            phases = {k['phase'] for k in r['kernels']}
            assert phases == ({0} if r['variant'] == 'stock' else {1, 2})
            if r['variant'] != 'stock':
                assert all(v == 0 for k in r['kernels'] for n in k['native']
                           for v in n['resources']['private_segment_fixed_size'])
summary['quality']['native_option_and_phase_audit'] = 'passed'
(RESULTS / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary['quality'], indent=2))
print('MERGED')
for r in summary['merged']:
    print(r['case'], r['row'][:-1], r['speedup'])
print('OPEN: true parent / automatic, seed 0; seed 1 where repeated')
for r in summary['open']:
    print(r['case'], round(r['speedup_vs_parent']['automatic'], 4),
          round(r['repeat']['speedup_vs_parent']['automatic'], 4) if 'repeat' in r else '')
