# Independent captured-bias atomic-guard validation

Completed 2026-10-21 on AMD Instinct MI350X (gfx950).

## Result

The large performance cliff reported in [PyTorch #194136](https://github.com/pytorch/pytorch/pull/194136)
reproduces. The smaller per-query comparison also improves, with a different
absolute latency and ratio on this pinned release environment.

| Captured bias | PR-reported before / after | Independently measured before / after | Speedup |
|---|---|---|---|
| Per head, eight destinations | 425.2 / 6.8 ms | 425.052 / 6.661 ms | 63.81x |
| Per query, 16,384 destinations | 324.9 / 296.8 µs | 261.967 / 224.127 µs | 1.17x |

Workload: dense attention, `B=1, H=8, S=2048, D=64`, fp16 Q/K/V and bias,
scale `D**-0.5`, seed 0, all-ones incoming output gradient. The timed function
clears leaf gradients and runs the complete backward pass on an existing graph.
Forward, compilation and reference calculations are outside the timer.

The table uses medians across three fresh processes per variant. Each process
measures six `triton.testing.do_bench` medians per workload, with 200 ms warmup
and 500 ms repetitions per trial: **12 measurements, 72 retained trials**.
Process order is before/after, after/before, before/after. All timing spreads
pass the preset 5% limit. Full precision values, individual process medians,
paired ratios and numerical errors are in [summary.json](summary.json).

## Environment and intervention

- Official PyTorch **2.14.1+rocm7.2**, source
  `5c4886908584029761b579af026dcfb627c84070`.
- Official **triton-rocm 3.8.0**, HIP 7.2.53211, host ROCm 7.2.1,
  Python 3.12.3. The [wheel lock](requirements.lock) retains exact artifacts.
- One MI350X, physical PCI `0000:e5:00.0`, selected through
  HIP ordinal 6 and independently mapped by the monitor.
- Default compile mode, explicit `BACKEND="TRITON"`, autotuning disabled.
  Generated backward tiles are `(64,64,64,64)`, one stage and four wavefronts
  in both variants. Forward uses two stages and four wavefronts.

The baseline is the stock release. For the candidate, the harness restores two
changes in memory: captured-buffer atomic stores mark the template as containing
atomics, and that template suppresses `tt.pointer_range` in its argument metadata.
Each candidate process saves the exact `guard.patch`. Installed package files
remain unchanged. This isolates the guard behavior on the same software build;
it is a controlled reproduction of the mechanism, with source hashes retained.

Generated forward and backward template bodies match exactly within every
pair. The native backward code changes from **16 `buffer_atomic_add_f32` sites**
to **16 `global_atomic_add_f32` sites** for each bias shape. IR, assembly,
generated Python and hashes are preserved for every process.

Before each measured workload, the monitor requires three consecutive idle
observations and no competing process on the selected physical GPU. An eight
second dense-matmul ramp and three second workload warmup precede timing.
Process isolation and loaded clocks are monitored during every measurement.
All accepted runs pass those checks, including the 1000 MHz loaded-clock floor.

## Numerical qualification

Outputs and Q/K/V gradients pass an independent chunked FP64 reference with
`atol=0.003, rtol=0.01` for every timed run. The reference implementation is
independently checked against CPU FP64 autograd on small inputs.

Both original score modifiers add a constant across each softmax row:
`score + bias[h]` or `score + bias[h, q]`. Softmax is invariant to such a shift,
so the true captured-bias gradient is zero. The FP64 reference confirms this.
These are useful **synthetic contention stress tests**; the parameter has no
effect on the ideal attention output.

The per-head captured gradient does **not** pass the preset absolute tolerance:
maximum absolute errors are 0.006210–0.012466
before and 0.006241–0.006462 after. Both
versions exhibit cancellation and rounding error. The per-query captured
gradient passes that tolerance. Timing acceptance therefore does not certify
the accuracy of the per-head captured gradient; no tolerance was widened.

As a separate numerical control, `score + bias * (kv_idx / S)` makes the bias
affect relative scores. With seed 1 and a random incoming output gradient,
both bias shapes pass FP64 checks for the output and **all four gradients**
before and after the guard. These four controls are not timed and contribute
no performance claim. Their complete metrics are retained in the summary.

The article's compiler appendix presents the measured per-head improvement as
a contention stress case with this numerical qualification. It does not establish
a general training speedup or arbitrary captured-gradient accuracy.

## Evidence and reproduction

The [raw archive](raw-results.tar.gz) contains complete trial records, monitoring
observations, exact process-local source patches, generated kernels, native
compiler artifacts, logs and captured-gradient tensors. The [manifest](manifest.json)
hashes every bundle file.

To audit and re-aggregate the existing evidence without using a GPU:

```bash
tar -xzf raw-results.tar.gz
python3 aggregate_atomics.py
```

To collect new evidence, use a new directory, the locked release environment
and an exclusively allocated MI350X. For each of three process pairs, run
`bench_atomics.py --variant before` and `--variant after`, with a distinct
absolute `--output` directory and `--device` ordinal. Reverse order in the
second pair. For the two numerical-control processes, also pass
`--modifier key_linear --seed 1 --gradient random --correctness-only`.
The harness refuses to overwrite existing output directories.

The [article](../../README.md#captured-gradient-atomics-a-contention-stress-test)
links this evidence and states the numerical qualification alongside the results.
