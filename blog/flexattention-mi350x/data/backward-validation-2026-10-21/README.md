# Independent backward validation — October 21, 2026

This campaign checks the landed backward-default tuning and open kernel
specialization separately. Results measure the **complete backward pass** on one
AMD Instinct MI350X (gfx950), including delta preprocessing, dQ/dK/dV kernels and
any additional reductions. They do not measure a complete training step.

## Software and availability

| Comparison | Baseline | Candidate | Compiler |
|---|---|---|---|
| Landed default tuning, [#194151](https://github.com/pytorch/pytorch/pull/194151) | Explicit old configuration | Explicit tuned configuration, on the same build | Official PyTorch 2.14.1+rocm7.2 (`5c4886908584029761b579af026dcfb627c84070`), Triton 3.8.0 |
| Open specialization, [#200959](https://github.com/pytorch/pytorch/pull/200959) | Actual parent build `656560bcbe2f0056162828918c6755888e5efe2b` | Actual PR build `d642873429820f9f05d26958a83bc8503e9b528f`, automatic selection | PyTorch 2.16.0a0, Triton 3.9.0+gitfc8d31a5 |

Both campaigns used ROCm 7.2.1 (HIP 7.2.53211) and Python 3.12.3. The development
builds use the exact Triton revision pinned by the PR:
`fc8d31a5439ec9842557a284dc7ca61aee69246b`. Compiler wheel URL and hash, build
flags, native library hashes, and expected upstream ROCm conversion changes are
retained in [source-build-provenance.json](source-build-provenance.json).

The backward defaults landed at
[`1e9d530d4a9`](https://github.com/pytorch/pytorch/commit/1e9d530d4a901a846f06d65f925d4b98fe842b2b),
which is an ancestor of the tested parent. They are absent from the tested
2.14.1 release, so the release experiment sets both configurations explicitly.
The specialization PR was **open** at verification time. See
[upstream-status.json](upstream-status.json). PyTorch's landing workflow closed
#194151 without GitHub marking it merged; its landed commit establishes inclusion.

## Landed backward defaults

Configuration order is `(BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2, stages, warps)`.
Old D64: `(64,64,64,64,1,4)`; old D128: `(64,128,128,64,1,4)`;
tuned: `(32,128,128,32,2,4)` for both dimensions. The first tile pair governs
dK/dV; the second governs dQ.

| Workload | Old ms | Tuned ms | Speedup |
|---|---|---|---|
| bf16, B1, H16/16, S2048, D64, causal | 0.2599 | 0.1448 | 1.79x |
| bf16, B1, H16/16, S4096, D64, causal | 0.7624 | 0.3653 | 2.09x |
| bf16, B1, H16/16, S8192, D64, causal | 2.5375 | 1.1238 | 2.26x |
| bf16, B1, H16/16, S2048, D128, causal | 0.2857 | 0.2431 | 1.18x |
| bf16, B1, H16/16, S4096, D128, causal | 0.7162 | 0.6141 | 1.17x |
| bf16, B1, H16/16, S8192, D128, causal | 2.2654 | 2.0042 | 1.13x |
| bf16, B1, H16/4, S4096, D64, causal | 1.0658 | 0.6958 | 1.53x |
| bf16, B1, H16/4, S4096, D128, causal | 1.5627 | 1.1113 | 1.41x |
| bf16, B1, H16/16, S4096, D128, window1024 | 0.3668 | 0.2744 | 1.34x |
| fp16, B1, H16/16, S4096, D64, causal | 0.7618 | 0.3658 | 2.08x |
| fp16, B1, H16/16, S4096, D128, causal | 0.7180 | 0.6148 | 1.17x |

Latencies are medians of two process medians. Speedup is the median of each
process's median paired ratio. Each process has six alternating-order trials.
Seed 0 uses an all-ones incoming gradient, matching the original benchmark;
seed 1 uses a random incoming gradient. All inputs are regenerated from the seed.
The second process reverses initial variant order. The 8192-token D64 first-seed
timing was rerun after stock-only tolerance calibration, described below.

The original scripts label their `q >= kv` mask as dense. These rows are **causal**.
The window case is causal with `q - 1024 < kv <= q`, and GQA rows have 16 query
heads sharing four KV heads. Their labels above reflect the executed masks.

For bf16, D128, H16/16, S2048/4096/8192, the isolated changes measured:

| Change | Speedup range |
|---|---|
| Tile shape only | 0.86x–0.89x |
| Two pipeline stages only | 0.93x–0.98x |
| Both changes | 1.13x–1.18x |

## Open PR: automatic selection against the actual parent

The primary comparison below uses `kernel_options={"BACKEND": "TRITON"}`
with default compile mode and autotuning disabled. All 23 workloads selected the
specialized dK/dV and dQ phases. Forward outputs and all three gradients passed
the independent reference for **two seeds**. Seven representative workloads
were timed again in fresh processes on both parent and PR builds.

| Case | Workload | Parent ms | PR automatic ms | Seed 0 speedup | Seed 1 speedup | Supplied writeup claim |
|---|---|---|---|---|---|---|
| 0 | fp16, B1, H8/2, S4096, D64, dense | 0.5342 | 0.2371 | 2.25x | 2.24x | — |
| 1 | bf16, B4, H64/8, S4096, D64, dense | 6.9651 | 5.9204 | 1.18x | 1.17x | — |
| 2 | bf16, B4, H21/21, S2048, D96, dense | 2.4369 | 0.9169 | 2.66x | — | 2.82x |
| 3 | bf16, B4, H21/21, S4096, D96, dense | 9.4093 | 3.2897 | 2.86x | 2.85x | 2.97x |
| 4 | bf16, B2, H8/2, S2048, D96, dense | 0.6983 | 0.5351 | 1.31x | — | — |
| 5 | bf16, B2, H8/2, S4096, D96, dense | 2.1968 | 1.2434 | 1.77x | — | — |
| 6 | bf16, B4, H64/8, S4096, D128, causal | 10.1469 | 6.4610 | 1.57x | — | 1.54x |
| 7 | bf16, B4, H10/10, S2048, D192, dense | 3.0956 | 1.1295 | 2.74x | — | — |
| 8 | bf16, B4, H10/10, S4096, D192, dense | 11.7893 | 4.0791 | 2.89x | 2.90x | — |
| 9 | bf16, B2, H8/2, S2048, D192, dense | 1.7475 | 0.6621 | 2.64x | — | — |
| 10 | bf16, B4, H32/32, S4096, D256, dense | 36.9686 | 14.1224 | 2.62x | 2.62x | 2.58x |
| 11 | bf16, B4, H32/8, S4096, D256, dense | 27.7500 | 15.7285 | 1.76x | — | 1.99x |
| 12 | fp16, B4, H64/8, S2048, D128, causal | 3.3002 | 2.0076 | 1.64x | — | 1.45x |
| 13 | fp16, B4, H64/8, S4096, D128, causal | 10.4861 | 6.4143 | 1.63x | — | 1.52x |
| 14 | fp16, B2, H64/8, S2048, D128, causal | 1.9601 | 1.0562 | 1.86x | — | 1.68x |
| 15 | fp16, B2, H64/8, S4096, D128, causal | 5.9581 | 3.1964 | 1.86x | 1.84x | 1.73x |
| 16 | fp16, B1, H64/8, S4096, D128, causal | 3.4035 | 1.9644 | 1.73x | — | 1.51x |
| 17 | fp32, B4, H64/8, S2048, D128, causal | 36.6937 | 12.7939 | 2.87x | 2.88x | 2.88x |
| 18 | fp32, B4, H64/8, S4096, D128, causal | 133.1444 | 49.3660 | 2.70x | — | 2.62x |
| 19 | fp32, B2, H64/8, S2048, D128, causal | 19.8740 | 6.9018 | 2.88x | — | 2.92x |
| 20 | fp32, B2, H64/8, S4096, D128, causal | 68.4639 | 24.8322 | 2.76x | — | 2.63x |
| 21 | fp32, B1, H64/8, S2048, D128, causal | 11.3262 | 5.1994 | 2.18x | — | 2.27x |
| 22 | fp32, B1, H64/8, S4096, D128, causal | 36.2062 | 12.3706 | 2.93x | — | 2.76x |

First-seed speedups range from **1.18x to 2.93x**.
For the seven repeated cases, speedup changed by at most **1.14%**
between processes/seeds. Parent and PR were measured in separate processes and
builds; their ratio is a ratio of median latencies, not a paired-trial statistic.
The repeat subset covers both the smallest gain and multiple dtype/head-dimension
families; it is not a full second timing sweep of all 23 shapes.

The supplied writeup describes PyTorch 2.13.0+rocm7.1 without exact stock/overlay
revisions. The current PR table also reports explicit tuning and a mixed repeat
protocol. Those ratios are therefore **not an exact like-for-like baseline** for
this campaign. They are retained for comparison, not used as new measurements.
For example, case 11's supplied 1.99x becomes 1.76x here.
The revised article uses the independent pinned-build results.

The PR was also measured with `bwd_ROCM_SEPARATE_DQ=False` as an opt-out control.
Cases 15 and 17 additionally compare explicit indexed and contiguous-block
settings. All variants include the entire backward in their timer; their
latencies and paired comparisons are in [summary.json](summary.json).
The headline baseline remains the actual parent build.

## Numerical checks and timing protocol

The reference is explicit FP64 attention and hand-derived derivatives, chunked
over heads and query rows to bound memory use. Its dense, causal and windowed
GQA cases were separately checked against CPU FP64 autograd, including awkward
chunk boundaries and a sequence longer than the window. Inputs are quantized in
the test dtype before reference conversion. Forward output, dQ, dK and dV must
be finite and pass both an elementwise and a relative-L2 criterion.

| Dtype | Absolute tolerance | Relative tolerance | Relative-L2 limit |
|---|---|---|---|
| bf16 | 0.02 | 0.02 | 0.01 |
| fp16 | 0.003 | 0.01 | 0.002 |
| fp32 | 0.0001 | 0.001 | 0.0001 |

Optimized-gradient relative-L2 error must additionally be no greater than twice
the corresponding stock error plus `1e-6`. Largest observed relative-L2 error
over all successful recorded checks, outputs and gradients:

| Dtype | Maximum relative-L2 error |
|---|---|
| bfloat16 | 0.0024874 |
| float16 | 0.0003093 |
| float32 | 0.0000043 |

**Calibration history:** the initial thresholds rejected one stock bf16 case
and five stock fp16 cases at cancellation-sensitive elements. Thresholds were
fixed using stock results before testing the open PR. Original failures,
harness versions and the rationale remain in
[accuracy-protocol-revision.json](accuracy-protocol-revision.json) and the raw
archive. Failed rows contribute no timing; all were rerun. Both seeds of every
merged-default workload were checked again under the final criteria.

Each case requires three idle GPU samples and no competing process before work
starts. Timings follow an 8-second matrix-multiply clock ramp and 3-second
workload warmup. Each of six trials uses `triton.testing.do_bench` with 200 ms
warmup, 500 ms repetitions and a median result. Variant order alternates, with
the starting order reversed for repeats. Forward, compilation and the FP64
reference are outside the timer. The timed closure clears input gradients and
calls `out.backward(dO, retain_graph=True)`. No CUDA graph timing is used.

ROCm monitoring resolves the selected HIP device to its PCI address, checks
process ownership and samples clocks every 0.5 seconds. Accepted timing windows
had no observed competing PID; loaded clocks ranged from
**1522 to 2202 MHz**. The largest
within-process trial range, divided by its median, was
**1.87%**. Polling does not reserve a
device or detect every short-lived process. No clock or power settings were
changed. These are repeated measurements on one GPU, not confidence intervals
across machines.

Clocks followed normal dynamic frequency scaling; different workloads and
variants can run at different frequencies. The monitor enforces a 1000 MHz
loaded-clock floor and at least two loaded observations, with no maximum clock
spread threshold. The raw records retain the full observed frequency variation.

## Source and compiled-kernel checks

[claims-audit.json](claims-audit.json) records the source review and pinned links.
Corrections to the supplied explanation:

- Stock backward has disjoint dQ and dK/dV programs in one launch and stores dQ
  directly. Splitting launches does not remove dQ atomics. Stock already has
  separate tile pairs; split launches permit independent launch/compiler settings.
- Five matrix multiplications describe ideal algebra with reused intermediates.
  These traversals separately reconstruct probabilities and dP; a blanket 2.5x
  forward FLOP count is not an accurate template count.
- Early dP exposes an independent matrix product. The supplied 17-to-12-unit
  timeline is illustrative, not measured hardware overlap.
- Panel products and accumulator fragments exist, with path-specific widths.
  The fp16 query-first path uses 32-column dV fragments. Source alone does not
  establish a causal reduction in registers or spills.
- Program-ID remapping does not pin a head to a physical XCD. Reversing causal
  row IDs also does not guarantee hardware dispatch order or a shorter tail.
- Query-first GQA shares the **Q block metadata** traversal (`Q_NUM_BLKS/Q_IDX`)
  across query heads, while each head still has its own tensor accesses.
- Automatic selection is limited to supported gfx950 shapes, layouts, masks,
  identity score modification and default-scale inputs, without unsupported
  captured/auxiliary-gradient paths. Explicit tuning settings take precedence.

The generated programs were matched by exact source content to native compiler
artifacts. Requested stages, wavefronts, MFMA dimension, `kpack` and
`waves_per_eu` matched the actual backward compiler metadata on the parent and PR.
For D96 GQA, the two PR phases actually use different pipeline depths (one for
dK/dV, two for dQ). The D192 B4/H10/S4096 profile actually compiles with
`matrix_instr_nonkdim=16` in both phases. No speedup is assigned to an individual
source mechanism without an isolated performance comparison.

All tested optimized backward templates declare zero
`.amdhsa_private_segment_fixed_size` in their assembly. Stock D192/D256 templates
in cases 7–11 declare nonzero sizes (340–556 bytes). This verifies a difference
in compiled private scratch allocation for these configurations; it does not
identify which individual source change caused it. Resource metadata and exact
assembly hashes for every mapped program are retained in the raw archive.

The PR's host selection tests passed **2,153 cases**, and its compiler-option
forwarding tests passed **six cases**. These supplement the independent numerical
checks and real GPU measurements; they were authored with the PR.

## Artifacts and reproduction

- [Raw results and generated backward code](raw-results.tar.gz): all accepted
  timing samples, original calibration failures, correctness metrics, source
  hashes, idle/process/clock samples, and native artifact mappings.
- [Machine-readable aggregate](summary.json), [harness](bench_backward.py),
  [aggregation script](aggregate_backward.py), [native mapping script](map_native.py).
- [Representative compiled artifacts](native/): compressed compiler metadata,
  Triton IR and AMD assembly for cases 0, 4, 5, 8, 10, 15 and 17.
- [File hashes](manifest.json), [exact source build provenance](source-build-provenance.json),
  [build environment](parent-build-environment.json), and [dependency snapshot](requirements-parent.txt).
  `build-original.sh` preserves the actual host-specific build command.

To inspect and re-aggregate the preserved campaign from this directory:

```bash
tar -xzf raw-results.tar.gz
python aggregate_backward.py
```

For new measurements, select an idle GPU explicitly and write new result files.
Use an environment with the official 2.14.1+rocm7.2 wheel for `--campaign merged`:

```bash
TORCHINDUCTOR_CACHE_DIR="$PWD/new-merged-cache" TRITON_CACHE_DIR="$PWD/new-merged-triton" \
  python bench_backward.py --campaign merged --device 0 --seed 0 --gradient ones \
  --output new-results/merged-0.jsonl
TORCHINDUCTOR_CACHE_DIR="$PWD/new-merged-cache" TRITON_CACHE_DIR="$PWD/new-merged-triton" \
  python bench_backward.py --campaign merged --device 0 --seed 1 --gradient random \
  --reverse-order --output new-results/merged-1.jsonl
```

For the open comparison, build the exact parent and PR commits with the pinned
compiler and matching native build flags from the provenance record. This
campaign used full checkouts with recursively initialized submodules, upstream
`tools/amd_build/build_amd.py`, and `pip install -e . -v --no-build-isolation`.
An explicit `-isystem /opt/rocm-7.2.1/include` in `CMAKE_HIP_FLAGS` prevented
older system HIP headers from mixing with ROCm 7.2.1. Keep builds isolated from
other work and never change source while a benchmark process is active.

Run once in each matching environment, with separate compiler caches:

```bash
TORCHINDUCTOR_CACHE_DIR="$PWD/new-parent-cache" TRITON_CACHE_DIR="$PWD/new-parent-triton" \
  python bench_backward.py --campaign parent --device 0 --seed 0 --gradient random \
  --output new-results/parent-0.jsonl
TORCHINDUCTOR_CACHE_DIR="$PWD/new-pr-cache" TRITON_CACHE_DIR="$PWD/new-pr-triton" \
  python bench_backward.py --campaign open --device 0 --seed 0 --gradient random \
  --output new-results/open-0.jsonl
```

Use `--seed 1 --correctness-only` to repeat every numerical check, and
`--seed 1 --cases 0,1,3,8,10,15,17 --reverse-order` for the matching performance
repeat in both environments. Open/parent cases use exactly `1 / math.sqrt(D)`
for the default scale: `D**-0.5` differs by one ULP for D128 and can bypass the
PR's exact default-scale eligibility check. The independent FP64 reference uses
`D**-0.5`; this rounding difference is negligible relative to the stated limits.
