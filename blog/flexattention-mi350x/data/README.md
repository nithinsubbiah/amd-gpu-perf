# FlexAttention Measurements and Reproduction

This guide supports the [MI350X article](../README.md). It retains the full
methods, result tables, resource evidence and numerical qualifications behind
the shorter post. For executable commands and environment setup, use the
[benchmark harness guide](../bench/README.md).

## Published forward measurements

The article's forward chart and wavefront diagram use the
[official PyTorch 2.14.1 campaign](publication-2.14.1-2026-10-20/README.md):
PyTorch 2.14.1+rocm7.2, triton-rocm 3.8.0 and ROCm 7.2.1 on MI350X.
The [wheel lock](publication-2.14.1-2026-10-20/environment/requirements.lock)
identifies the exact installed artifacts.

Three fresh processes cover 36 configurations each, with six timing trials per
configuration: 108 measurements and 648 retained trials. All runs use the
grad-enabled forward context, with `OUTPUT_LOGSUMEXP=True`, inputs that do not
require gradients, pinned kernel options and autotuning disabled. All correctness,
GPU-isolation, monitoring and timing-repeatability checks passed.

- [Raw process 1](publication-2.14.1-2026-10-20/run1.jsonl),
  [process 2](publication-2.14.1-2026-10-20/run2.jsonl),
  [process 3](publication-2.14.1-2026-10-20/run3.jsonl).
- [Summary CSV](publication-2.14.1-2026-10-20/forward-summary.csv): medians across
  processes for the 30 configurations used in the published comparisons. Ratios are
  calculated before rounding the displayed values.
- [Comparison tables](publication-2.14.1-2026-10-20/comparisons.md) and
  [statistics](publication-2.14.1-2026-10-20/statistics.json), including all 36
  configurations and descriptive process ranges.
- [Protocol and environment](publication-2.14.1-2026-10-20/README.md).

## Detailed Measurement Protocol

### Forward environment and scope

The forward configuration comparisons use one MI350X (gfx950), batch size 1, 16 heads at sequence
length 4096 and 32 at 8192, head dimensions 64 and 128, and either no mask or a
causal mask produced by `create_block_mask`. The software is **official PyTorch
2.14.1+rocm7.2**, **triton-rocm 3.8.0**, and **ROCm 7.2.1**, with Python 3.12.3.
The PyTorch source revision is `5c4886908584029761b579af026dcfb627c84070`.

Throughput counts `4 * B * H * D * V` FLOPs, where `V` is the number of live score
entries: `S * S` without a mask or `(S * S + S) / 2` for a causal mask. Compare
configurations within the same row; causal and unmasked throughput count
different amounts of useful work.

### Measurement protocol

We pin previous and updated configurations through `kernel_options` on the same
build, with autotuning disabled. This isolates the settings from changes between
software releases. The forward campaign uses the default grad-enabled context
with inputs that do not require gradients: it emits logsumexp, but times no
backward pass. Disabling grad mode changes the generated kernel and constitutes
a different comparison.

The campaign retains **three fresh processes × 36 configurations × six timing
trials = 648 trials**. Each process uses private compiler caches; the second
reverses configuration order. Before timing each configuration, the harness
checks finite outputs against an eager reference, requires three idle GPU
observations with no competing processes, ramps clocks with eight seconds of
dense matmul, and warms the attention workload for three seconds. Each
`triton.testing.do_bench` trial uses 200 ms warmup and 500 ms of repetitions.
GPU process and clock monitoring continues during timing. Forward output checks
use `(atol, rtol) = (0.003, 0.01)` for fp16 and `(0.02, 0.02)` for bf16.

Forward configuration tables and charts use median throughput across the three processes, each of
which reports its median trial. Ratios use unrounded values. All correctness,
isolation and timing checks passed; the largest within-configuration trial
range was 4.49% of its median. These observed ranges are descriptive, not
confidence intervals. All twelve checks of the unforced forward defaults also
passed and selected the expected configuration.

The landed backward comparison uses the same official wheel with old and new
options set explicitly. The open backward PR uses separately pinned source
builds, identified in its section. The addressing experiment uses the separate
development environment recorded in the addressing section below. Each comparison measures a different
change; their speedups should not be multiplied.

### Backward measurement and numerical checks

Both backward tuning and specialization measurements include gradient clearing,
delta preprocessing, dQ/dK/dV kernels and any additional reductions. Forward
execution, compilation and reference calculations are outside the timer. Each
configuration has six timing trials with the same idle-device checks, clock
ramp, warmup and `do_bench` settings as the forward campaign.

Outputs and all three gradients are checked against independently implemented
FP64 attention and derivatives, also cross-checked against CPU FP64 autograd.
Inputs are quantized to the test dtype before reference conversion. The final
tolerances were calibrated on stock kernels and applied to both variants:

| Data type | Absolute tolerance | Relative tolerance | Relative-L2 limit |
|---|---|---|---|
| bf16 | 0.02 | 0.02 | 0.01 |
| fp16 | 0.003 | 0.01 | 0.002 |
| fp32 | 0.0001 | 0.001 | 0.0001 |

Every output and gradient must be finite and pass both the elementwise and
relative-L2 checks. Optimized-gradient relative-L2 error must also stay within
twice the stock error plus `1e-6`. All reported workloads passed with two input
seeds. The landed-default comparison uses an all-ones incoming gradient for
the first seed and a random gradient for the second; the specialization
comparison uses random incoming gradients for both seeds.

## Full Configuration Comparisons

### Combined fp16 forward defaults

| Sequence length | Head dim | Mask | Previous defaults | Updated defaults | Speedup |
|---|---|---|---|---|---|
| 4096 | 64 | none | 237.9 | 504.2 | 2.12x |
| 4096 | 64 | causal | 102.7 | 312.8 | 3.04x |
| 4096 | 128 | none | 277.6 | 609.3 | 2.20x |
| 4096 | 128 | causal | 154.5 | 401.6 | 2.60x |
| 8192 | 128 | none | 285.5 | 631.9 | 2.21x |
| 8192 | 128 | causal | 193.1 | 463.2 | 2.40x |

*Forward throughput in TFLOP/s. Previous: `(128, 64, 1, 8)`; updated:
`(128, 64, 2, 4)`, ordered as `(BLOCK_M, BLOCK_N, num_stages, num_warps)`.*

### Forward wavefront count

| Sequence length | Head dim | Mask | 8 wavefronts | 4 wavefronts | Speedup |
|---|---|---|---|---|---|
| 4096 | 64 | none | 231.9 | 504.2 | 2.17x |
| 4096 | 64 | causal | 119.1 | 312.8 | 2.63x |
| 4096 | 128 | none | 327.3 | 609.3 | 1.86x |
| 4096 | 128 | causal | 187.1 | 401.6 | 2.15x |
| 8192 | 128 | none | 333.5 | 631.9 | 1.90x |
| 8192 | 128 | causal | 227.9 | 463.2 | 2.03x |

*Forward throughput in TFLOP/s; higher is better. Both columns use `BLOCK_M=128`,
`BLOCK_N=64`, and `num_stages=2`. The upstream PR also measures larger batches and
reports a consistent benefit.*

### BF16 pipeline depth

| Sequence length | Head dim | Mask | 1 stage | 2 stages | Speedup |
|---|---|---|---|---|---|
| 4096 | 64 | none | 442.9 | 519.4 | 1.17x |
| 4096 | 64 | causal | 274.2 | 315.2 | 1.15x |
| 4096 | 128 | none | 540.9 | 637.3 | 1.18x |
| 4096 | 128 | causal | 372.0 | 411.0 | 1.10x |
| 8192 | 128 | none | 576.2 | 660.9 | 1.15x |
| 8192 | 128 | causal | 430.1 | 480.3 | 1.12x |

*Forward throughput in TFLOP/s, with four wavefronts in both columns. The upstream
review swept 16 shapes with [attention-gym](https://github.com/meta-pytorch/attention-gym)
and reported a geometric mean speedup of 1.13x.*

### Landed backward defaults: all 11 workloads

Independent measurements on MI350X show **1.13x to 2.26x faster bf16
backward** across the tested causal, grouped-query and sliding-window workloads.
The full set of 11 workloads is:

| Data type | Sequence length | Head dim | Query/KV heads | Mask | Backward speedup |
|---|---|---|---|---|---|
| bf16 | 2048 | 64 | 16/16 | causal | 1.79x |
| bf16 | 4096 | 64 | 16/16 | causal | 2.09x |
| bf16 | 8192 | 64 | 16/16 | causal | 2.26x |
| bf16 | 2048 | 128 | 16/16 | causal | 1.18x |
| bf16 | 4096 | 128 | 16/16 | causal | 1.17x |
| bf16 | 8192 | 128 | 16/16 | causal | 1.13x |
| bf16 | 4096 | 64 | 16/4 | causal | 1.53x |
| bf16 | 4096 | 128 | 16/4 | causal | 1.41x |
| bf16 | 4096 | 128 | 16/16 | causal window, 1024 | 1.34x |
| fp16 | 4096 | 64 | 16/16 | causal | 2.08x |
| fp16 | 4096 | 128 | 16/16 | causal | 1.17x |

*Batch size is 1 throughout. The 16/4 rows use grouped-query attention;
the window includes keys satisfying `q_idx - 1024 < kv_idx <= q_idx`.*

### Backward tile and pipeline interaction

The tile and pipeline changes work together. For bf16 causal attention at head
dimension 128 and sequence lengths 2048–8192:

| Change from previous defaults | Backward speedup |
|---|---|
| Tile shape alone | 0.86x–0.89x |
| Two pipeline stages alone | 0.93x–0.98x |
| Tile shape and two stages together | 1.13x–1.18x |

*Values below 1x indicate a slowdown. Batch size is 1, with 16 query and KV heads.*

### Addressing ablation

A separate experiment toggled the pointer-range annotation at
`B=1, H=16, S=4096, D=128`, fp16 with grad mode disabled, holding
`(BLOCK_M, BLOCK_N, num_stages, num_warps) = (128, 64, 2, 4)` fixed:

| Mask | Buffer path enabled | Buffer path disabled | Speedup |
|---|---|---|---|
| none | 613.9 | 588.1 | 1.04x |
| causal | 399.6 | 305.5 | 1.31x |

*Forward throughput in TFLOP/s on MI350X. This comparison uses PyTorch
`2.14.0a0+gitb5460ee` with a locally modified Python checkout at `37200109b999`,
including template flag handling, and Triton `3.8.0+gitc8305ccc`.
Each setting has three fresh processes with private
caches and three timing trials per process; values are medians of process medians.*

The experiment uses the same idle-device checks and warmup protocol described
above. All output and isolation checks passed. Assembly confirms that disabling
the annotation replaces buffer loads and stores with global operations.
This comparison measures a different change and contributes no additional factor
to the headline result.

### Proposed backward specialization

We built the PR and its exact parent with the same pinned compiler and measured
automatic selection. Across 23 supported configurations, complete backward
speedups range from **1.18x to 2.93x**. Representative results:

| Data type | Batch | Query/KV heads | Sequence length | Head dim | Mask | Backward speedup |
|---|---|---|---|---|---|---|
| bf16 | 4 | 21/21 | 4096 | 96 | dense | 2.86x |
| fp16 | 2 | 64/8 | 4096 | 128 | causal | 1.86x |
| fp32 | 4 | 64/8 | 2048 | 128 | causal | 2.87x |

*MI350X, ROCm 7.2.1, PyTorch 2.16.0a0 source builds: parent `656560bcbe2f`
versus PR `d64287342982`; Triton `fc8d31a5439e`. Ratios compare median latencies
from the first measurement pass. All 23 configurations passed independent FP64
checks with two seeds; seven received a second performance measurement on both
builds, with speedup ratios differing by at most 1.14% between the two passes.
Each build uses six timing trials per configuration, with autotuning disabled
and `BACKEND="TRITON"`.*

Automatic selection is guarded by the supported shapes, layouts, masks and
other input conditions; explicit tuning settings take precedence. These results
describe this PR revision, with its own software baseline, and are not part of
the released forward defaults measured earlier.

The [backward measurement report](backward-validation-2026-10-21/README.md) contains
all 23 specialization configurations and their absolute latencies. Both comparisons
check outputs and all three gradients with two seeds using the criteria above.


## Compiler artifacts

The current wavefront diagram uses code generated during that same release
campaign: fp16, `B=1, H=16, S=4096, D=128`, unmasked, with `BLOCK_M=128`,
`BLOCK_N=64` and two pipeline stages. Representative artifacts from process 1:

| Wavefronts | Triton GPU IR | AMD assembly |
|---|---|---|
| 4 | [IR](publication-2.14.1-2026-10-20/run1/codegen/BXHM6QD5JDNKH3R5P5JFTP3A75EG5KKR5A2YLV43ZTIWFKHPWT2A-triton_tem_fused__to_copy_flex_attention_ones_slice_sum_transpose_zeros_2.ttgir.gz) | [Assembly](publication-2.14.1-2026-10-20/run1/codegen/BXHM6QD5JDNKH3R5P5JFTP3A75EG5KKR5A2YLV43ZTIWFKHPWT2A-triton_tem_fused__to_copy_flex_attention_ones_slice_sum_transpose_zeros_2.amdgcn.gz) |
| 8 | [IR](publication-2.14.1-2026-10-20/run1/codegen/YDIJXRA5LN5XHH5NZUCAY74FIRG535S2UAR3YJGZGK7B5RBSODJA-triton_tem_fused__to_copy_flex_attention_ones_slice_sum_transpose_zeros_2.ttgir.gz) | [Assembly](publication-2.14.1-2026-10-20/run1/codegen/YDIJXRA5LN5XHH5NZUCAY74FIRG535S2UAR3YJGZGK7B5RBSODJA-triton_tem_fused__to_copy_flex_attention_ones_slice_sum_transpose_zeros_2.amdgcn.gz) |

| Wavefronts | MFMA sites | Sites × wavefronts | `ds_write` sites |
|---|---|---|---|
| 4 | 64 | 256 | 9 |
| 8 | 48 | 384 | 72 |

The [compiler audit](publication-2.14.1-2026-10-20/compiler-audit.json) links
artifacts for all three processes, verifies hashes, and confirms these counts
agree across them. The gzip files preserve the generated text. Decompress with
`gzip -dc` to inspect instruction lines beginning with
`v_mfma_f32_32x32x16_f16` or `ds_write`. Static sites do not determine dynamic
instruction counts; loop execution and masking also matter.

### Resources and scheduling clues

The release [compiler audit](publication-2.14.1-2026-10-20/compiler-audit.json) also
retains register counts and shared-memory allocation. For fp16, B=1, H=16,
S=4096, D64, unmasked, eight wavefronts:

| Pipeline stages | Total VGPR allocation | Shared memory, bytes | Compiler-reported waves per SIMD |
|---|---|---|---|
| 1 | 128 | 16,384 | 4 |
| 2 | 140 | 50,112 | 3 |

These allocations agree across all three release processes. The VGPR value is
LLVM's `TotalNumVgprs`; shared memory comes from Triton metadata because this
kernel uses dynamic LDS and the assembly's fixed group-segment size is zero.
Occupancy is the compiler's resource estimate, not a measured active-wave counter.
The configuration's throughput changes from 237.9 to 231.9 TFLOP/s. Multiple
scheduling changes accompany the allocation changes, so this supports a resource
tradeoff without establishing that occupancy alone causes the slowdown.

For the separate addressing ablation, retained assembly from all three on/off
processes has these allocations, with no scratch spills:

| Mask | Pointer range | Total VGPR allocation | Compiler-reported waves per SIMD |
|---|---|---|---|
| none | on | 224 | 2 |
| none | off | 214 | 2 |
| causal | on | 252 | 2 |
| causal | off | 260 | 1 |

The corresponding artifacts and text hashes are recorded in each
`validation-2026-10-19/address-{on,off}{1,2,3}/ablation.json`.
The [addressing report](validation-2026-10-19/README.md#memory-addressing-ablation)
provides the original timings and environment. Buffer lowering changes address
arithmetic, predication and resource allocation together. In the inspected causal
assembly, static `v_cndmask` sites actually increase (79 to 104), while
`s_and_saveexec` sites fall (9 to 0). A claim that the speedup simply removes
conditional selects is therefore unsupported. Static counts alone do not assign
the runtime gain to one instruction class or predict achieved occupancy.

## Backward and addressing

The [backward validation report](backward-validation-2026-10-21/README.md)
retains the landed default comparison on the official 2.14.1 wheel and the open
specialization PR against its exact parent. It includes two-seed FP64 checks,
full backward timings, compiler evidence and a hashed raw archive.

The [addressing ablation](validation-2026-10-19/README.md#memory-addressing-ablation)
uses a separate pinned environment with template flag handling.

The [captured-gradient atomic validation](atomic-validation-2026-10-21/README.md)
compares stock PyTorch 2.14.1 with the atomic guard restored in memory, retaining
three process pairs, 72 timing trials and native compiler evidence. The stress-test section below identifies these additive-bias cases as synthetic
contention tests and reports the per-head zero-gradient accuracy limitation. Separate nonconstant
modifier controls pass checks for all gradients.

## Architecture Defaults

MI300X (gfx942) and MI350X (gfx950) have different LDS capacities and matrix
throughput, so they can favor different tile sizes and wavefront counts. These
settings balance data reuse and latency hiding against the resources needed by
each workgroup. A separate default table for each architecture ([PyTorch #181283](https://github.com/pytorch/pytorch/pull/181283)) allows either
to be tuned without changing the other's defaults. The half-precision forward
entries are shown as `(BLOCK_M, BLOCK_N, num_stages, num_warps)`:

| Data type | Head dim | MI300X (gfx942) | MI350X (gfx950) |
|---|---|---|---|
| bf16 | 64, 128 | 64, 64, 2, 8 | 128, 64, 2, 4 |
| bf16 | 256 | 32, 64, 2, 8 | 32, 64, 2, 4 |
| fp16 | 64, 128 | 64, 32, 1, 8 | 128, 64, 2, 4 |
| fp16 | 256 | 32, 32, 1, 8 | 32, 64, 2, 4 |

These are the entries introduced by the split; our measurements cover MI350X
head dimensions 64 and 128. The split itself leaves the gfx950 values unchanged,
so it contributes no additional MI350X speedup to the figures here. The smaller
MI300X tiles and head-dimension-256 entries also illustrate why the wavefront
count needs to be evaluated with the rest of the configuration.

The architecture-table change also retains these fp32 entries on both gfx942
and gfx950: `(128, 32, 1, 4)` at head dimensions 64 and 128, and
`(64, 16, 1, 4)` at head dimension 256. The forward measurements here cover half
precision; they do not establish whether fp32 benefits from additional stages.

## Captured-Gradient Atomics: A Contention Stress Test

For dense fp16 attention at `B=1, H=8, S=2048, D=64`, an additive bias captured
by `score_mod` makes backward accumulate into either eight per-head destinations
or 16,384 per-query destinations. We compared the stock PyTorch 2.14.1+rocm7.2
wheel with the atomic guard restored in memory, using triton-rocm 3.8.0 and
ROCm 7.2.1. Three fresh process pairs retained 72 timing trials:

| Captured bias | Stock backward | Guard restored | Speedup |
|---|---|---|---|
| Per head: `bias[h]` | 425.052 ms | 6.661 ms | 63.81x |
| Per query: `bias[h, q_idx]` | 261.967 µs | 224.127 µs | 1.17x |

*Complete backward latency; lower is better. Each value is the median of three
process medians. Compilation and forward execution are outside the timer.*

The generated template bodies match within every pair. The backward assembly
changes from 16 `buffer_atomic_add_f32` sites to 16 `global_atomic_add_f32` sites.
The large difference between the two workloads illustrates how strongly
contention affects this path.

These biases add a constant across each softmax row, so their true gradients
are zero. The per-head captured gradient exceeds our preset absolute tolerance
of 0.003 in both versions, with maximum errors of 0.0062–0.0125 before and
0.0062–0.0065 after. Outputs and Q/K/V gradients pass the FP64 checks, as does
the per-query bias gradient. A separate nonconstant modifier,
`score + bias * (kv_idx / S)`, passes checks for the output and all four gradients
with both bias shapes before and after the guard; those controls were not timed.

The 63.81x result characterizes this synthetic contention stress test. Its
zero-gradient numerical issue limits what it establishes about training.

## Regenerating figures

From the article directory, with matplotlib installed:

```bash
python bench/plot_landed.py
python bench/plot_mechanisms.py
python bench/aggregate_landed.py
# Validate and aggregate the complete raw release campaign:
python bench/aggregate_landed.py 'data/publication-2.14.1-2026-10-20/run[123].jsonl'
```

The defaults read the official release summary and compiler audit. The figure
scripts write editable SVG and PNG exports under `images/`, without using a GPU.
The mechanism diagrams are explanatory schematics, scoped by their captions.
They introduce no new performance measurements. For a new timing campaign, use
[the benchmark instructions](../bench/README.md).

## Earlier records

The [pre-integration article and figures](history/pre-release-integration-2026-10-21/article-and-figures.tar.gz)
and their [manifest](history/pre-release-integration-2026-10-21/manifest.json)
preserve the earlier development-build presentation, including its plot scripts.
Historical measurements, compiler artifacts and rejected attempts remain intact.
The [initial release attempt](publication-2.14.1-2026-10-19/README.md) stopped
before a full process passed and contributes no timing results to the article.

## Historical development measurements

[forward-summary.csv](forward-summary.csv) preserves the aggregate throughputs
from the original article at commit
`e2f2049b24fd891b756c78cde4cb5b7298f54065`. Each row is the reported median
throughput across three independent processes, with three timed repeats per
configuration inside each process. It contains 30 distinct configurations:
six workloads, three fp16 configurations, and two bf16 configurations.

The `tflops` values retain the original one-decimal precision. Speedups in the
historical article are ratios of those values, rounded to two decimals. The
current article uses the release campaign linked above. The extra fp16 `(num_stages=1,
num_warps=4)` factorial configuration is measured by the reproduction harness
but its full historical results are not present in this summary.

The original `/tmp/hb_r1.log`, `/tmp/hb_r2.log`, and `/tmp/hb_r3.log` files are
unavailable. The CSV is an aggregate snapshot, not a replacement for those raw
runs. It cannot establish a confidence interval or recover paired process-level
speedup distributions. The revision therefore removes the original `+/- 0.02x`
uncertainty claim. Fresh JSONL campaigns are stored and analyzed separately.

The [environment record](environment.json) distinguishes the historical build
information from compiler metadata recovered later. The historical campaign
recorded Triton 3.8.0 without a full compiler SHA. The available local package
identifies `3.8.0+gitc8305ccc`; that identifies a source setup for new runs but
does not prove the exact compiler revision used by the historical campaign.

A [fresh validation campaign](validation-2026-10-19/README.md) retains complete
process runs, generated kernels, source identities, monitoring evidence, and
separate comparison charts. It does not replace this historical summary.
Its initial full sweep uses grad-disabled execution. A
[controlled recheck](validation-2026-10-19/baseline-recheck/README.md) resolves
the resulting baseline difference by restoring the original grad-enabled
context and recording the generated logsumexp output flag.

The PyTorch commit belongs to the author's development fork. Its optional
Gluon candidate is disabled by default; these tables compare the ordinary
Triton template's configurations with autotuning disabled.

### Historical compiler artifacts

The earlier work-distribution diagram used outputs from a separate compiler
inspection of fp16 forward with `BLOCK_M=128`, `BLOCK_N=64`, head dimension 128,
`num_stages=2`, and target gfx950. The preserved kernel is unmasked with
`B=1, H=16, S=4096`. These files support the mechanism and instruction counts;
they are not timing samples from the three-process throughput campaign.

| Wavefronts | Triton GPU IR | AMD assembly |
|---|---|---|
| 4 | [warps4.ttgir.gz](codegen/warps4.ttgir.gz) | [warps4.amdgcn.gz](codegen/warps4.amdgcn.gz) |
| 8 | [warps8.ttgir.gz](codegen/warps8.ttgir.gz) | [warps8.amdgcn.gz](codegen/warps8.amdgcn.gz) |

The gzip files preserve the original text. [counts.json](codegen/counts.json)
contains SHA-256 hashes of the decompressed files and these counts:

| Wavefronts | MFMA sites | MFMA sites × wavefronts | `ds_write` sites |
|---|---|---|---|
| 4 | 128 | 512 | 9 |
| 8 | 96 | 768 | 138 |

To inspect the assembly, decompress with `gzip -dc` and count instruction lines
beginning with `v_mfma_f32_32x32x16_f16` or `ds_write`. Dynamic instruction counts
also depend on loop iterations and masking. The archived draft also reports other tile-size inspections;
the historical artifacts bundled here cover only the two cases above.
