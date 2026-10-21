# FlexAttention measurement data

## Published forward measurements

The article's forward tables, charts and wavefront diagram use the
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
  processes for the 30 configurations used in article tables. Ratios are
  calculated before rounding the displayed values.
- [Comparison tables](publication-2.14.1-2026-10-20/comparisons.md) and
  [statistics](publication-2.14.1-2026-10-20/statistics.json), including all 36
  configurations and descriptive process ranges.
- [Protocol and environment](publication-2.14.1-2026-10-20/README.md).

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

## Backward and addressing

The [backward validation report](backward-validation-2026-10-21/README.md)
retains the landed default comparison on the official 2.14.1 wheel and the open
specialization PR against its exact parent. It includes two-seed FP64 checks,
full backward timings, compiler evidence and a hashed raw archive.

The [addressing ablation](validation-2026-10-19/README.md#memory-addressing-ablation)
uses a separate pinned environment with template flag handling.

The [captured-gradient atomic validation](atomic-validation-2026-10-21/README.md)
compares stock PyTorch 2.14.1 with the atomic guard restored in memory, retaining
three process pairs, 72 timing trials and native compiler evidence. The article's
appendix identifies these additive-bias cases as synthetic contention tests and
reports the per-head zero-gradient accuracy limitation. Separate nonconstant
modifier controls pass checks for all gradients.

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
