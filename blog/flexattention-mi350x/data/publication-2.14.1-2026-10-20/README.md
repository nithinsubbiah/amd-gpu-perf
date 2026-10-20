# PyTorch 2.14.1 publication validation

This replacement GPU campaign started on 2026-10-20 using the verified official
release environment. All three full processes use **six timing trials per
configuration**, 200 ms warmup and 500 ms repetitions per trial, and fresh private
compiler caches. The second process reverses configuration order. The campaign
completed successfully: **108 measurements and 648 retained timing trials**.
All correctness, isolation, loaded-clock, timing-repeatability, source, and
compiler-artifact checks passed. The twelve unforced-default checks and
four-configuration smoke check also passed.

[Completion marker](complete.json), [final validation record](validation.json),
[statistics](statistics.json), [comparison tables](comparisons.md), and
[compiler audit](compiler-audit.json) retain the evidence. The selected physical
GPU was again idle with no KFD processes in three consecutive
[postflight observations](environment/postflight.json).

The [previous attempt](../publication-2.14.1-2026-10-19/README.md) passed all twelve
unforced-default checks and the four-configuration smoke check, then stopped
because its short timing window provided only one of two required loaded-clock
observations. [Failure evidence](../publication-2.14.1-2026-10-19/failure-analysis.json)
shows no competing process and stable timing trials. Six trials extend monitoring
coverage while retaining every trial and all existing acceptance thresholds.
No partial results from the rejected attempt are included here.

Original campaign files and a manifest of all preexisting article files are
preserved in the [resume archive](../publication-2.14.1-2026-10-19/history/20261020T001809Z/).
Historical results and article figures retain their existing attribution.

## Frozen environment

| Component | Identity |
|---|---|
| PyTorch | Official `2.14.1+rocm7.2` wheel |
| PyTorch source | `5c4886908584029761b579af026dcfb627c84070` |
| Triton distribution | `triton-rocm==3.8.0`, as required by the PyTorch wheel |
| HIP build version | `7.2.53211` |
| Host ROCm installation | 7.2.1 |
| Python | 3.12.3 |
| Platform | Ubuntu 24.04, x86_64 |
| Selected GPU ordinal | 6; its physical mapping is verified before timing |
| Local isolated environment | `/home/nmeganat/flex/.venv-blog-2.14.1-rocm7.2` |

The [installation report](environment/install-report.json) retains the official
PyTorch/Triton wheel URLs and SHA256 hashes. The [tooling report](environment/tooling-install-report.json)
covers NumPy, matplotlib, and their dependencies. [Exact versions](environment/requirements.txt)
and a [wheel URL lock](environment/requirements.lock) are retained for reconstruction.
The wheel URL lock targets CPython 3.12 on Linux x86_64.

The [software snapshot](environment/software.json) records the build settings.
[Source checksums](environment/packaged-source-sha256.json) verify 2,597 installed
Python/template files against the distributions' wheel RECORD hashes. The
source audit does not establish the Triton wheel's full source commit; its
download URL and artifact hash identify the actual compiler used.

## Release audit

The [release audit](environment/release-audit.json) reads the installed source.
It confirms gfx950 defaults `(BLOCK_M, BLOCK_N, num_stages, num_warps)` of
`(128, 64, 2, 4)` for fp16 and bf16 at head dimensions 64 and 128. Both gfx942
and gfx950 tables match the architecture comparison in the article.
The frontend still uses grad mode to control `OUTPUT_LOGSUMEXP` on HIP.

The later [template/atomics guard](https://github.com/pytorch/pytorch/pull/194136)
is absent: template metadata calls `config_of(signature)` without the shared
guard or pointer override. The official 2.14.1 environment is suitable for
the forward configuration comparison. The existing addressing on/off script
requires that later flag handling and needs a separate pinned upstream environment.
Its backward results must retain attribution to the PR's measurements.

## Campaign command and retained artifacts

The execution context must expose `/dev/kfd` and `/dev/dri` and permit GPU
process monitoring. This completed campaign was launched with:

```bash
/home/nmeganat/flex/.venv-blog-2.14.1-rocm7.2/bin/python \
  /home/nmeganat/flex/blog-editorial/blog/flexattention-mi350x/data/publication-2.14.1-2026-10-20/run_publication.py \
  --device 6
```

The launcher checks access before creating GPU-run directories. It then verifies
installed sources, checks unforced release defaults and reference correctness
for all twelve workloads, runs a four-configuration smoke check, and runs three
complete 36-configuration benchmark processes. The second process reverses
configuration order. The grad-enabled, forward-only protocol matches the original
script; input tensors do not require gradients. Each process has fresh private
Inductor and Triton caches.

Every timing point requires three consecutive idle observations with no competing
processes, a clock ramp, target-kernel warmup, reference correctness, repeated
timings, and monitoring. The existing harness aborts on observed competition or
failed validation. Polling does not reserve a GPU; an exclusive allocation is
the preferred isolation mechanism.

The launcher retains JSONL trials, generated Python, compiler IR, assembly,
metadata, monitoring samples, aggregate comparisons, statistics, and charts.
It creates `complete.json` only after all three full processes and their audit
pass. It does not update article figures automatically.

Run artifacts must have new names. If a GPU attempt fails, preserve its files
and investigate the cause before starting a replacement campaign. The previous
failed attempt is retained in its original directory. This completed campaign
already has outputs; do not rerun the command over these files. A reproduction
needs a new sibling campaign directory with the scripts, environment records,
and source archive, preserving the same six-trial protocol.

## Accepted release results

Each speedup below is the ratio of median throughputs across three complete
processes. The fp16 combined comparison changes `(stages, wavefronts)` from
`(1, 8)` to `(2, 4)`; the wavefront-only comparison holds two stages fixed;
the bf16 comparison changes one stage to two while keeping four wavefronts.
All use grad-enabled forward execution with inputs that do not require gradients
and generated `OUTPUT_LOGSUMEXP=True`.

| Sequence | Head dim | Mask | fp16 combined | fp16 wavefronts | bf16 pipelining |
|---|---|---|---|---|---|
| 4096 | 64 | none | 2.12x | 2.17x | 1.17x |
| 4096 | 64 | causal | 3.04x | 2.63x | 1.15x |
| 4096 | 128 | none | 2.20x | 1.86x | 1.18x |
| 4096 | 128 | causal | 2.60x | 2.15x | 1.10x |
| 8192 | 128 | none | 2.21x | 1.90x | 1.15x |
| 8192 | 128 | causal | 2.40x | 2.03x | 1.12x |

All 30 configurations shared with the historical summary are within **-1.59% to
+1.08%** of their historical throughputs. The six additional fp16 `(1, 4)`
configurations remain in the complete JSONL records; the derived
[summary CSV](forward-summary.csv) preserves the historical 30-row schema.
The historical CSV itself is unchanged.

The maximum within-point trial range is **4.49%** of the median, below the 5%
limit. Across-process throughput ranges have a median of **0.27%** and a maximum
of **3.98%**, normalized by each configuration's median. These are descriptive
observed ranges, not confidence intervals. Full paired process ratios are in
`statistics.json` and `comparisons.md`.

Maximum absolute reference differences are **0.001953125** for fp16 and
**0.015625** for bf16, within the recorded tolerances. Every measured point has
at least four loaded-clock observations meeting the 90% utilization requirement;
two are required. Observed loaded SCLK spans **1322–2200 MHz**, above the 1000 MHz
floor. The protocol records DVFS variation and imposes the trial-repeatability
limit rather than a clock-range limit.

The separate release charts are available as
[fp16 PNG](figures/01-fp16-speedup.png) / [SVG](figures/01-fp16-speedup.svg) and
[bf16 PNG](figures/02-bf16-pipelining.png) / [SVG](figures/02-bf16-pipelining.svg).
Both chart exports were visually reviewed. The article's existing tables and
images continue to use the historical development-build measurements.

## Compiler evidence

`audit_compiler.py` matches all 108 measurements to their generated Python,
Triton GPU IR, assembly, and metadata; verifies retained artifact hashes and
requested stage/wavefront settings; and checks the representative mechanism
counts agree across all three processes. It also verifies the historical IR and
assembly checksums and recounts the historical MFMA and LDS-write sites.

For fp16, `B=1, H=16, S=4096, D=128`, unmasked, with `BLOCK_M=128`,
`BLOCK_N=64`, and two stages:

| Wavefronts | Release MFMA sites | Release sites × wavefronts | Release `ds_write` sites | Historical sites × wavefronts | Historical `ds_write` sites |
|---|---|---|---|---|---|
| 4 | 64 | 256 | 9 | 512 | 9 |
| 8 | 48 | 384 | 72 | 768 | 138 |

Both release kernels use `v_mfma_f32_32x32x16_f16`. Four wavefronts retain the
matching `[4, 1]` layouts; eight retain `[8, 1]` followed by `[4, 2]`. Thus the
layout-based explanation still applies, but the literal static counts differ.
The retained historical IR contains two `scf.for` sites and the release IR one;
static sites do not establish dynamic instruction counts or an additional
runtime speedup. The [compiler audit](compiler-audit.json) links each artifact
and records buffer/global access counts, registers, and shared memory.

The historical diagram and its original evidence remain unchanged. A version
attributed to this release must use the release counts above.

## Remaining publication work

1. Validate the addressing section on a separately pinned upstream build that
   contains the template/atomics guard. Keep the backward results attributed to
   the PR's measurements.
2. Integrate accepted release tables, charts, and revised mechanism counts into
   the article with updated software and six-trial protocol attribution, while
   retaining the historical results and figures.

CPU status: all twelve harness validation tests passed. The launcher log is
`publication.log`. The [preservation audit](preservation-audit.json) checks the
preexisting file manifest and the archive of replaced campaign status records.
