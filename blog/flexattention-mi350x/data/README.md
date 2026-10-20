# FlexAttention measurement data

## Forward summary

[forward-summary.csv](forward-summary.csv) preserves the aggregate throughputs
from the original article at commit
`e2f2049b24fd891b756c78cde4cb5b7298f54065`. Each row is the reported median
throughput across three independent processes, with three timed repeats per
configuration inside each process. It contains 30 distinct configurations:
six workloads, three fp16 configurations, and two bf16 configurations.

The `tflops` values retain the original one-decimal precision. Speedups in the
article are ratios of those values, rounded to two decimals. The charts and
aggregate tables read this CSV directly. The extra fp16 `(num_stages=1,
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

The [official PyTorch 2.14.1 campaign](publication-2.14.1-2026-10-20/README.md)
passed twelve unforced-default checks and three complete grad-enabled processes:
108 measurements and 648 timing trials. Its release charts, statistics, and
compiler-artifact comparison remain separate from the historical data and
figures. The [initial rejected attempt](publication-2.14.1-2026-10-19/README.md)
and prior environment records are preserved.

The PyTorch commit belongs to the author's development fork. Its optional
Gluon candidate is disabled by default; these tables compare the ordinary
Triton template's configurations with autotuning disabled.

## Compiler artifacts

The work-distribution diagram uses preserved outputs from a separate compiler
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
also depend on loop iterations and masking. The additional tile-size examples
in the article's appendix preserve the original draft's reported inspections;
the artifacts bundled here cover the two main diagram cases.

## Regenerating figures

From the article directory, with matplotlib installed:

```bash
python bench/plot_landed.py
python bench/plot_mechanisms.py
python bench/aggregate_landed.py
```

Both figure scripts write SVG files with editable text and PNG exports under
`images/`. The mechanism diagrams are explanatory schematics; their geometry
and instruction counts are scoped in their captions. They introduce no new
performance measurements.
