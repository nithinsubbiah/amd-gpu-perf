# Tuning PyTorch FlexAttention for AMD Instinct MI350X

PyTorch's [FlexAttention](https://pytorch.org/blog/flexattention/) lets you express
attention variants — causal masks, sliding windows, ALiBi, soft-capping — as small
Python functions that `torch.compile` fuses into a Triton kernel. The kernel's
configuration determines how well it uses the GPU. On AMD Instinct MI350X, two
changes to the ROCm defaults improved fp16 forward throughput by **2.1x to 3.0x**
on the shapes we measured: increasing the software-pipelining depth from one to
two stages and reducing the wavefront count from eight to four.

![Previous and updated configurations on MI350X: fp16 FlexAttention forward throughput improves by 2.12x to 3.03x across six measured cases.](images/01-fp16-speedup.png)

*Combined effect of `num_stages=1 → 2` and `num_warps=8 → 4`, with both configurations
compiled on the same software build. Batch size is 1; the tests cover head dimensions
64 and 128, sequence lengths 4096 and 8192, and unmasked and causal attention.*

The bf16 defaults already used four wavefronts, so they benefit from the
pipelining change alone: **1.11x to 1.18x**. These configuration changes are included
in the [PyTorch 2.14.1 ROCm defaults](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/torch/_inductor/heuristics/template/triton.py).
The investigation also led to
architecture-specific defaults and improvements to memory addressing in Inductor
and Triton. Together, they show how configuration choices and compiler lowering
shape the performance of the same attention algorithm.

## How Inductor configures FlexAttention

Inductor fills in a Triton template with the query tile height `BLOCK_M`, the
key/value tile width `BLOCK_N`, the software-pipelining depth `num_stages`, and the
number of wavefronts cooperating on each tile, `num_warps`. In the default compile
mode, it chooses a configuration from a table. With
`torch.compile(mode="max-autotune")`, it benchmarks candidate configurations.

A wavefront is a group of GPU threads that execute together; on gfx950 it contains
64 threads. Wavefronts cooperate on the attention kernel's two matrix products:
`QKᵀ`, followed by softmax, then `PV`. Their assignment to those products determines
both how much work they perform and whether intermediate results need to move
between them.

## Making Every Wavefront Count

The fp16 defaults used `num_warps=8` at head dimensions 64 and 128, while bf16 used
four for the same shapes. Changing the fp16 default to four[^1] improved forward
throughput by **1.86x to 2.61x**, holding `num_stages=2` fixed:

| Sequence length | Head dim | Mask | 8 wavefronts | 4 wavefronts | Speedup |
|---|---|---|---|---|---|
| 4096 | 64 | none | 233.9 | 508.0 | 2.17x |
| 4096 | 64 | causal | 119.1 | 311.4 | 2.61x |
| 4096 | 128 | none | 331.2 | 616.2 | 1.86x |
| 4096 | 128 | causal | 187.0 | 402.7 | 2.15x |
| 8192 | 128 | none | 337.7 | 637.2 | 1.89x |
| 8192 | 128 | causal | 225.5 | 467.4 | 2.07x |

*Forward throughput in TFLOP/s; higher is better. Both columns use `BLOCK_M=128`,
`BLOCK_N=64`, and `num_stages=2`. The upstream PR also measures larger batches and
reports a consistent benefit.*

The generated code explains why fewer wavefronts help. In the inspected fp16,
head-dimension-128 kernel, both configurations use the same matrix instruction,
`v_mfma_f32_32x32x16_f16`, which produces a 32×32 output tile per wavefront. Four
such tiles cover the 128 query rows. Asking for eight wavefronts changes how
Triton distributes that work across the two matrix products.

![Eight wavefronts replicate QK work and require an LDS layout conversion; four wavefronts use matching layouts for QK and PV in the inspected configuration.](images/03-wavefront-distribution.png)

*Work assignments for `BLOCK_M=128`, `BLOCK_N=64`, head dimension 128, fp16 on
gfx950. The two QK assignments in each band on the left cover the same keys;
the two PV wavefronts divide alternating 32-column tiles. The counters are static
compiler evidence, with MFMA instruction sites weighted by wavefront count.*

Triton's first matrix product assigns all wavefronts along the query-row axis,
keeping each softmax row within a wavefront. With eight wavefronts, pairs of them
compute the same QK bands. For the second product, Triton assigns four wavefronts
along the rows and two along the output columns. Moving between those layouts
requires a conversion through LDS, the GPU's shared memory. With four
wavefronts, both products use a `[4, 1]` layout.

The saved assembly contains 768 wave-weighted MFMA instruction sites with eight
wavefronts, versus 512 with four. Static `ds_write` sites fall from 138 to 9.
These counts explain redundant computation and the added layout conversion;
the throughput table measures their runtime impact. The [compiler appendix](#compiler-appendix)
includes the layout details and inspectable artifacts.

The change to Inductor is one configuration value. The rendered Triton kernel
body stays the same, while the compiler derives different layouts and machine
instructions from that value. For other tile sizes and instruction geometries,
inspect both matrix products and measure the candidates. The demonstrated benefit
of four wavefronts is specific to these configurations.

## Overlapping memory loads and computation

Triton's software pipeliner can schedule loads for upcoming key/value blocks
alongside computation on the current block. The ROCm defaults used
`num_stages=1`, limiting that software-pipelining opportunity. Increasing the
default to two stages[^2]
provides a consistent gain on the measured bf16 cases:

![Changing from one to two software-pipeline stages improves bf16 FlexAttention forward throughput by 1.11x to 1.18x on MI350X.](images/02-bf16-pipelining.png)

| Sequence length | Head dim | Mask | 1 stage | 2 stages | Speedup |
|---|---|---|---|---|---|
| 4096 | 64 | none | 445.4 | 523.6 | 1.18x |
| 4096 | 64 | causal | 274.6 | 313.9 | 1.14x |
| 4096 | 128 | none | 548.2 | 647.6 | 1.18x |
| 4096 | 128 | causal | 373.0 | 412.7 | 1.11x |
| 8192 | 128 | none | 579.1 | 667.1 | 1.15x |
| 8192 | 128 | causal | 431.0 | 484.1 | 1.12x |

*Forward throughput in TFLOP/s, with four wavefronts in both columns. The upstream
review swept 16 shapes with [attention-gym](https://github.com/meta-pytorch/attention-gym)
and reported a geometric mean speedup of 1.13x.*

The two knobs interact. In the grad-enabled forward setup measured here,
pipelining is slightly negative with eight wavefronts on the unmasked fp16
head-dimension-64 case: 239.1 to 233.9 TFLOP/s, or 0.98x. With four wavefronts,
pipelining benefits that shape. The reproduction harness measures all four
combinations so each effect can be checked in the same execution context.
When tuning a new target, benchmark combinations of tile size, wavefront count,
and pipeline depth.

## Streamlining Memory Addressing

Memory access has an instruction cost beyond transferring the tensor values.
A kernel must also calculate their addresses. When the compiler can prove that
tensor offsets fit in 32 bits, it can represent those accesses as a shared
64-bit base plus smaller offsets and use eligible AMD buffer loads and stores.

![Known storage bounds allow Inductor and Triton to express accesses using a shared 64-bit base and bounded 32-bit lane offsets, enabling eligible buffer loads and stores.](images/04-memory-addressing.png)

*The accessed addresses remain 64-bit. The optimization narrows the offsets
where the compiler can prove that doing so preserves the addresses.*

Inductor supplies that information through a pointer-range annotation[^3]: on
HIP, tensor arguments whose storage is provably
smaller than 2 GiB receive `tt.pointer_range=32`. Triton's pointer canonicalization and
buffer-operation lowering can then use the bounded-offset representation.
The compiler suppresses the annotation for user-defined Triton kernels[^4],
where Inductor cannot establish
the necessary pointer bounds.

Triton's 64-bit offset support[^5]
also allows eligible offset expressions to reach buffer operations when it can
prove they are safe to narrow. The useful condition is the proof of the offset
range; the expression's initial integer width alone does not decide eligibility.

### Applying the controls to template kernels

The annotation needs consistent controls across Inductor's code-generation
paths. FlexAttention uses a template path that bypassed both
`TORCHINDUCTOR_EMIT_POINTER_RANGE_32=0` and the existing suppression for kernels
containing atomics. The template guard[^6]
moves the decision into a shared helper and records atomics emitted by
FlexAttention's captured-buffer gradient lowering.
This later template guard is absent from PyTorch 2.14.1; the backward results
below are measurements reported in the PR.

> **A workload where the guard matters.** With a learnable per-head bias captured
> by `score_mod`, backward accumulates many contributions into just eight
> addresses. Suppressing the annotation avoids buffer atomics for this highly
> contended accumulation. Restoring the guard changed backward latency from **425.2 ms to
> 6.8 ms** at `B=1, H=8, S=2048, D=64`, fp16 on MI350X. A per-query bias spreads
> the accumulation over 16,384 addresses and improves by **1.09x** instead.
> The penalty depends strongly on contention. These backward measurements are
> separate from the forward results above.

### Supporting compiler changes

The buffer path also needed an SSA dominance fix.[^7]
A separate packed fp32-to-bf16 conversion[^8]
uses `v_perm_b32` for round-toward-zero packing; its reported reduction in register
pressure and spills was measured on gfx1250. These changes support efficient
AMD kernels beyond the MI350X comparisons here.

Both configurations in every forward comparison use the same compiler with
buffer operations enabled, so these compiler changes are already shared by
both columns. A separate [fresh addressing ablation](data/validation-2026-10-19/README.md#memory-addressing-ablation)
at `B=1, H=16, S=4096, D=128`, fp16 with grad mode disabled, measured
**399.6 versus 305.5 TFLOP/s** for
causal attention with and without the buffer path: **1.31x**. The unmasked case
improved by **1.04x**. Saved assembly verifies that the accesses switch between
buffer and global operations. This comparison measures a different change and
contributes no additional factor to the headline result.

## Measurements and reproduction

The combined fp16 results shown in the opening figure are:

| Sequence length | Head dim | Mask | Previous defaults | Updated defaults | Speedup |
|---|---|---|---|---|---|
| 4096 | 64 | none | 239.1 | 508.0 | 2.12x |
| 4096 | 64 | causal | 102.7 | 311.4 | 3.03x |
| 4096 | 128 | none | 278.7 | 616.2 | 2.21x |
| 4096 | 128 | causal | 153.5 | 402.7 | 2.62x |
| 8192 | 128 | none | 289.1 | 637.2 | 2.20x |
| 8192 | 128 | causal | 193.5 | 467.4 | 2.42x |

*Forward throughput in TFLOP/s. Previous: `(128, 64, 1, 8)`; updated:
`(128, 64, 2, 4)`, ordered as `(BLOCK_M, BLOCK_N, num_stages, num_warps)`.*

All forward tables use one MI350X (gfx950), batch size 1, 16 heads at sequence
length 4096 and 32 at 8192, head dimensions 64 and 128, and either no mask or a
causal mask produced by `create_block_mask`. The software build was ROCm 7.2.1,
PyTorch `2.14.0a0+gitb5460ee`, and Triton 3.8.0. The
[environment record](data/environment.json) identifies the PyTorch source
revision and the available compiler metadata.

Throughput counts `4 * B * H * D * V` FLOPs, where `V` is the number of live score
entries: `S * S` without a mask or `(S * S + S) / 2` for a causal mask. This
normalization makes causal and unmasked throughput different measures of useful
work; compare configurations within the same workload row.

### Measurement protocol

The previous configurations are pinned through `kernel_options` on the same
build as the updated ones. This isolates the configuration changes from other
changes between software releases. The headline result measures the default
Triton path with autotuning disabled.

The original script uses the default grad-enabled context, with input tensors
that do not require gradients; no backward pass is timed. In this context,
FlexAttention still emits logsumexp. Disabling grad mode changes that generated
kernel, so the harness records both grad mode and `OUTPUT_LOGSUMEXP`. Its default
matches the original script; use `--grad-mode disabled` for a separate
grad-disabled forward comparison.

The reported values are medians across three independent processes. Each process
used a dense-matmul clock ramp, an idle selected GPU, and three
`triton.testing.do_bench` measurements per configuration, each with a 200 ms
warmup and 500 ms of repetitions. The preserved
[summary CSV](data/forward-summary.csv) contains the aggregate throughputs used
in the tables and charts. It does not contain the individual process samples,
so we report no uncertainty interval derived from them.

The [reproduction harness](bench/bench_landed.py) makes the checks explicit for
new runs: compile and check finite outputs against an eager reference before
timing; ramp clocks after compilation for each configuration; monitor the
selected physical GPU for competing processes and clock changes; and retain
the individual timing trials and environment in JSONL. A failed check aborts
the run. The [benchmark instructions](bench/README.md) give the tolerances,
monitoring limits, source setup, and commands for retaining complete runs.

The [validation report](data/validation-2026-10-19/README.md) retains three
complete grad-disabled processes, their generated kernels, and their monitoring
evidence. A [targeted recheck](data/validation-2026-10-19/baseline-recheck/README.md)
identified the grad-mode mismatch in the first validation harness and reproduced
the historical baseline after restoring the original context. The grad-disabled
tables and charts remain separate from the article's grad-enabled measurements.

A separate [official PyTorch 2.14.1 validation](data/publication-2.14.1-2026-10-20/README.md)
passed all twelve unforced-default checks and three complete grad-enabled
processes, retaining 108 measurements and 648 timing trials. It includes new
release charts and a compiler comparison; the tables and figures in this article
retain their historical development-build attribution.

### Using the updated defaults

A PyTorch ROCm build containing the changes uses the updated defaults without
an opt-in flag:

```python
import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

B, H, S, D = 1, 16, 4096, 128
q, k, v = (torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
           for _ in range(3))
# The mask ignores batch and head indices, so its storage can be shared.
block_mask = create_block_mask(
    lambda b, h, q_idx, kv_idx: q_idx >= kv_idx,
    None, None, S, S, device="cuda",
)
compiled = torch.compile(flex_attention, fullgraph=True)
out = compiled(q, k, v, block_mask=block_mask, scale=D**-0.5)
```

To reproduce the comparisons on an exclusively allocated GPU, run the bundled
harness in three fresh processes, reversing configuration order in the second:

```bash
mkdir -p results
HIP_VISIBLE_DEVICES=0 python bench/bench_landed.py --output results/run1.jsonl
HIP_VISIBLE_DEVICES=0 python bench/bench_landed.py --reverse-order --output results/run2.jsonl
HIP_VISIBLE_DEVICES=0 python bench/bench_landed.py --output results/run3.jsonl
python bench/aggregate_landed.py 'results/run*.jsonl'
```

Run these commands from this article's directory. They create a new measurement
campaign; the published summary data stays separate. To tune other shapes, use
`torch.compile(flex_attention, mode="max-autotune", fullgraph=True)` and validate
the selected configuration on your workload.

## One default table per architecture

MI300X (gfx942) and MI350X (gfx950) have different LDS capacities and matrix
throughput, so they can favor different tile sizes and wavefront counts. These
settings balance data reuse and latency hiding against the resources needed by
each workgroup. A separate default table for each architecture[^9] allows either
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

## Summary

Revisiting two defaults recovered substantial MI350X performance: four
wavefronts avoid replicated work and a layout conversion in the inspected fp16
kernel, and two pipeline stages improve overlap across key/value iterations.
Together, they deliver 2.1x to 3.0x higher fp16 forward throughput on the measured
shapes, with 1.11x to 1.18x from pipelining alone on bf16.

The practical lesson is to tune the configuration as a whole when bringing up
an architecture. Per-architecture tables give validated defaults a home, and
compiler inspection helps explain which combinations deserve measurement.
`max-autotune` and `kernel_options` expose the same choices for your own shapes.

## Compiler appendix

### How Triton assigns wavefronts

In the inspected compiler, [`planWarps`](https://github.com/triton-lang/triton/blob/c8305cccdeb7c2eb07441a5e696e7f9b17cd8e24/third_party/amd/lib/TritonAMDGPUTransforms/AccelerateAMDMatmul.cpp)
recognizes chained matrix products. The first receives `[num_warps, 1]` to keep
softmax reductions within a wavefront. The second distributes wavefronts down
the rows first, capping that axis according to the output shape and the chosen
MFMA instruction's row dimension, then assigns the remaining wavefronts to
output columns.

For the inspected 32×32 MFMA cases, that row capacity is
`ceil(BLOCK_M / 32)`. At `BLOCK_M=128`, four wavefronts give matching layouts;
eight give `[8, 1]` followed by `[4, 2]`. Additional saved compiler inspections
reported the following layouts and static LDS write counts:

| `BLOCK_M` | `num_warps` | Wave layouts emitted | Static `ds_write` sites |
|---|---|---|---|
| 128 | 4 | `[4, 1]` | 9 |
| 128 | 2 | `[2, 1]` | 17 |
| 64 | 2 | `[2, 1]` | 9 |
| 128 | 8 | `[8, 1]` and `[4, 2]` | 138 |
| 64 | 8 | `[8, 1]` and `[2, 4]` | 136 |

Matching wave layouts avoids this particular conversion. Other LDS operations
remain, and these static counts alone do not establish the fastest
configuration for a different shape or architecture.

The [preserved IR and assembly](data/README.md#compiler-artifacts) for the
four- and eight-wavefront kernels let you inspect the main diagram's evidence.
There are 128 MFMA instruction sites at four wavefronts and 96 at eight;
weighting them gives 512 and 768 respectively. Loop execution and masking
still determine the dynamic instruction count.

### Other defaults and compiler maintenance

The architecture-table change also retains these fp32 entries on both gfx942
and gfx950: `(128, 32, 1, 4)` at head dimensions 64 and 128, and
`(64, 16, 1, 4)` at head dimension 256. The measurements here cover half
precision; they do not establish whether fp32 benefits from additional stages.

Two further Triton maintenance changes accompany this work: a driver support
check for cooperative groups[^10] and corrected architecture selection in a
`num_ctas` test.[^11] They improve the surrounding
compiler support and testing without contributing a separately measured speedup
in this article.

## Pull request references

All changes referenced below have landed in their respective upstream repositories.

[^1]: [PyTorch #180720: Change the fp16 default from eight to four wavefronts](https://github.com/pytorch/pytorch/pull/180720).
[^2]: [PyTorch #176676: Enable software pipelining for FlexAttention](https://github.com/pytorch/pytorch/pull/176676).
[^3]: [PyTorch #176675: Emit pointer-range annotations for small tensor arguments](https://github.com/pytorch/pytorch/pull/176675).
[^4]: [PyTorch #178541: Suppress pointer-range annotations for user-defined Triton kernels](https://github.com/pytorch/pytorch/pull/178541).
[^5]: [Triton #9619: Enable buffer operations for eligible 64-bit offsets](https://github.com/triton-lang/triton/pull/9619).
[^6]: [PyTorch #194136: Apply pointer-range guards to template kernels](https://github.com/pytorch/pytorch/pull/194136).
[^7]: [Triton #9912: Fix SSA dominance in buffer-operation lowering](https://github.com/triton-lang/triton/pull/9912).
[^8]: [Triton #10592: Pack fp32-to-bf16 RTZ conversions with v_perm_b32](https://github.com/triton-lang/triton/pull/10592).
[^9]: [PyTorch #181283: Add architecture-specific FlexAttention defaults](https://github.com/pytorch/pytorch/pull/181283).
[^10]: [Triton #8935: Check driver support for cooperative groups](https://github.com/triton-lang/triton/pull/8935).
[^11]: [Triton #8847: Correct architecture selection in the num_ctas test](https://github.com/triton-lang/triton/pull/8847).
