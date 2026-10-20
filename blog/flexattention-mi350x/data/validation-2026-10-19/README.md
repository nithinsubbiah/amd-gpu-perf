# Editorial and performance validation — 2026-10-19

**Correction after the baseline recheck:** the initial three-process campaign
below ran inside `torch.no_grad()`, while the original blog script used the
default grad-enabled context. This changes `OUTPUT_LOGSUMEXP` and the generated
kernel. The approximately 201 TFLOP/s baseline therefore measured a different
execution context. Replaying the original script recovered 240.8 TFLOP/s, close
to its historical 239.1. See the [controlled recheck](baseline-recheck/README.md).

The initial campaign and addressing ablation passed their numerical and
monitoring checks. Their retained numbers describe grad-disabled execution.
The article's historical grad-enabled tables and charts remain backed by
[the preserved summary](../forward-summary.csv).

## Initial grad-disabled forward results

| Comparison | Range of ratios of median throughputs |
|---|---|
| fp16, previous to updated configuration | 2.20–3.01x |
| fp16, eight to four wavefronts at two stages | 1.87–2.61x |
| bf16, one to two stages at four wavefronts | 1.10–1.18x |

[The full comparison tables](comparisons.md) include paired process ratios and
their observed ranges. [Statistics](statistics.json) retain the detailed
validation summary and each historical-to-fresh throughput difference.
[Fresh fp16](figures/01-fp16-speedup.png) and
[bf16 charts](figures/02-bf16-pipelining.png), with editable SVG counterparts,
are separate from the article's historical figures.

All 108 configuration measurements passed finite-output and reference checks,
with 324 timing-trial medians retained. The largest trial range was 1.52% of
its median. Across processes, the median configuration's throughput range was
0.38%; the largest was 4.51%. For the longer causal bf16 case, paired pipelining
ratios ranged from 1.077x to 1.124x. These are observed ranges, not confidence
intervals.

Maximum absolute output differences were 0.001953125 for fp16 and 0.015625 for
bf16, within the recorded absolute/relative tolerances. Every accepted point
retains three consecutive idle readings before warmup, process isolation during
timing, and generated-source evidence of the requested compiler settings.
Observed loaded SCLK ranged from 1337 to 2200 MHz across different kernels;
the largest sampled within-point clock range was 9.52%, despite stable timing
trials. These measurements use ordinary dynamic frequency scaling.

The historical unmasked fp16 `S=4096, D=64, stages=1, warps=8` baseline is
239.1 TFLOP/s; the grad-disabled median is 201.3. Its combined ratio is 2.50x
under this different context. The targeted recheck established that disabling
grad mode changes logsumexp output, register allocation, and reported occupancy
for this case. The historical baseline reproduces with grad mode restored.

For publication, retain the historical tables with the original execution
context. A replacement full campaign should consistently use the same
grad-enabled context. The initial grad-disabled campaign remains available as
a separate measurement; it should not be presented as a matched reproduction.

To reproduce the aggregation from the article directory:

```bash
python bench/aggregate_landed.py 'data/validation-2026-10-19/run*.jsonl'
python bench/plot_landed.py 'data/validation-2026-10-19/run*.jsonl' data/validation-2026-10-19/figures
```

## Memory-addressing ablation

Six fresh processes measure fp16 `B=1, H=16, S=4096, D=128`, holding
grad mode disabled and
`(BLOCK_M, BLOCK_N, num_stages, num_warps)=(128, 64, 2, 4)` fixed. Three enable
the pointer-range annotation and three disable it. Each process has a private
Inductor/Triton cache and uses the same correctness, idle-device, warmup,
monitoring, and trial-repeatability checks as the forward sweep. Setting order
alternates: on/off, off/on, on/off.

| Mask | Buffer path on, TFLOP/s | Buffer path off, TFLOP/s | Ratio of medians | Paired ratio range |
|---|---|---|---|---|
| none | 613.9 | 588.1 | 1.04x | 1.039–1.044x |
| causal | 399.6 | 305.5 | 1.31x | 1.301–1.324x |

[The machine-readable comparison](addressing-comparisons.json) preserves the
ratios before rounding. All 12 measurements passed correctness and monitoring
checks. Every process retains generated Python, compressed AMD assembly, and
an `ablation.json` manifest. Enabled kernels contain buffer loads and stores;
disabled kernels contain no buffer loads or stores and use global operations.
Some global loads remain in the enabled causal kernel.

The source identities, benchmark hashes, and protocol match across all six
processes; only the annotation flag and private cache paths differ. The
[analysis script](analyze_addressing.py) audits the monitoring records, timing
arithmetic, source hashes, annotation, and assembly counts before aggregating:

```bash
python data/validation-2026-10-19/analyze_addressing.py
```

The draft's approximately 1.33x historical addressing result is close to this
fresh 1.31x causal result, but its raw historical samples are unavailable.
The article now cites this retained ablation. This benefit is specific to the
workload and is not multiplied into the forward configuration gains.

After all GPU checks finished, SMI again reported 0% utilization on all eight
GPUs and no KFD processes.

## Environment and isolation

- AMD Instinct MI350X, gfx950; physical SMI GPU 6, PCI `0000:e5:00.0`.
- Python environment: `/home/nmeganat/flex/.venv-src`.
- PyTorch reports `2.14.0a0+gitb5460ee`, but its editable Python checkout is at
  `37200109b999958ae0042b21b2da3eea48ffec4a` with local changes. The run metadata
  hashes the active FlexAttention templates and relevant Inductor sources.
- Installed Triton distribution: `3.8.0+gitc8305ccc`. The available Triton
  checkout's different HEAD does not identify the installed compiler.
- [Checkout identities and source patches](source/checkouts.json) distinguish
  the current environment from the historical build.
- Compressed snapshots of `bench_landed.py`, `gpu_monitor.py`, and
  `bench_addressing.py` under `source/` preserve the measured harness versions.
- Initial SMI inspection found all eight GPUs idle, with no KFD processes.
  Every configuration additionally requires three consecutive 0% utilization
  readings and no other process mapped to the selected physical GPU.

The protocol uses eight seconds of dense matmul followed by three seconds of
the attention configuration before timing. Each configuration has three
`do_bench` medians, with 200 ms warmup and 500 ms repetitions each. A process
aborts on failed correctness, observed competition, unavailable monitoring,
loaded SCLK below 1000 MHz, or a trial range above 5% of the median. Three full
processes are required; the second reverses configuration order. Clocks and
all trials are retained. GPU clocks and power settings are unchanged.

Polling cannot reserve a device or detect every brief interference. No
scheduler reservation was obtained; the evidence establishes observed idleness
and isolation, rather than an enforced exclusive allocation.

## Protocol development and retained attempts

[The rejected attempts](attempts/) are excluded from the fresh campaign.
An early clock rule included observations near load transitions. Restricting
it to observations with at least 90% reported GPU utilization still rejected
an attempt whose SCLK ranged from 1643 to 1836 MHz, while its measured latencies
were `0.136325`, `0.136324`, and `0.136085` ms: a 0.18% range.

Consequently, the final protocol records instantaneous SCLK variation and uses
actual latency repeatability as its default stability criterion. A strict clock
range remains available through `--max-clock-spread-pct`. The revised protocol
is identical for all accepted full-process runs; rejected files are preserved
without mixing their measurements into the final aggregate.

`smoke-steady.jsonl` contains the successful four-configuration hardware check.
Its completion marker intentionally prevents inclusion in the full campaign.

## Editorial and artifact review

- All 18 historical comparison ratios agree with the preserved CSV; none of
  the original 30 distinct throughput values changed.
- Both explanatory diagrams render clearly. The wavefront mapping is scoped
  to the inspected fp16 `BLOCK_M=128, BLOCK_N=64, D=128` kernel.
- Saved assembly confirms 128 versus 96 MFMA sites, or 512 versus 768 sites
  weighted by wavefront count, and 9 versus 138 static LDS write sites. These
  are compiler counts, not dynamic instruction counts or a speedup prediction.
- Addressing text retains 64-bit addresses and narrows only proven offsets.
  Its storage bound is stated as less than 2 GiB. The template atomics guard
  is explained separately from forward load/store addressing.
- The fp32-to-bf16 packing claim specifies round-toward-zero and identifies
  gfx1250 as the target of its reported spill reduction.
- The original pipelining anecdote reproduces with matching grad mode.
  Its earlier removal was based on the grad-disabled campaign. The prose now
  restores the result and states the execution context explicitly.
- Local Markdown targets resolve, the four SVGs parse, and the figure scripts
  retain editable SVG text alongside PNG exports.
- Eleven CPU validation tests, Ruff, codespell, and `git diff --check` pass.

The backward atomics example is supported by its linked upstream PR and was
already in the draft. It has not been rerun here. New backward PR additions
remain outside this review.
