# Measurement notes and campaign history

This is the evidence companion to the [article](README.md). Historical tables
below preserve the measurements from the previous draft, with their campaign
boundaries. They are not a time series on one unchanged machine and compiler.
The [October 13 draft](https://github.com/nithinsubbiah/amd-gpu-perf/blob/1aab02806edd53534e2e52060ce98d1a5e25ebdb/blog/gfx1250-mha-prefill/README.md)
retains the full earlier experiment narrative.

All throughput tables use useful attention TFLOP/s unless labeled otherwise.
A reported percentage gain is throughput gain; the latest regression table in
the article explicitly reports **latency increase** instead. These percentages
are not interchangeable.

## Latest main comparison: #2641

[main-comparison.json](data/main-comparison.json) contains all 36 aggregate
rows (18 shapes × two dtypes), plus the 72 underlying run records. It retains
per-round timing samples, ordering, useful FLOP counts, and FP64 errors before
and after timing. Local filesystem paths and unrelated host telemetry are
omitted. It is measurement data, not a benchmark harness.

| Item | Recorded value |
|---|---|
| Target | gfx1250 |
| PyTorch | `2.14.0+rocm10.1.0` |
| HIP reported by PyTorch | `7.16.26385` |
| Triton version reported by these runs | `3.9.10` |
| Baseline commit | `b5474d8da835aa1e9562bc66f9e9bdcbb77913da` |
| Candidate kernel contents | Match `34bf268840a553cbc02b3dadcb41bb54d13180d0` |
| Final merge, not retimed here | `091a4fc66dc024458605af6db6e384d08d52cb2b` |
| Repetition | Two runs; nine alternating timing rounds per run |
| Timing | Warmed graph execution; repetition count retained per record |
| Correctness | Independent FP64 reference on sampled query rows before and after timing; whole-output finiteness |
| Exclusivity | Shared `gpu-lock`; monitor observed no foreign GPU clients |
| Clocks | Adaptive, not pinned |

The candidate was measured from a saved source file. Its SHA-256 is
`8b0aa29e4f07d1d0e023accb1b4cfaadbf4323e3870f53d0b752423909c306d1`;
this matches the kernel at the candidate commit above. The merged kernel has
SHA-256 `9f17fa440586a4046005ecdf6d18c47175758eed542944fc757eaea56d9cb895`.
Keeping these separate prevents a measurement of an earlier candidate from
becoming an unsupported claim about the final numerical fixes.

For each backend, the aggregate latency is the median of the two run medians.
With two runs, that is their arithmetic midpoint. Throughput is
`useful_flops / (median_us × 1e6)`, and speedup is
`baseline_latency / candidate_latency`. The geometric mean weights all 18
shapes equally, including the two regressions; it is not a traffic-weighted
application result. Run-to-run ranges are retained, but two runs do not provide
a strong estimate of long-term variance.

For a causal left window W, a query at position i sees
`min(i + 1, W + 1)` keys. Thus useful matrix work is
`4 × Hq × D × sum(min(i + 1, W + 1))` over every query in every sequence.
With no finite window, this becomes `2 × Hq × D × sum(L × (L + 1))`.

### Complete shape matrix

The following ratios come from the retained aggregate latencies. A value below
1.0 is a regression. Hq and Hkv are query and KV head counts.

| Lengths | Hq:Hkv | D | Window | BF16 speedup | FP16 speedup |
|---|---:|---:|---:|---:|---:|
| 4 × 4096 | 8:1 | 128 | Full | 1.155× | 1.224× |
| 16 × 512 | 8:1 | 64 | Full | 1.047× | 1.095× |
| 1 × 2048 | 32:8 | 128 | Full | 1.533× | 1.517× |
| 1 × 8192 | 8:8 | 128 | Full | 1.812× | 1.791× |
| 1 × 16384 | 8:1 | 128 | Full | 1.235× | 1.225× |
| 1 × 32768 | 8:1 | 128 | Full | 1.249× | 1.246× |
| 1 × 4096 | 8:1 | 128 | Full | 1.676× | 1.674× |
| 1 × 8192 | 8:1 | 128 | Full | 1.825× | 1.795× |
| `[16384, 12289, 8193]` | 8:1 | 128 | Full | 1.501× | 1.498× |
| `[4096, 3500, 3000]` | 32:8 | 128 | Full | 1.546× | 1.547× |
| `[851, 914, 1053]` | 32:8 | 128 | Full | 1.675× | 1.684× |
| `[4097, 129, 1, 0]` | 8:1 | 128 | Full | 4.366× | 4.352× |
| 32 × 128 | 32:8 | 128 | Full | 1.770× | 1.766× |
| 32 × 128 | 8:8 | 128 | Full | 1.206× | 1.203× |
| 32 × 128 | 8:1 | 128 | Full | 1.278× | 1.281× |
| 64 × 95 | 8:4 | 128 | Full | 1.571× | 1.567× |
| 8 × 1024 | 8:1 | 128 | Full | 0.948× | 0.948× |
| 4 × 4096 | 8:1 | 128 | 512 | 0.922× | 0.917× |

### Final review validation

The timing candidate passed 253 kernel tests and 28 generic API tests. After
further review fixes, the final targeted run passed 62 tests, with 402 tests
deselected. A negative control established that the prior kernel failed the
new large-value regression with non-finite output. These counts describe
separate runs. An interrupted broader run of an abandoned probability-scaling
prototype is not validation of the final value-scaling implementation.

The final refactor also preserved selection and applied grid/flag behavior
across 465,360 CPU-generated configurations. Tests and numerical documentation
are available in [#2641](https://github.com/lightseekorg/tokenspeed/pull/2641).
No GPU benchmark was rerun for this article update.

## Separate 90k vendor comparison

[90k-comparison.json](data/90k-comparison.json) retains the aggregate results,
all six runs' timing order and samples, and the 40 selected query positions
used for FP64 validation. This campaign used:

- TokenSpeed PR head `1d074b7099117bef74e258478bb07a390383083a`, with the same
  candidate kernel hash as the main comparison above;
- AITER `2bafed472893c6a8008758dec2591b9a9c73547b` and FlyDSL `0.4.0`;
- `tokenspeed-triton==3.9.10.post20261008`, PyTorch `2.14.0+rocm10.1.0`, and
  a 256-CU gfx1250 device;
- causal MHA B1/S90000/Hq40/Hkv40/D128, with BF16 and FP16;
- three fresh-process runs, 13 paired rounds per run, three graph replays per
  timed sample, GPU locking and process monitoring, and unpinned clocks.

Each backend's headline latency is the median of its three run medians.
The article shows the varlen entries; both batch and varlen entries remain
in the JSON. Assembly was available only for BF16 in the tested build. The
largest sampled relative RMS errors for TokenSpeed were about 0.141% in BF16
and 0.0176% in FP16. Validation of sampled rows is not validation of every
output element. The margins over the other implementations are small and
should not be generalized to other shapes or the subsequently fixed source.

## Historical geometry and scheduling campaigns

### #1332, #1337, and #1357

[#1332](https://github.com/lightseekorg/tokenspeed/pull/1332) paired BM256 with
eight waves. The first screen compared tile/wave combinations:

| Configuration | Relative to BM128/four waves |
|---|---:|
| BM256, four waves | 1.08x |
| BM128, eight waves | 0.96x |
| **BM256, eight waves** | **1.38x** |

The accepted BF16 Hq32/Hkv8/D128 results were:

| Shape | Before | After | Gain |
|---|---:|---:|---:|
| B1/S8192 | 1,181 | 1,651 TFLOP/s | 1.40x |
| B4/S4096 | 1,136 | 1,568 TFLOP/s | 1.38x |
| B4/S2048 | 983 | 1,311 TFLOP/s | 1.33x |
| B1/S2048 | 734 | 872 TFLOP/s | 1.19x |
| Ragged 851/914/1053 | 510 | 575 TFLOP/s | 1.13x |

The recorded protocol checked correctness and pinned GPU clocks to 2.4 GHz.
The previous draft did not retain a per-row trial count for this campaign;
these rows should not be assigned a statistical confidence interval.
[#1337](https://github.com/lightseekorg/tokenspeed/pull/1337) and
[#1357](https://github.com/lightseekorg/tokenspeed/pull/1357) hardened coverage
and simplified the selector, without claiming a further throughput gain.

### #1476, #1496, and #1505

The individual acceptance experiments used five order-balanced pairs at
2.4 GHz. #1476 measured max-ILP; #1496 measured TDM producer participation;
#1505 measured longest-first query ordering. Their acceptance results appear
in the main article. The max-ILP code-generation comparison was:

| Static metric | Default | max-ILP |
|---|---:|---:|
| VGPRs | 488 | 466 |
| VGPR bank switches | 824 | 638 |
| ALU waits | 126 | 98 |
| Instructions | 5,177 | 4,928 |

Longest-first gains varied with workgroup distribution:

| Occupancy case | Gain |
|---|---:|
| Two workgroups/CU, S4096 | +23.1% |
| Two workgroups/CU, S8192 | +25.8% |
| Three workgroups/CU | +10.7% |
| Eight workgroups/CU | +2.3% |
| One workgroup/CU | +0.7% |

A later replay enabled the three policies in sequence on the same source and
`tokenspeed_triton 3.8.10`. All 92 stage/shape measurements were correctness-
and clock-gated. The source snapshot SHA was not retained, so this supports
within-replay attribution only:

| Workload | Baseline | max-ILP | TDM producers | Longest-first | Cumulative |
|---|---:|---:|---:|---:|---:|
| BF16 GQA B4/H8-KV1/S4096 | 1,238 | 1,271 | 1,352 | 1,654 | **+33.6%** |
| FP16 GQA B4/H8-KV1/S4096 | 1,227 | 1,260 | 1,332 | 1,616 | **+31.7%** |
| FP8 E4M3, same shape | 1,754 | 1,915 | 1,991 | 2,531 | **+44.3%** |
| Ragged 851/914/1053, H32/KV8 | 556 | 596 | 620 | 675 | **+21.4%** |
| Sinks + LSE, H8/KV1/S4096 | 1,239 | 1,293 | 1,353 | 1,665 | **+34.4%** |

The complete stack's unweighted arithmetic-mean throughput gain across the
eight full-causal production-supplement rows was 31.60%. This is a different
statistic and a different experiment from #2641's 18-shape geometric mean.

## Profiling and rejected experiments

The roofline campaign used TokenSpeed `5eea3c0ea9` and
`tokenspeed-triton==3.8.10.post20260906`. These are measured/probe-specific
ceilings, not portable peak specifications:

| Quantity | Result |
|---|---:|
| Achievable HBM bandwidth | 17.72 TB/s |
| BF16 dependent-WMMA ceiling | 4,990 TFLOP/s |
| WMMA share of VALU issue slots | 56.872% |
| Raw issue-share upper bound | 2,837.9 TFLOP/s |
| Attention-core dependency roof | 2,460.1 TFLOP/s |

The no-L2-reuse memory estimate was 3,821 TFLOP/s for B4/S4096/Hq8/Hkv1/D128.
Production reached 1,671.4 TFLOP/s; a B2/S32768 point reached 2,033 TFLOP/s.
The trace from the #1496 baseline, before longest-first scheduling, reported:

| Stall class | Share |
|---|---:|
| Barriers | 26.6% |
| WMMA | 17.8% |
| Tensor waits | 16.0% |
| LDS waits | 9.2% |
| Packed VALU | 6.2% |
| Exponentials | 5.9% |

These rejected experiments used advancing sources on the `post20260906`
compiler. They are individual rejection records, not a same-source A/B matrix:

| Experiment | Result | Diagnosis |
|---|---|---|
| Extra producer/consumer warps | 145 TFLOP/s, 494 spills | AMD backend could not reallocate registers between partitions |
| Same-warp ready/empty mbarriers | 691 TFLOP/s, 55 spills | Barrier state crossed the register cliff |
| First dynamic three-tile schedule | 1,357 -> 1,325 TFLOP/s | 4,950 -> 8,062 instructions and longer live ranges |
| Packed-FP scalarization | 1,658 -> 1,615 TFLOP/s | VGPRs rose 467 -> 498 |
| Split accumulator scaling | 1,647 -> 1,591 TFLOP/s | Added dependency waits |
| Explicit TDM L2 prefetch | 1,666 -> 1,401-1,440 TFLOP/s | More overlap did not offset added scheduling pressure |
| Remove TDM barriers | Incorrect output/NaNs | Tensor waits do not provide cross-wave LDS visibility |

Later rejected experiments on `3.8.10.post20260920` were:

| Experiment | Result |
|---|---|
| Deferred accumulator rescale | BF16 B8/S1024 fell 940.6 -> 873.4 TFLOP/s (-7.1%); VGPRs rose 492 -> 510 with eight spills |
| LDS-to-TDM output | BF16 regressed 0.6% at S1024 and 1.2% at S4096; FP16 S1024 improved 0.1%, which is neutral |

The barrier-removal failures are correctness failures. The spill counts and
instruction growth help explain performance failures, but code-generation
statistics alone were never treated as a substitute for hardware timing.

## Historical pipeline and packed-layout campaigns

### #1883: deeper overlap and full-tile specialization

The development base was public TokenSpeed `fa9b11054d`, using
`tokenspeed-triton==3.8.10.post20260920`. B4/S4096/Hq8/Hkv1:

| D | Dtype | Production | Optimized | Gain |
|---:|---|---:|---:|---:|
| 64 | BF16 | 1,143.289 | 1,217.896 TFLOP/s | +6.526% |
| 64 | FP16 | 1,128.819 | 1,203.317 TFLOP/s | +6.600% |
| 128 | BF16 | 1,706.595 | 1,797.178 TFLOP/s | +5.308% |
| 128 | FP16 | 1,670.852 | 1,755.176 TFLOP/s | +5.047% |

Five balanced pairs per workload gave 40 passing endpoints and 20 positive
pairs, with clock gates. The D128 static comparison reduced VGPRs 496→492,
barriers 48→26, and instructions 4,320→3,821, without spills. The timing gains
belong to the combined pipeline and full-tile specializations.

### #1924: short, ragged, and windowed packed GQA

Candidate `986b7271` was compared with main `a4178a9d`, using
`3.8.10.post20260920`. These were separate, noninterleaved campaigns, and some
baseline measurements preceded hash-seed pinning. The attribution is weaker
than a fresh interleaved A/B; retain the caveat alongside the numbers.
All rows use D128 and Hq8/Hkv1:

| Workload | Dtype | Main | Candidate | Gain |
|---|---|---:|---:|---:|
| B8/S1024 | BF16 | 940.604 | 965.937 | +2.69% |
| B8/S1024 | FP16 | 926.250 | 969.381 | +4.66% |
| B4/S2048 | BF16 | 1,093.759 | 1,343.995 | +22.88% |
| B4/S2048 | FP16 | 1,086.203 | 1,224.341 | +12.72% |
| Ragged `[4096,3584,2305,1024]` | BF16 | 1,079.838 | 1,225.355 | +13.48% |
| Ragged `[4096,3584,2305,1024]` | FP16 | 1,069.063 | 1,206.647 | +12.87% |
| B4/S4096/window-512 | BF16 | 713.997 | 1,207.723 | +69.15% |
| B4/S4096/window-512 | FP16 | 709.232 | 1,183.293 | +66.84% |

The ragged batch was `[4096, 3584, 2305, 1024]`. The accepted ragged selector
covered other nonuniform four-sequence batches with maximum length 4096,
but this table does not establish gains for every such batch. The windowed
result combines packed ownership, mask handling, tile width, and scheduling.

### #1939: long BF16 packed routes

Rebased candidate `d4296da2`, compiler `3.8.10.post20260920`, and three
order-balanced pairs per workload:

| Workload | Existing route | Packed long route | Gain |
|---|---:|---:|---:|
| B4/S4096/H8/KV1/D128 BF16 | 1,817.521 | 1,908.198 TFLOP/s | +4.99% |
| B2/S8192/H8/KV1/D128 BF16 | 1,954.280 | 2,004.708 TFLOP/s | +2.58% |

All 12 endpoints passed correctness and clock gates. Pairwise gains were
4.94–5.03% at S4096 and 2.38–2.72% at S8192; maximum grouped-event CV was
0.113%. This PR merged as `e88f5558` and intentionally kept the selector to
the two measured BF16 routes.

### #1941: wider transport and instruction overlap

Final acceptance used main `c50a2f09` and compiler
`3.9.10.post20261008`. The result is composite: routes affected by removal of
a back-to-back-WMMA control were freshly rerun at candidate `8111c787`;
unaffected rows came from candidate `ec24b15c` under the matching release
campaign. The removed control is not part of the merged implementation.

Each campaign used three cyclic orders, seeded inputs, FP32 correctness gates,
grouped device-event timing, exclusive GPU locking, and clock telemetry.
Changed BF16 routes, all D128:

| Workload | Main | #1941 candidate | Gain |
|---|---:|---:|---:|
| B8/S1024 H8/KV1 | 1,009.420 | 1,081.001 | +7.09% |
| B4/S4096 H8/KV1 | 1,775.209 | 1,825.571 | +2.84% |
| B2/S8192 H8/KV1 | 1,852.726 | 1,989.113 | +7.36% |
| MHA B4/S4096 H8/KV8 | 1,716.488 | 1,749.812 | +1.94% |
| GQA B4/S4096 H32/KV8 | 1,685.644 | 1,781.200 | +5.67% |
| MQA B4/S4096 H32/KV1 | 1,724.719 | 1,818.613 | +5.44% |
| Ragged `[4096,3584,2305,1024]` | 1,032.361 | 1,246.486 | +20.74% |
| Sliding B4/S4096/window-512 | 1,154.712 | 1,231.263 | +6.63% |

Changed FP16 routes, all D128:

| Workload | Main | #1941 candidate | Gain |
|---|---:|---:|---:|
| B8/S1024 H8/KV1 | 984.686 | 1,088.932 | +10.59% |
| Ragged `[4096,3584,2305,1024]` | 1,053.109 | 1,255.265 | +19.20% |
| Sliding B4/S4096/window-512 | 1,124.550 | 1,219.566 | +8.45% |

Unspecified head counts in the ragged/window rows are Hq8/Hkv1. All 24 BF16
pairs favored the candidate; maximum grouped-event CV across final reruns
was 0.908%. The PR merged as `259a410b`.

## Historical October 13 assembly baseline

This table is the **pre-#2641** comparison preserved from the earlier article.
It is not the current kernel's ranking. It used TokenSpeed `676acaa5`,
`tokenspeed-triton==3.9.10.post20261008`, and AITER `b68b0e5c`, with four
order-balanced pairs per row passing FP32 correctness, sample-variability, and
clock gates. All rows are BF16 D128 full causal attention:

| Workload | TokenSpeed | AITER ASM | TokenSpeed delta |
|---|---:|---:|---:|
| B8/S1024 H8/KV1 | 1,122.741 | 1,076.356 | **+4.309%** |
| B4/S2048 H8/KV1 | 1,396.029 | 1,561.445 | -10.594% |
| B4/S4096 H8/KV1 | 1,984.203 | 1,974.260 | **+0.504%** |
| B2/S8192 H8/KV1 | 2,120.680 | 2,215.748 | -4.291% |
| B1/S16384 H8/KV1 | 2,046.347 | 2,362.868 | -13.396% |
| MHA B4/S4096 H8/KV8 | 1,914.618 | 1,979.250 | -3.265% |
| GQA B4/S4096 H32/KV8 | 1,909.226 | 1,927.814 | -0.964% |
| MQA B4/S4096 H32/KV1 | 1,950.156 | 1,982.261 | -1.620% |
| Ragged `[4096,3584,2305,1024]` | 1,249.077 | 1,480.615 | -15.638% |

The S4096 pair range crossed zero (−0.873% to +1.644%), so its positive
aggregate was statistical parity, not a stable win. S16384 used the then-
current generic eight-wave route and consistently trailed assembly. That
historical gap helped motivate subsequent long-context work; it should not be
carried forward as a claim about #2641. The ragged row uses Hq8/Hkv1.

## Source and validation lineage

The main article links all eleven optimization/validation PRs and credits the
kernel's other contributors. The test-path companion
[#1522](https://github.com/lightseekorg/tokenspeed/pull/1522) separated vendor
coverage. The later architecture-scoped CI follow-up
[#2743](https://github.com/lightseekorg/tokenspeed/pull/2743) is separate from
the kernel performance results; its purpose is to direct changed architecture
paths to the appropriate existing runner coverage. A test passing on gfx950
is not hardware validation of gfx1250 code.

Dependency changes are also separate from kernel attribution:
[#1410](https://github.com/lightseekorg/tokenspeed/pull/1410) moved TokenSpeed
to the `post20260906` compiler, and
[#1656](https://github.com/lightseekorg/tokenspeed/pull/1656) moved it to
`post20260920`. Earlier and later absolute throughput values should not be
combined into one claimed cumulative speedup.
