# Engineering a Faster FlashAttention Prefill Kernel on gfx1250

*Draft for technical review.*

## An evidence-driven journey through ten optimization-workstream PRs

Updated through [TokenSpeed #1941](https://github.com/lightseekorg/tokenspeed/pull/1941)
and the October 13, 2026 post-merge baseline.

This optimization workstream produced ten public pull requests for
TokenSpeed's gfx1250 Gluon MHA prefill kernel. Eight changed performance; two
hardened the selector and test coverage. The work progressed from tile
geometry, through compiler and launch scheduling, to a deeper
K-ahead/V-behind pipeline and packed-GQA specializations for short, ragged, and
sliding-window attention. The latest work extended packed ownership to long
BF16 routes, decoupled transport width from compute width, compacted ragged
launches, fused selected K/V transfers, and deliberately interleaved useful
VALU/LDS work between WMMAs.

These ten PRs are the optimization series, not the kernel's complete upstream
lineage. The implementation originally landed through
[#718](https://github.com/lightseekorg/tokenspeed/pull/718) and
[#719](https://github.com/lightseekorg/tokenspeed/pull/719), then received
generation-based organization in
[#880](https://github.com/lightseekorg/tokenspeed/pull/880), FP8 support in
[#953](https://github.com/lightseekorg/tokenspeed/pull/953), CDNA5 namespace
updates in [#1411](https://github.com/lightseekorg/tokenspeed/pull/1411), and
symbol naming updates in
[#1678](https://github.com/lightseekorg/tokenspeed/pull/1678).
A CI companion, [#1522](https://github.com/lightseekorg/tokenspeed/pull/1522),
moved the dedicated test path.

This article records what changed, what each change measured, what failed, and
why. Percentages from different campaigns are not added together. Cumulative
claims come only from experiments that replayed each policy on the same source,
toolchain, shape, and measurement protocol.

Across the separately reported campaigns:

* the public roofline campaign measured 2.033 PFLOP/s of useful causal work on
  B2/S32768/H8/KV1/D128 BF16 using TokenSpeed `5eea3c0ea9` with
  `3.8.10.post20260906`;
* a same-source/same-toolchain replay of
  [#1476](https://github.com/lightseekorg/tokenspeed/pull/1476),
  [#1496](https://github.com/lightseekorg/tokenspeed/pull/1496), and
  [#1505](https://github.com/lightseekorg/tokenspeed/pull/1505) produced an
  unweighted +31.60% mean across eight full-causal production-supplement rows;
* the combined
  [#1883](https://github.com/lightseekorg/tokenspeed/pull/1883)
  deep-pipeline/full-tile stack improved B4/S4096/H8/KV1 by 5.047-6.600% on
  the published compiler;
* candidate `986b7271` measured 2.69-69.15% above recorded main `a4178a9d`
  baselines on selected routes; these were separate, non-interleaved
  campaigns;
* [#1939](https://github.com/lightseekorg/tokenspeed/pull/1939) improved the
  long packed BF16 H8/KV1 routes by 4.99% at B4/S4096 and 2.58% at B2/S8192;
* [#1941](https://github.com/lightseekorg/tokenspeed/pull/1941) then improved
  every changed BF16 route by 1.94-20.74% and the three changed FP16 routes by
  8.45-19.20% on the released `3.9.10.post20261008` compiler; and
* the post-merge TokenSpeed `676acaa5` baseline reached 1,122.7 TFLOP/s at
  B8/S1024, 4.31% above the matched AITER ASM path, while exposing the next
  deficits at S2048, S16384, and ragged attention.

Those results came from specialization, not a universal fast path. Each
retained path has explicit eligibility guards. One important caveat is that
[#1924](https://github.com/lightseekorg/tokenspeed/pull/1924)'s ragged gate
accepts any nonuniform B4 batch with `max_seqlen == 4096`, while the reported
ragged performance uses one distribution:
`[4096, 3584, 2305, 1024]`.

## A mental model of the kernel

### Notation and hardware terms

| Term | Meaning |
|---|---|
| B / S | Batch size / sequence length |
| Hq / Hkv | Query-head count / KV-head count |
| D | Head dimension |
| BM / BN | Query-row tile / KV-row tile |
| MHA / GQA / MQA | Equal query/KV heads / grouped query heads / one KV head |
| CU | Compute unit |
| Wave | The 32-lane gfx1250 hardware execution group |
| Warp | The Triton/Gluon API term used by `num_warps`; on this target it maps to a wave32 |
| VGPR | Per-wave vector register allocation; excess live state causes spills |
| LDS | On-chip workgroup-shared memory |
| HBM | Off-chip high-bandwidth memory |
| TDM | Tensor data mover used for asynchronous global-to-LDS copies |
| WMMA | Matrix multiply-accumulate instruction |
| VALU | Vector arithmetic issue pipeline used by both WMMA and softmax/vector instructions |
| ILP | Instruction-level parallelism: independent instructions available for overlap |
| Online softmax | Streaming softmax that carries a running maximum and denominator across KV tiles |
| LSE | Optional log-sum-exp output |
| ATT | Instruction-level advanced thread trace |
| `mbarrier` | Memory barrier primitive used for explicit producer/consumer handshakes |
| coexec | Compiler scheduling strategy that co-schedules compute and memory work |

The easiest way to understand the optimizations is to follow one workgroup.
It owns a block of query rows and repeatedly walks blocks of K and V:

1. TDM moves a K or V block from global memory into LDS.
2. Waves load K from LDS and issue QK WMMAs.
3. The kernel applies causal or window masks and updates online softmax.
4. Waves load V and issue PV WMMAs into an FP32 output accumulator.
5. The final accumulator is normalized, rearranged into a coalesced store
   layout, cast, and written out.

```mermaid
flowchart LR
    Q["Q tile<br/>VGPR"] --> QK["QK<br/>WMMA"]
    K["K<br/>HBM"] --> TK["TDM K"] --> KL["K tile<br/>LDS"] --> QK
    QK --> M["Causal / window<br/>mask"] --> SM["Online<br/>softmax"] --> P["P tile"]
    P --> PV["PV<br/>WMMA"]
    V["V<br/>HBM"] --> TV["TDM V"] --> VL["V tile<br/>LDS"] --> PV
    PV --> A["FP32 output<br/>accumulator"] --> N["Normalize<br/>and cast"]
    N --> S["Coalescing<br/>LDS shuffle"] --> O["Output<br/>HBM"]

    classDef memory fill:#dbeafe,stroke:#2563eb,color:#172554
    classDef mover fill:#e0f2fe,stroke:#0284c7,color:#0c4a6e
    classDef matrix fill:#ede9fe,stroke:#7c3aed,color:#2e1065
    classDef softmax fill:#fef3c7,stroke:#d97706,color:#78350f
    classDef output fill:#dcfce7,stroke:#16a34a,color:#14532d

    class K,V,KL,VL memory
    class TK,TV mover
    class Q,QK,PV,A matrix
    class M,SM,P softmax
    class N,S,O output
```

Two LDS slots allow one tile to compute while a later tile is moving. That
sounds like a conventional double-buffered pipeline, but four constraints make
this kernel unusually sensitive:

* **The output accumulator is large.** Increasing the query tile increases
  useful work per workgroup, but also increases per-lane VGPR demand.
* **TDM is asynchronous but not free.** Descriptor production, tensor waits,
  LDS visibility, and workgroup barriers all affect overlap.
* **Softmax competes with matrix math.** On gfx1250, WMMA and vector softmax
  instructions consume shared issue capacity.
* **Causal work is imbalanced.** Later query blocks visit more KV tiles than
  early blocks, so dispatch order changes the tail even when kernel code does
  not.

That gives us four distinct optimization levers:

| Bottleneck | Lever |
|---|---|
| Too little useful work per workgroup | Tile geometry |
| Poor local instruction overlap | Compiler and software-pipeline scheduling |
| Too much transfer/synchronization participation | TDM producer specialization |
| CUs draining unevenly | Workgroup ordering |

The later packed-GQA work adds a fifth lever: change work ownership so query
heads sharing one KV head are computed together.

---

## 1. Scope and measurement contract

The target is TokenSpeed's packed, variable-length causal prefill kernel for
gfx1250. It supports:

* MHA, GQA, and MQA;
* D64 and D128;
* BF16, FP16, FP8 E4M3, and FP8 E5M2;
* uniform and ragged sequences;
* full and sliding-window attention;
* optional sinks and natural-log LSE output.

Most headline results use one 256-CU gfx1250 GPU at a verified 2.4 GHz. Useful
causal throughput is:

`2 * D * Hq * sum_s(S * (S + 1)) / time`

This counts QK and PV work over causal score elements. Sliding-window results
use the actual attended score count rather than full-causal FLOPs.
All throughput values below are useful TFLOP/s unless stated otherwise.
Absolute values from different tables should not be compared unless the shape,
dtype, source revision, and compiler are the same.

The campaigns span multiple source and compiler revisions. Exact provenance is
listed in the evidence map; only within-table comparisons are treated as
direct performance evidence.

The acceptance rules became stricter as the project progressed:

1. Run a PyTorch reference before timing.
2. Hold the GPU lock and reject foreign processes.
3. Verify clocks throughout the run; discard SUSPECT rows.
4. Use independent processes and order-balanced control/candidate pairs.
5. Pin Python's hash seed and isolate JIT caches when code generation depends
   on Python object ordering.
6. Reject timing rows with unstable grouped-event samples.
7. Check static resources before timing: a promising source transformation
   that spills is not a candidate.

These rules matter. Early in the project, an aggregate sweep moved the clock by
11.4% under a 2,030 W load. It produced plausible numbers, but the complete
sweep was discarded.

---

## 2. The optimization roadmap

| PR | What it teaches | Measured result |
|---|---|---:|
| [#1332](https://github.com/lightseekorg/tokenspeed/pull/1332) | Match a wider M tile with enough waves to hold its accumulator | Up to 1.40x |
| [#1337](https://github.com/lightseekorg/tokenspeed/pull/1337) | Force both tile families through correctness tests | Coverage and selector hardening |
| [#1357](https://github.com/lightseekorg/tokenspeed/pull/1357) | Remove a runtime query when the architecture already fixes the answer | Same decisions, lower launch-side complexity |
| [#1476](https://github.com/lightseekorg/tokenspeed/pull/1476) | Apply an LLVM schedule only where measured codegen improves | +3.27% acceptance case |
| [#1496](https://github.com/lightseekorg/tokenspeed/pull/1496) | Restrict TDM descriptor production without removing compute waves | +5.94% acceptance case |
| [#1505](https://github.com/lightseekorg/tokenspeed/pull/1505) | Start expensive causal workgroups first to shorten the dispatch tail | +23.08% acceptance case |
| [#1883](https://github.com/lightseekorg/tokenspeed/pull/1883) | Overlap K and V at different depths, then specialize provably full rows | +5.0-6.6% on B4/S4096 |
| [#1924](https://github.com/lightseekorg/tokenspeed/pull/1924) | Pack GQA rows and specialize ragged/sliding boundaries | +2.7-69.2% on selected routes |
| [#1939](https://github.com/lightseekorg/tokenspeed/pull/1939) | Extend packed ownership to measured long BF16 H8/KV1 routes | +4.99% S4096; +2.58% S8192 |
| [#1941](https://github.com/lightseekorg/tokenspeed/pull/1941) | Decouple transport/compute width, compact ragged work, fuse selected TDM loads, and fill WMMA issue gaps | +1.94-20.74% on changed BF16 routes |

All ten optimization-workstream PRs merged. The
[#1924](https://github.com/lightseekorg/tokenspeed/pull/1924) tables in this
article remain anchored to candidate commit `986b7271`; the later sections use
the independently measured #1939 and #1941 campaigns rather than combining
percentages across toolchains.

### The kernel before this optimization series

[#718](https://github.com/lightseekorg/tokenspeed/pull/718) introduced the
implementation on a staging branch, and
[#719](https://github.com/lightseekorg/tokenspeed/pull/719) landed it on
`main`. The initial kernel already supported:

* FP16/BF16 with D64 and D128;
* packed/ragged `cu_seqlens`;
* causal and sliding-window attention;
* sinks and optional natural-log LSE output; and
* TDM K/V staging with an explicit prologue, steady-state loop, and epilogue.

[#880](https://github.com/lightseekorg/tokenspeed/pull/880) reorganized the
kernel under the generation-specific gfx1250 path without changing the
algorithm. [#953](https://github.com/lightseekorg/tokenspeed/pull/953) added
native FP8 E4M3/E5M2 WMMA paths while preserving BF16 output. None of these
PRs reported performance numbers for this kernel.

---

## 3. Establishing the geometry: [#1332](https://github.com/lightseekorg/tokenspeed/pull/1332), [#1337](https://github.com/lightseekorg/tokenspeed/pull/1337), and [#1357](https://github.com/lightseekorg/tokenspeed/pull/1357)

### 3.1 BM256's full gain required eight waves

The original production configuration used a 128x64 query/KV tile with four
waves and two K/V buffers. Widening the query tile to BM256 produced its full
gain only when the workgroup also grew to eight waves. BM256/four waves reached
1.08x; BM256/eight waves reached 1.38x and was selected.

Warps partition the M dimension. With BM256 and only four waves, each lane owns
twice as much FP32 accumulator state. Doubling the wave count halves that
per-lane burden and keeps the tile away from the spill cliff.

```mermaid
flowchart TB
    subgraph N["Baseline: BM128 / 4 waves"]
        direction LR
        N0["W0<br/>32 rows"] --- N1["W1<br/>32 rows"] --- N2["W2<br/>32 rows"] --- N3["W3<br/>32 rows"]
    end

    subgraph W4["Widen only: BM256 / 4 waves"]
        direction LR
        F0["W0<br/>64 rows"] --- F1["W1<br/>64 rows"] --- F2["W2<br/>64 rows"] --- F3["W3<br/>64 rows"]
    end

    subgraph W8["Selected: BM256 / 8 waves"]
        direction LR
        E0["W0<br/>32"] --- E1["W1<br/>32"] --- E2["W2<br/>32"] --- E3["W3<br/>32"]
        E3 --- E4["W4<br/>32"] --- E5["W5<br/>32"] --- E6["W6<br/>32"] --- E7["W7<br/>32"]
    end

    N -->|"More work per CTA"| W4
    W4 -->|"Restore per-wave accumulator size"| W8

    classDef baseline fill:#dbeafe,stroke:#2563eb,color:#172554
    classDef pressure fill:#fee2e2,stroke:#dc2626,color:#7f1d1d
    classDef selected fill:#dcfce7,stroke:#16a34a,color:#14532d

    class N0,N1,N2,N3 baseline
    class F0,F1,F2,F3 pressure
    class E0,E1,E2,E3,E4,E5,E6,E7 selected
```

The isolated configuration screen in
[#1332](https://github.com/lightseekorg/tokenspeed/pull/1332) showed:

| Configuration | Relative to BM128/four waves |
|---|---:|
| BM256, four waves | 1.08x |
| BM128, eight waves | 0.96x |
| **BM256, eight waves** | **1.38x** |

The accepted headline shape results used BF16, Hq32/KV8, and D128:

The GPU was pinned at 2.4 GHz, and every configuration passed the per-sequence
causal reference before timing. The PR did not report a per-row trial count for
this shape table.

| Shape | Before | After | Gain |
|---|---:|---:|---:|
| B1/S8192 | 1,181 | 1,651 TFLOP/s | 1.40x |
| B4/S4096 | 1,136 | 1,568 TFLOP/s | 1.38x |
| B4/S2048 | 983 | 1,311 TFLOP/s | 1.33x |
| B1/S2048 | 734 | 872 TFLOP/s | 1.19x |
| Ragged 851/914/1053 | 510 | 575 TFLOP/s | 1.13x |

The wider tile has two costs:

* it halves the workgroup grid; and
* it doubles the causally masked half of the diagonal tile.

The selector therefore requires `max_seqlen >= 1024` and a rectangular grid of
at least 256 workgroups:
`batch_size * n_heads * ceil(max_seqlen / 256)`. This early tile selector does
not count live ragged workgroups; that distinction was added later for the
longest-first ordering policy. Shapes that fail either gate keep BM128. Those
guards held 1x1024, 6x512, and underfilled H8/KV1 shapes within noise.

### 3.2 [#1337](https://github.com/lightseekorg/tokenspeed/pull/1337) and [#1357](https://github.com/lightseekorg/tokenspeed/pull/1357): coverage and selector correction

[#1337](https://github.com/lightseekorg/tokenspeed/pull/1337) added direct
coverage for both tile families, D64/D128, and full/sliding attention. Existing
tests selected only the narrow tile, while naturally selecting BM256 required
too much simulator work. The new tests force each configuration on a small
shape and assert that the override actually took effect.

The merged PR also pinned the two selector boundaries and added a cached runtime
CU-count query. Review feedback arrived after merge: gfx1250 has a fixed
architectural count of 256, so
[#1357](https://github.com/lightseekorg/tokenspeed/pull/1357) replaced that
query with a module constant. Tile decisions remained unchanged.

This pair did not create a headline throughput gain. It made the first gain
reviewable and protected it against silent layout or launch regressions.

---

## 4. Scheduling the existing work better: [#1476](https://github.com/lightseekorg/tokenspeed/pull/1476), [#1496](https://github.com/lightseekorg/tokenspeed/pull/1496), and [#1505](https://github.com/lightseekorg/tokenspeed/pull/1505)

The next three PRs did not change attention math. They changed compiler
scheduling, TDM work participation, and dispatch order.

### 4.1 [#1476](https://github.com/lightseekorg/tokenspeed/pull/1476): select max-ILP where it wins

[#1476](https://github.com/lightseekorg/tokenspeed/pull/1476) selected LLVM's
`max-ilp` scheduler for full D128 attention at S512 and above.

Why can a compiler policy matter when the algorithm is unchanged? QK, softmax,
PV, address generation, and asynchronous-copy bookkeeping create several
partially independent instruction chains. A scheduler can expose more of that
independence—or extend live ranges until register pressure erases the benefit.
The right policy is therefore shape-specific, not an architecture-wide switch.

The schedule was already spill-free. The measured gain coincided with lower
VGPR use, bank switches, ALU waits, and instruction count; the campaign does
not isolate which codegen change caused it:

| Static metric | Default | max-ILP |
|---|---:|---:|
| VGPRs | 488 | 466 |
| VGPR bank switches | 824 | 638 |
| ALU waits | 126 | 98 |
| Instructions | 5,177 | 4,928 |

Five order-balanced pairs (ten runs total) at a verified 2.4 GHz improved
1,510.4 to 1,559.8 TFLOP/s, **+3.27%**. All ten runs passed the reference and
had 0.00% clock spread.

The head commit retained an earlier 1,518.0 to 1,569.6 result (+3.40%); this
article uses the final PR-body acceptance campaign.

The selector deliberately excludes D64, short sequences, and sliding windows.
Medium sliding windows regressed by as much as 7.4% under max-ILP.

### 4.2 [#1496](https://github.com/lightseekorg/tokenspeed/pull/1496): specialize TDM producer participation

The eight-wave kernel used TDM descriptors for K and V. By default, all eight
waves participated in descriptor work.

[#1496](https://github.com/lightseekorg/tokenspeed/pull/1496) supplied
`warp_used_hint=0x0F`, assigning that work to four waves while preserving:

* separate K and V completion events;
* issue order;
* wait distances; and
* all eight compute waves.

The key distinction is between **producing the transfer descriptor** and
**retaining compute capacity**. This change narrows descriptor participation;
it does not convert four waves into dedicated producers or remove them from QK
and PV. That avoided the register-partition problems encountered by later,
heavier warp-specialization experiments.

On B4/S4096/H8/KV1/D128 BF16, five correctness-gated runs per variant (ten
total), order-balanced at a verified 2.4 GHz, measured:

| TDM participation | TFLOP/s |
|---|---:|
| Eight waves | 1,283.4 |
| **Four producer waves** | **1,359.6** |

The gain was **+5.94%**. Additional paired gains ranged from +3.0% on ragged
attention to +5.8% on FP16 D128. D64, sinks/LSE, and both FP8 formats also
improved.

Sliding windows regressed, so the policy requires full attention, the
BM256/BN64/eight-wave geometry, and at least 256 workgroups.

A separate fused-event screen reached 1,455 TFLOP/s because K then waited for V
completion. The retained summary does not identify that row's exact matched
control, so it is not compared directly with the 1,283.4 to 1,359.6 acceptance
campaign above.

Post-[#1496](https://github.com/lightseekorg/tokenspeed/pull/1496) follow-up
experiments, not changes shipped in the PR, found:

* three-buffer pipelining regressed 1,361 to 1,281 TFLOP/s;
* coarse four-stage warp pipelining reached only 1,018 TFLOP/s; and
* ready-only mbarriers deadlocked, showing that this producer/consumer
  implementation required both ready and empty-buffer handshakes.

### 4.3 [#1505](https://github.com/lightseekorg/tokenspeed/pull/1505): schedule the longest causal workgroups first

Causal query blocks do unequal work. On BM256/BN64 at S4096, the first query
block visits four KV tiles; the last visits 64. Ascending dispatch leaves the
most expensive workgroups in the tail after most CUs have gone idle.

The scheduling idea is the same one used for unequal CPU jobs: start the
longest jobs first, then let short jobs fill the holes near the end. Here the
"job length" is monotonic in query-block position, so reversing logical block
order approximates longest-processing-time scheduling without building a
runtime queue.

```mermaid
flowchart TB
    subgraph ASC["Ascending logical order"]
        direction LR
        A0["q0<br/>4 KV tiles"] --> A1["q1<br/>8 KV tiles"] --> AM["…"] --> A15["q15<br/>64 KV tiles"] --> AT["Expensive<br/>dispatch tail"]
    end

    subgraph DESC["Longest-first logical order"]
        direction LR
        D15["q15<br/>64 KV tiles"] --> DM["…"] --> D1["q1<br/>8 KV tiles"] --> D0["q0<br/>4 KV tiles"] --> DT["Cheap<br/>final drain"]
    end

    ASC -->|"Same workgroups and outputs;<br/>only ID mapping changes"| DESC

    classDef short fill:#dcfce7,stroke:#16a34a,color:#14532d
    classDef medium fill:#fef3c7,stroke:#d97706,color:#78350f
    classDef long fill:#fee2e2,stroke:#dc2626,color:#7f1d1d
    classDef tail fill:#f3f4f6,stroke:#6b7280,color:#111827

    class A0,D0 short
    class A1,D1,AM,DM medium
    class A15,D15 long
    class AT,DT tail
```

[#1505](https://github.com/lightseekorg/tokenspeed/pull/1505) maps early
physical IDs to late logical query blocks. Per-workgroup math, memory accesses,
and output addresses do not change.

Five correctness-gated CLEAN runs per order (ten total), order-balanced at
2.4 GHz, measured:

| Order | Mean TFLOP/s |
|---|---:|
| Ascending | 1,357.0 |
| **Longest-first** | **1,670.2** |

That is **+23.08%**.

The effect is a dispatch-tail result, not a local codegen result. VGPRs,
spills, WMMAs, barriers, waits, and memory instructions remained effectively
unchanged. Instrumented dispatch duration fell from 102,233 to 81,843 ns while
shader busy percentage stayed at 99.6-99.7%.

The gain depends on workgroup distribution:

| Occupancy case | Gain |
|---|---:|
| Two workgroups/CU, S4096 | +23.1% |
| Two workgroups/CU, S8192 | +25.8% |
| Three workgroups/CU | +10.7% |
| Eight workgroups/CU | +2.3% |
| One workgroup/CU | +0.7% |

The final selector counts live ragged workgroups from `cu_seqlens_cpu`, not the
rectangular max-sequence grid. With one grid head, an adversarial
`[4096] + [1] * 31` batch has 512 rectangular groups but only 47 live groups
and correctly stays in ascending order.

### 4.4 Isolated attribution of the three-PR stack

We later replayed the three policies on one source revision and one compiler,
forcing each feature off or on in sequence. All 92 measurements were
correctness-gated, independently locked, and clock-stable. One measurement is
one isolated stage/shape timing. The active-row counts overlap because a shape
can activate more than one policy; the eight-row cumulative mean uses only the
full-causal production supplement. The harness did not persist the source SHA,
so this table supports within-replay attribution, not comparison with another
campaign.

| Workload | Baseline | max-ILP | TDM producers | Longest-first | Cumulative |
|---|---:|---:|---:|---:|---:|
| BF16 GQA B4/H8-KV1/S4096 | 1,238 | 1,271 | 1,352 | 1,654 | **+33.6%** |
| FP16 GQA B4/H8-KV1/S4096 | 1,227 | 1,260 | 1,332 | 1,616 | **+31.7%** |
| FP8 E4M3, same shape | 1,754 | 1,915 | 1,991 | 2,531 | **+44.3%** |
| Ragged 851/914/1053, H32/KV8 | 556 | 596 | 620 | 675 | **+21.4%** |
| Sinks + LSE, H8/KV1/S4096 | 1,239 | 1,293 | 1,353 | 1,665 | **+34.4%** |

Unweighted arithmetic means across active rows were:

* [#1476](https://github.com/lightseekorg/tokenspeed/pull/1476):
  **+3.24%** across 18 rows;
* [#1496](https://github.com/lightseekorg/tokenspeed/pull/1496):
  **+4.60%** across 19 rows;
* [#1505](https://github.com/lightseekorg/tokenspeed/pull/1505):
  **+16.22%** across 20 rows; and
* the complete stack: **+31.60%** across eight full-causal rows in the
  production supplement.

H32/KV8 had many more workgroups and gained 8.5% cumulatively; H8/KV1 at 512
workgroups gained 33.6%.

At this point, local scheduling and global work distribution had both moved.
The next step was to determine which physical ceiling remained, rather than
continuing to tune by intuition.

---

## 5. Roofline analysis changed the optimization target

We next measured HBM, WMMA, issue-share, and dependency ceilings.

On the exact TokenSpeed toolchain, the measured ceilings were:

| Quantity | Result |
|---|---:|
| Achievable HBM bandwidth | 17.72 TB/s |
| BF16 dependent-WMMA ceiling | 4,990 TFLOP/s |
| WMMA share of VALU issue slots | 56.872% |
| Raw issue-share upper bound | 2,837.9 TFLOP/s |
| Attention-core dependency roof | 2,460.1 TFLOP/s |

```mermaid
flowchart LR
    W["BF16 WMMA ceiling<br/>4,990 TFLOP/s"] --> I["× 56.872%<br/>WMMA issue share"]
    I --> U["Raw issue upper bound<br/>2,837.9 TFLOP/s"]
    U --> D["Account for LDS +<br/>QK → softmax → PV dependencies"]
    D --> R["Attention-core roof<br/>2,460.1 TFLOP/s"]
    M["No-L2 memory roof<br/>3,821 TFLOP/s"] -. "higher; not binding" .-> R
    R --> S4["S4096 production<br/>1,671.4 = 67.9%"]
    R --> S32["S32768 roofline run<br/>2,033 = 82.6%"]

    classDef ceiling fill:#ede9fe,stroke:#7c3aed,color:#2e1065
    classDef transform fill:#fef3c7,stroke:#d97706,color:#78350f
    classDef memory fill:#dbeafe,stroke:#2563eb,color:#172554
    classDef result fill:#dcfce7,stroke:#16a34a,color:#14532d

    class W,U,R ceiling
    class I,D transform
    class M memory
    class S4,S32 result
```

The attention-core probe retained the production LDS load, QK WMMA,
online-softmax, rescale, and PV WMMA dependency chain, while amortizing
TDM/HBM and dispatch-tail costs. It was a more realistic optimization target
than raw WMMA peak or issue share.

This is the important distinction between a machine roof and a kernel roof.
The GPU can sustain nearly 5 PFLOP/s in a dependent WMMA probe, but attention
must interleave matrix instructions with max reductions, exponentials,
normalization, LDS reads, and synchronization. Those operations consume issue
slots and create serial dependencies that a pure matrix benchmark does not
contain.

For B4/S4096/H8/KV1/D128 BF16:

* useful work was 137.5 GFLOP;
* the memory roof assuming no K/V L2 reuse was still 3,821 TFLOP/s, above the
  dependency roof;
* production reached 1,671.4 TFLOP/s, or 67.9% of the core roof; and
* the roofline campaign's B2/S32768/H8/KV1/D128 BF16 max-ILP point reached
  2,033 TFLOP/s, or 82.6%.

ATT on the merged
[#1496](https://github.com/lightseekorg/tokenspeed/pull/1496) baseline, before
[#1505](https://github.com/lightseekorg/tokenspeed/pull/1505), showed where
time went:

| Stall class | Share |
|---|---:|
| Barriers | 26.6% |
| WMMA | 17.8% |
| Tensor waits | 16.0% |
| LDS waits | 9.2% |
| Packed VALU | 6.2% |
| Exponentials | 5.9% |

Only 41-43% of co-resident cycles had at least one wave issuing. The kernel was
not primarily HBM-bound. The remaining work was synchronization and the
QK-softmax-PV dependency chain.

That result changed the optimization question. We no longer asked, "How do we
move fewer HBM bytes?" We asked:

* Can K and V arrive earlier without increasing live state?
* Can full rows skip predicates and invalid-row repair?
* Can the output shuffle move fewer bytes through LDS?
* Can softmax reduce a fragment before scaling it, shortening the live range?

---

## 6. Register pressure and failed approaches

The roofline narrowed the search, but many plausible transformations still
failed. Static resource checks often explained why.

Unless stated otherwise, the table below concerns the
B4/S4096/H8/KV1/D128 BF16 production family on the `post20260906` public
compiler. The TokenSpeed source advanced between screens, so these are
independent rejection records, not one same-revision A/B matrix.

| Experiment | Result | Diagnosis |
|---|---|---|
| Extra producer/consumer warps | 145 TFLOP/s, 494 spills | AMD backend could not reallocate registers between partitions |
| Same-warp ready/empty mbarriers | 691 TFLOP/s, 55 spills | Barrier state crossed the register cliff |
| First dynamic three-tile schedule | 1,357 -> 1,325 TFLOP/s | 4,950 -> 8,062 instructions and longer live ranges |
| Packed-FP scalarization | 1,658 -> 1,615 TFLOP/s | VGPRs rose 467 -> 498 |
| Split accumulator scaling | 1,647 -> 1,591 TFLOP/s | Added dependency waits |
| Explicit TDM L2 prefetch | 1,666 -> 1,401-1,440 TFLOP/s | More overlap did not offset added scheduling pressure |
| Remove TDM barriers | Incorrect output/NaNs | Tensor waits do not provide cross-wave LDS visibility |

Two later screens used the `3.8.10.post20260920` compiler and are not part of
the roofline campaign above:

| Experiment | Result |
|---|---|
| Deferred accumulator rescale | BF16 B8/S1024 fell 940.6 -> 873.4 TFLOP/s (-7.1%); VGPRs rose 492 -> 510 with eight spills |
| LDS-to-TDM output | BF16 regressed 0.6% at S1024 and 1.2% at S4096; FP16 S1024 improved 0.1%, which is neutral |

Three general rules emerged:

1. **No-spill is necessary, not sufficient.** Several spill-free variants
   still lost because they exposed LDS latency or lengthened dependencies.
2. **Fewer instructions is not automatically faster.** One output-store
   variant removed roughly 100 instructions and two barriers but regressed.
3. **Synchronization cannot be deleted by inspection.** Removing barriers
   generated plausible but incorrect output.

The failed first deep-pipeline port was especially useful. The schedule itself
was not disproven; the dynamic implementation was. A later compile-time
specialization made the same underlying idea viable.

---

## 7. [#1883](https://github.com/lightseekorg/tokenspeed/pull/1883): deeper overlap plus full-tile specialization

[#1883](https://github.com/lightseekorg/tokenspeed/pull/1883) merged a deep
K-ahead/V-behind pipeline together with five full-tile optimizations. In the
deep schedule:

* K is prefetched for tile `t+2`;
* V is prefetched for tile `t+1`;
* K and V retain separate completion events; and
* the original loop is compile-time excluded from the specialization.

K and V do not have the same deadline. K must be ready before the next QK
matrix multiply. V is consumed only after QK and softmax produce probabilities.
The accepted steady state reflects those different deadlines:

| Current work | Previous work | Future movement |
|---|---|---|
| QK for tile `t` | Softmax/PV for tile `t-1` | K for `t+2`, V for `t+1` |

```mermaid
flowchart LR
    subgraph NOW["One steady-state interval"]
        direction TB
        KT2["TDM K(t+2)"]
        VT1["TDM V(t+1)"]
        QKT["QK(t)"]
        PVT1["Softmax + PV(t-1)"]
    end

    KT2 -. "extra K lead time" .-> QKT2["QK(t+2)"]
    VT1 -. "V consumed later" .-> PVT1F["PV(t+1)"]
    QKT -.-> SMT["Softmax / PV(t)"]

    classDef transfer fill:#dbeafe,stroke:#2563eb,color:#172554
    classDef matrix fill:#ede9fe,stroke:#7c3aed,color:#2e1065
    classDef softmax fill:#fef3c7,stroke:#d97706,color:#78350f
    classDef future fill:#f3f4f6,stroke:#6b7280,color:#111827

    class KT2,VT1 transfer
    class QKT matrix
    class PVT1 softmax
    class QKT2,PVT1F,SMT future
```

This is deeper than loading K and V together for `t+1`. It gives K an
additional tile of lead time without forcing V to occupy state equally early.
Compile-time specialization is important: the failed dynamic implementation
kept both schedules and their drains live, increasing instructions and VGPRs.

The five full-tile optimizations were:

| Optimization | Why it helps |
|---|---|
| Cast normalized output before the coalescing layout shuffle | The LDS shuffle moves BF16/FP16 instead of FP32, halving that payload |
| Remove all-invalid-row repair on proven full rows | Every active row has a valid causal key, so the generic `-inf` repair is redundant |
| Elide bounds masks on full Q loads, boundary tiles, output stores, and LSE stores | Removes predicates and address-selection work whose outcome is statically known |
| Keep the D128 maximum in scaled log2 units | Avoids repeatedly scaling both the old and new row maxima |
| Reduce QK before multiplying by the positive scale | Preserves `scale * max(QK) == max(scale * QK)` while avoiding a full scaled fragment live range |

The last transformation illustrates the difference between fewer operations
and shorter live ranges. Scaling every QK element first is mathematically
simple, but keeps another full fragment live. Reducing first leaves only one
row maximum to scale.

Public TokenSpeed base `fa9b11054d` was used during
[#1883](https://github.com/lightseekorg/tokenspeed/pull/1883) development. The
public performance table below is packed causal B4/S4096/Hq8/Hkv1 measured with
`tokenspeed-triton==3.8.10.post20260920`:

| D | Dtype | Production | Optimized | Gain |
|---:|---|---:|---:|---:|
| 64 | BF16 | 1,143.289 | 1,217.896 TFLOP/s | +6.526% |
| 64 | FP16 | 1,128.819 | 1,203.317 TFLOP/s | +6.600% |
| 128 | BF16 | 1,706.595 | 1,797.178 TFLOP/s | +5.308% |
| 128 | FP16 | 1,670.852 | 1,755.176 TFLOP/s | +5.047% |

Each row is the mean of five order-balanced pairs. All 40 retained
invocations were correctness-gated and clock-stable, and all 20 pairs favored
the optimized kernel.

Static D128 codegen improved from 496 to 492 VGPRs, 48 to 26 barriers, and
4,320 to 3,821 instructions with zero spills. D64 fell from 356 to 344 VGPRs,
48 to 26 barriers, and 3,327 to 2,845 instructions.

The final selector is intentionally opportunistic:

* BF16 or FP16;
* D64 or D128;
* BM256/BN64, eight waves, two buffers;
* full causal attention;
* enough live workgroups;
* uniform, BM-aligned sequences; and
* S2048 minimum for D64 because S1024 was neutral.

Ragged, tail, sliding, underfilled, short D64, and FP8 launches keep the prior
path.

---

## 8. [#1924](https://github.com/lightseekorg/tokenspeed/pull/1924): packed GQA, ragged tails, and sliding windows

Because [#1883](https://github.com/lightseekorg/tokenspeed/pull/1883)
deliberately left short, ragged, and sliding-window launches on the established
path, the next work focused on changing work ownership and specializing the
remaining boundaries. The mechanisms screened were:

* pack `(query position, query head in group)` into one BM tile;
* launch one grid head per KV head;
* use a finite running-max seed for fully masked rows;
* split masked boundaries from clean interior work; and
* use two K/V ping-pong buffers.

Not every candidate worked:

* deferred accumulator rescaling created eight spills and regressed about 7.1%;
* Q staging through TDM reduced instructions but regressed 0.4-0.6%;
* a generic LDS-to-TDM output path was neutral to slower;
* packed TDM output exposed a multi-batch store-layout hazard; and
* a unified one/two/deep packed pipeline generated 159 spills.

At [#1924](https://github.com/lightseekorg/tokenspeed/pull/1924) candidate
commit `986b7271`, the design is narrower:

* BM128/four-wave packed GQA with `waves_per_eu=2`;
* the deep schedule retained on ragged tails with explicit row guards;
* a finite `-1e30` running-max seed so fully masked rows avoid
  `-inf - (-inf)` without per-tile invalid repair;
* scaled softmax state on complete packed rows;
* compile-time mask elision on aligned sliding interiors; and
* BN32 plus max-memory-clause scheduling for window 512.

Several configuration names deserve explanation:

* `waves_per_eu=2` is an occupancy hint: it asks code generation to preserve
  enough resources for two resident waves per execution unit, providing
  latency-hiding capacity without changing the logical tile.
* **Scaled softmax state** keeps the running D128 maximum in the same log2
  domain used by `exp2`, avoiding repeated conversion of row maxima.
* **BN32** halves the KV tile width used by the established BN64 path. It
  doubles loop stages, but gives the window boundaries finer granularity and
  can reduce masked boundary work. That rationale is a hypothesis; BN32 was
  selected empirically as part of the combined route.
* **max-memory-clause** asks LLVM to keep compatible independent memory
  operations adjacent so the backend can form longer memory clauses before
  dependent compute. The expected benefit is more memory-level overlap, but
  its contribution was not isolated from BN32 and mask elision.

### Packing query heads that share K/V

In H8/KV1 GQA, eight query heads read the same K/V head. The established path
assigns workgroups along the query-head axis. The packed path instead maps each
BM row to:

`row -> (query_position, query_head_within_group)`

With BM128 and group size eight, one workgroup covers 16 query positions across
all eight query heads. The grid's head axis shrinks from query heads to KV
heads, and K/V movement is shared by the packed query-head work.

```mermaid
flowchart LR
    subgraph LEGACY["Query-head grid"]
        KV0L["KV head 0"]
        KV0L --> L0["WG q-head 0"]
        KV0L --> L1["WG q-head 1"]
        KV0L --> L2["WG q-head 2"]
        KV0L --> LD["…"]
        KV0L --> L7["WG q-head 7"]
    end

    subgraph PACKED["KV-head grid: one packed BM128 workgroup"]
        KV0P["KV head 0"] --> TILE["16 query positions × 8 query heads"]
        TILE --> R0["rows 0–7<br/>q-pos 0, heads 0–7"]
        TILE --> R1["rows 8–15<br/>q-pos 1, heads 0–7"]
        TILE --> RD["…"]
        TILE --> R15["rows 120–127<br/>q-pos 15, heads 0–7"]
    end

    LEGACY -->|"Pack the shared-K/V group into M"| PACKED

    classDef kv fill:#dbeafe,stroke:#2563eb,color:#172554
    classDef legacy fill:#f3f4f6,stroke:#6b7280,color:#111827
    classDef packed fill:#dcfce7,stroke:#16a34a,color:#14532d

    class KV0L,KV0P kv
    class L0,L1,L2,LD,L7 legacy
    class TILE,R0,R1,RD,R15 packed
```

That trade changes with sequence length. BM128/four waves wins at the selected
short shapes; the existing BM256/eight-wave path remains stronger on long
uniform attention.

### Making fully masked tail rows numerically cheap

A ragged tail can contain packed rows whose query position is outside the
sequence. Their score row is entirely `-inf`. If the running maximum also
starts at `-inf`, online softmax encounters `-inf - (-inf)`, which is NaN.

The generic path repairs that case on every tile. The packed ragged path starts
the maximum at finite `-1e30` instead:

```mermaid
flowchart LR
    subgraph INF["−∞ seed"]
        I0["m_prev = −∞<br/>row_max = −∞"] --> I1["m_new = −∞"]
        I1 --> I2["m_prev − m_new<br/>= NaN"]
        I2 --> I3["Needs invalid-row repair"]
    end

    subgraph FINITE["Finite seed"]
        F0["m_prev = −1e30<br/>row_max = −∞"] --> F1["m_new = −1e30"]
        F1 --> F2["m_prev − m_new = 0<br/>exp2(masked score) = 0"]
        F2 --> F3["No per-tile repair"]
    end

    INF -->|"Change only the empty-row seed"| FINITE

    classDef bad fill:#fee2e2,stroke:#dc2626,color:#7f1d1d
    classDef good fill:#dcfce7,stroke:#16a34a,color:#14532d

    class I0,I1,I2,I3 bad
    class F0,F1,F2,F3 good
```

* masked probabilities still evaluate to zero;
* maximum-difference arithmetic remains finite;
* invalid output rows are still suppressed by store guards; and
* the fast scaled-max softmax can remain in the deep pipeline.

Finite seeding is one component of the combined ragged route, which measured
13.48% higher BF16 and 12.87% higher FP16 throughput than the recorded main
baseline. The campaign did not isolate the seed's contribution. Valid-row
outputs passed the stated reference tolerances.

### Removing masks from aligned window interiors

Sliding attention needs masks at the left and right edges of a query block, but
not for every KV tile between them. When the window, query span, and BN tile are
aligned, the compiler can prove that steady-state interior tiles are fully
inside the window and omit `apply_mask` entirely.

```mermaid
flowchart LR
    L["Left boundary tile<br/>apply window mask"] --> C1["Aligned interior<br/>mask-free"]
    C1 --> C2["Aligned interior<br/>mask-free"] --> CD["…"] --> C3["Aligned interior<br/>mask-free"]
    C3 --> R["Right / causal boundary<br/>apply mask"]

    classDef boundary fill:#fee2e2,stroke:#dc2626,color:#7f1d1d
    classDef clean fill:#dcfce7,stroke:#16a34a,color:#14532d

    class L,R boundary
    class C1,C2,CD,C3 clean
```

The retained window-512 route combines that proof with BN32 and
max-memory-clause scheduling. Those choices were measured together; the data
does not assign the final gain to one component in isolation.

Candidate `986b7271` was compared with merged-main commit `a4178a9d` from
[#1883](https://github.com/lightseekorg/tokenspeed/pull/1883); each retained
row aggregates ten groups of 100 launches:

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

This table reproduces the PR-body before/after evidence for merged main
`a4178a9d` and candidate `986b7271`. It combines separate
correctness-gated CLEAN campaigns rather than one interleaved A/B:

* main S1024/S2048 and ragged values are means of five retained rows;
* candidate S1024 uses three BF16 and five FP16 rows;
* candidate S2048 and ragged values use three rows;
* sliding main is one correctness-gated CLEAN retained row; the candidate is
  the arithmetic mean of three retained rows.

The short/uniform main rows predate deterministic Python hash pinning, so this
table should not be treated as a strict interleaved A/B. Candidate measurements
used pinned hash seeds and isolated caches.

Routing requires H8/KV1, D128, BF16/FP16, and no sinks or LSE. Uniform routing
is exact to B8/S1024 and B4/S2048; sliding routing is exact to uniform
B4/S4096/window-512. The ragged gate is broader: it accepts any nonuniform B4
batch whose maximum sequence length is 4096, although only
`[4096,3584,2305,1024]` is reported here. D64, FP8, other head layouts, sinks,
LSE, other uniform shapes, and other windows retain the established path.

---

## 9. [#1939](https://github.com/lightseekorg/tokenspeed/pull/1939): carry packed ownership into long BF16 attention

[#1924](https://github.com/lightseekorg/tokenspeed/pull/1924) deliberately
limited uniform packed H8/KV1 dispatch to S1024 and S2048. The next direct
screen showed that packing was also useful for the two measured long BF16
routes when paired with the established BM256/BN64 deep pipeline.

[#1939](https://github.com/lightseekorg/tokenspeed/pull/1939) enabled packed
ownership for:

* B4/S4096/H8/KV1/D128 BF16; and
* B2/S8192/H8/KV1/D128 BF16.

The accepted configuration used BM256, BN64, eight waves,
`waves_per_eu=1`, the four-wave TDM producer hint, longest-first query-block
order, and coexec scheduling. It changed the grid from one axis per query head
to one axis per KV head while keeping the deep K-ahead/V-behind schedule.

Fresh direct A/B on rebased commit `d4296da2` used three order-balanced pairs
per workload:

| Workload | Existing route | Packed long route | Gain |
|---|---:|---:|---:|
| B4/S4096/H8/KV1/D128 BF16 | 1,817.521 | 1,908.198 TFLOP/s | +4.99% |
| B2/S8192/H8/KV1/D128 BF16 | 1,954.280 | 2,004.708 TFLOP/s | +2.58% |

All 12 endpoints passed correctness and clock gates. Pairwise gains were
4.94-5.03% at S4096 and 2.38-2.72% at S8192, with maximum grouped-event CV of
0.113%.

The selector stayed narrow. FP16, ragged, sliding, D64, FP8, other head
layouts, sinks/LSE, and unmeasured sequence lengths retained their existing
paths. That last restriction matters: B1/S16384 was not measured and therefore
was not added to the packed selector.

---

## 10. [#1941](https://github.com/lightseekorg/tokenspeed/pull/1941): transport/compute decoupling and instruction-level overlap

The final optimization PR broadened the idea from "pack this shape" to
"specialize each stage of the pipeline independently." It was developed
against changing compiler branches, then rebased and accepted on the released
`tokenspeed-triton==3.9.10.post20261008` package. The merged result is
[TokenSpeed #1941](https://github.com/lightseekorg/tokenspeed/pull/1941),
merge commit `259a410b`.

The retained mechanisms were:

1. BN128 transport followed by two sequential BN64 online-softmax updates;
2. packed four-wave ownership for measured MHA/GQA/MQA tuples;
3. fused K/V TDM issue only on the BF16 H32/KV8 GQA route;
4. QK WMMA scheduling groups containing independent prior-tile VALU work and
   V LDS reads;
5. coexec scheduling for the measured window-512 route; and
6. a compact task list for the measured four-sequence ragged batch.

### 10.1 Wider transport without wider live compute state

A direct BN128 QK tile keeps too much score and softmax state live. It crosses
an allocation cliff even though moving 128 K/V rows at once is attractive.
The retained wide-transport route separates those decisions:

```mermaid
flowchart LR
    H["HBM K and V<br/>128 rows each"] --> T["One BN128 K transfer<br/>+ one BN128 V transfer"]
    T --> L["BN128 K/V in LDS"]
    L --> A["First BN64<br/>QK → softmax → PV"]
    A --> B["Second BN64<br/>QK → softmax → PV"]
    B --> O["One online-softmax<br/>state and output accumulator"]

    classDef memory fill:#dbeafe,stroke:#2563eb,color:#172554
    classDef mover fill:#e0f2fe,stroke:#0284c7,color:#0c4a6e
    classDef compute fill:#ede9fe,stroke:#7c3aed,color:#2e1065
    classDef state fill:#dcfce7,stroke:#16a34a,color:#14532d

    class H,L memory
    class T mover
    class A,B compute
    class O state
```

The transport issues half as often, but each compute step still has BN64 live
state and preserves the established online-softmax update order. This route is
used only for measured uniform packed shapes whose launch and addressing
contracts are rechecked at the host.

The same PR moved the selected long packed BF16 routes from the earlier
eight-wave ownership to a measured four-wave configuration. This is another
example of why one dimension cannot stand in for the whole resource shape:
BM, BN, wave count, transport width, and ownership all affect allocation.

### 10.2 Fill WMMA issue shadows with independent work

Matched ATT showed a structural difference from AITER ASM:

* AITER commonly placed four useful instructions between WMMAs;
* TokenSpeed commonly placed zero or one; and
* TokenSpeed's WMMA stall cost was therefore much higher even when total
  shader-busy time looked healthy.

The retained schedule interleaves groups of four independent VALU
instructions from the previous softmax stage and one V LDS read per eight QK
WMMAs. It does not add a runtime barrier or change attention math. It tells the
machine scheduler which already-independent operations should fill otherwise
empty issue slots.

An earlier matched ATT A/B held the back-to-back-WMMA hardware control enabled
on both sides, isolating the scheduling-group change. Under that controlled
setup, the measured WMMA stall cost fell from 3.92 to 2.47 cycles/hit at
S4096 and from 4.09 to 2.50 at S8192. These are mechanism-attribution numbers,
not absolute stall metrics for the final control-free kernel. The same
scheduling pattern was retained for measured MHA.

An explicit back-to-back-WMMA hardware control was measured separately and was
useful, but it was intentionally removed before merge so this PR would depend
only on the released compiler contract. It is not part of the performance
figures below.

### 10.3 Compact the ragged launch instead of scheduling empty rectangles

The measured ragged batch is:

```text
[4096, 3584, 2305, 1024]
```

With H8/KV1 packed into BM128, one workgroup covers 16 query positions. A
rectangular launch based on S4096 creates:

```text
256 blocks/sequence × 4 sequences = 1024 workgroups
```

The compact mapping launches only the live blocks:

```text
ceil(4096/16) + ceil(3584/16) + ceil(2305/16) + ceil(1024/16)
= 256 + 224 + 145 + 64
= 689 workgroups
```

Each flat task is mapped back to `(sequence, local query block)` inside the
kernel. Query blocks are reversed within each sequence; this is not a global
sort across sequences. A compile-time four-iteration loop loads the start/end
`cu_seqlens` pair for each sequence, so the implementation is restricted to
exactly four sequences. Every otherwise eligible nonuniform B4/max-S4096
batch uses the compact mapping; other batch sizes keep the generic mapping.

### 10.4 Fuse transfers only where producer separation wins

The BF16 H32/KV8 wide GQA route issues K and V through one fused TDM operation
with disjoint producer-wave subsets:

```text
waves 0-1 -> K
waves 2-3 -> V
cache policy -> .cg
```

Direct GQA A/B pairs were positive and the generated kernel remained
zero-scratch. The same mechanism regressed MQA, so it is not generalized.
Full-loop fusion on the H8/KV1 path also regressed because it coupled K
readiness to V completion; it was rejected.

For B4/S4096/window-512, coexec replaced max-memory-clause only on the exact
measured route. Sliding attention remains a separate specialization because
its mask and issue pattern differ from full causal attention.

### 10.5 Fast-path contracts are part of the optimization

The final host launcher explicitly checks:

* compact ragged requires exactly four sequences;
* wide transport also requires packed selection, uniform rows, no
  sinks/LSE/window, ordinary 32-bit local addressing, and the expected grid;
* D64, FP8, sinks/LSE, wide addresses, unsupported head tuples, and unmeasured
  shapes use established fallbacks; and
* S2048 keeps its existing packed route because the new variant was neutral.

Simulator-sized runtime tests cover compact and non-compact ragged launches,
wide and fused-wide transport, QK interleaving, FP8, sinks/LSE, and launch
guards. The wide-address path rebases sequence/head pointers with 64-bit
arithmetic while preserving 32-bit local buffer offsets.

### 10.6 Final #1941 gain on the released compiler

The final acceptance summary is composite. Routes affected by removing the
back-to-back-WMMA control were freshly rerun at candidate `8111c787`; routes
whose selectors never enabled that control reuse the matching release campaign
at candidate `ec24b15c`. Both campaigns used TokenSpeed main `c50a2f09`, AITER
`b68b0e5c7`, and `tokenspeed-triton==3.9.10.post20261008`. Each used three
cyclic orders, seeded inputs, FP32 correctness gates, grouped device-event
timing, exclusive GPU locking, and retained clock telemetry.

Changed BF16 routes:

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

All 24 BF16 candidate-versus-main pairs were positive.

Changed FP16 routes:

| Workload | Main | #1941 candidate | Gain |
|---|---:|---:|---:|
| B8/S1024 H8/KV1 | 984.686 | 1,088.932 | +10.59% |
| Ragged `[4096,3584,2305,1024]` | 1,053.109 | 1,255.265 | +19.20% |
| Sliding B4/S4096/window-512 | 1,124.550 | 1,219.566 | +8.45% |

Other FP16 routes were intentionally unchanged. Maximum grouped-event CV
across the final reruns was 0.908%.

---

## 11. The post-merge baseline: where the compiler-generated kernel now stands

After [#1941](https://github.com/lightseekorg/tokenspeed/pull/1941) merged, we
rebuilt the comparison from TokenSpeed `676acaa5`, the released
`3.9.10.post20261008` compiler, and AITER `b68b0e5c`. This is a new campaign,
so its absolute values should not be compared with the #1941 main/candidate
table above.

Four order-balanced pairs per row passed FP32 correctness, grouped-event CV,
and clock gates:

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

S1024 is a clear result: every pair favors TokenSpeed, with pairwise deltas
from +2.425% to +6.785%. S4096 is statistical parity rather than a robust win:
its pair range is -0.873% to +1.644%. GQA and MQA are within 2% of hand-written
ASM.

S16384 was added after the first post-merge roundup. It currently uses the
generic eight-wave grid rather than the packed specialization, and all four
pairs consistently trail ASM by 13.22-13.54%. It is now tracked as a
long-context coverage and optimization target.

Finite-window sliding attention is absent from this table because AITER ASM is
ineligible; TokenSpeed's measured sliding route instead competes with AITER
FlyDSL.

The final result is narrower—and more useful—than saying that one
compiler-generated kernel "beats ASM." TokenSpeed now beats ASM on the short
H8/KV1 route, is statistically mixed at S4096 and within 2% on both H32 head
layouts, supports features outside the ASM contract, and has clearly measured
remaining work at S2048, S16384, and ragged attention.

---

## 12. Evidence map

Optimization-workstream PRs:

* [#1332 — widen the M tile](https://github.com/lightseekorg/tokenspeed/pull/1332)
* [#1337 — cover both tile shapes](https://github.com/lightseekorg/tokenspeed/pull/1337)
* [#1357 — hardcode the occupancy cutoff](https://github.com/lightseekorg/tokenspeed/pull/1357)
* [#1476 — max-ILP scheduling](https://github.com/lightseekorg/tokenspeed/pull/1476)
* [#1496 — TDM producer warps](https://github.com/lightseekorg/tokenspeed/pull/1496)
* [#1505 — longest-first causal order](https://github.com/lightseekorg/tokenspeed/pull/1505)
* [#1883 — deep pipeline and full-tile specialization](https://github.com/lightseekorg/tokenspeed/pull/1883)
* [#1924 — packed GQA and window/ragged paths](https://github.com/lightseekorg/tokenspeed/pull/1924)
* [#1939 — long BF16 packed GQA](https://github.com/lightseekorg/tokenspeed/pull/1939)
* [#1941 — packed MHA prefill tuning](https://github.com/lightseekorg/tokenspeed/pull/1941)

Other direct kernel-lineage PRs:

* [#718 — staging introduction](https://github.com/lightseekorg/tokenspeed/pull/718)
* [#719 — main landing](https://github.com/lightseekorg/tokenspeed/pull/719)
* [#880 — generation-based organization](https://github.com/lightseekorg/tokenspeed/pull/880)
* [#953 — FP8 support](https://github.com/lightseekorg/tokenspeed/pull/953)
* [#1411 — CDNA5 namespace](https://github.com/lightseekorg/tokenspeed/pull/1411)
* [#1678 — registered-symbol naming](https://github.com/lightseekorg/tokenspeed/pull/1678)

Test/CI companion:

* [#1522 — vendor test-path split](https://github.com/lightseekorg/tokenspeed/pull/1522)

Toolchain provenance:

* The three-policy replay used one source snapshot and
  `tokenspeed_triton 3.8.10`; the harness did not persist that snapshot's SHA,
  so only within-replay deltas are claimed.
* The public roofline campaign used TokenSpeed `5eea3c0ea9` and
  `tokenspeed-triton==3.8.10.post20260906`.
* Public TokenSpeed `fa9b11054d` was the
  [#1883](https://github.com/lightseekorg/tokenspeed/pull/1883) development
  base.
* Public [#1883](https://github.com/lightseekorg/tokenspeed/pull/1883) and the
  packed-GQA campaign used `tokenspeed-triton==3.8.10.post20260920`.
* [#1939](https://github.com/lightseekorg/tokenspeed/pull/1939) used
  `tokenspeed-triton==3.8.10.post20260920` and merged as `e88f5558`.
* Final [#1941](https://github.com/lightseekorg/tokenspeed/pull/1941)
  acceptance used main `c50a2f09`, fresh control-free candidate `8111c787`,
  unaffected release rows from candidate `ec24b15c`, and
  `tokenspeed-triton==3.9.10.post20261008`; it merged as `259a410b`.
* The post-merge TokenSpeed/AITER baseline used TokenSpeed `676acaa5`, the
  same released compiler, and AITER `b68b0e5c`.
* [#1410](https://github.com/lightseekorg/tokenspeed/pull/1410) moved
  TokenSpeed to `post20260906`, and
  [#1656](https://github.com/lightseekorg/tokenspeed/pull/1656) moved it to
  `post20260920`; neither changed this kernel.
