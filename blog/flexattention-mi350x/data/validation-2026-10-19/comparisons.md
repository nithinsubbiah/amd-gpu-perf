**Execution context:** these retained full-campaign results use grad mode disabled. The historical blog used grad mode enabled; see [the corrected baseline recheck](baseline-recheck/README.md).

Input: 3 complete process runs

### bf16 pipelining (1 -> 2 stages, 4 wavefronts)

| Sequence length | Head dim | Mask | Before | After | Speedup | Median paired ratio [min, max] |
|---|---|---|---|---|---|---|
| 4096 | 64 | none | 443.9 | 518.5 | 1.17x | 1.167x [1.166, 1.172] |
| 4096 | 64 | causal | 274.5 | 314.4 | 1.15x | 1.140x [1.138, 1.145] |
| 4096 | 128 | none | 541.6 | 638.4 | 1.18x | 1.179x [1.168, 1.182] |
| 4096 | 128 | causal | 374.1 | 412.2 | 1.10x | 1.102x [1.098, 1.106] |
| 8192 | 128 | none | 574.2 | 657.2 | 1.14x | 1.146x [1.144, 1.146] |
| 8192 | 128 | causal | 427.9 | 480.8 | 1.12x | 1.116x [1.077, 1.124] |

### fp16 wavefronts (8 -> 4, 2 stages)

| Sequence length | Head dim | Mask | Before | After | Speedup | Median paired ratio [min, max] |
|---|---|---|---|---|---|---|
| 4096 | 64 | none | 232.3 | 504.1 | 2.17x | 2.170x [2.160, 2.174] |
| 4096 | 64 | causal | 119.4 | 311.1 | 2.61x | 2.606x [2.594, 2.613] |
| 4096 | 128 | none | 325.1 | 607.8 | 1.87x | 1.869x [1.861, 1.872] |
| 4096 | 128 | causal | 186.7 | 403.0 | 2.16x | 2.158x [2.151, 2.169] |
| 8192 | 128 | none | 331.5 | 627.7 | 1.89x | 1.893x [1.893, 1.895] |
| 8192 | 128 | causal | 224.2 | 464.1 | 2.07x | 2.068x [2.036, 2.071] |

### fp16 combined (1 stage / 8 waves -> 2 stages / 4 waves)

| Sequence length | Head dim | Mask | Before | After | Speedup | Median paired ratio [min, max] |
|---|---|---|---|---|---|---|
| 4096 | 64 | none | 201.3 | 504.1 | 2.50x | 2.504x [2.501, 2.516] |
| 4096 | 64 | causal | 103.2 | 311.1 | 3.01x | 3.013x [3.003, 3.023] |
| 4096 | 128 | none | 276.9 | 607.8 | 2.20x | 2.195x [2.188, 2.198] |
| 4096 | 128 | causal | 154.6 | 403.0 | 2.61x | 2.606x [2.596, 2.611] |
| 8192 | 128 | none | 284.9 | 627.7 | 2.20x | 2.203x [2.201, 2.207] |
| 8192 | 128 | causal | 193.7 | 464.1 | 2.40x | 2.396x [2.380, 2.399] |

