Input: 3 complete process runs; grad mode: enabled

### bf16 pipelining (1 -> 2 stages, 4 wavefronts)

| Sequence length | Head dim | Mask | Before | After | Speedup | Median paired ratio [min, max] |
|---|---|---|---|---|---|---|
| 4096 | 64 | none | 442.9 | 519.4 | 1.17x | 1.173x [1.171, 1.186] |
| 4096 | 64 | causal | 274.2 | 315.2 | 1.15x | 1.150x [1.144, 1.152] |
| 4096 | 128 | none | 540.9 | 637.3 | 1.18x | 1.179x [1.156, 1.180] |
| 4096 | 128 | causal | 372.0 | 411.0 | 1.10x | 1.102x [1.101, 1.111] |
| 8192 | 128 | none | 576.2 | 660.9 | 1.15x | 1.147x [1.146, 1.147] |
| 8192 | 128 | causal | 430.1 | 480.3 | 1.12x | 1.117x [1.083, 1.127] |

### fp16 wavefronts (8 -> 4, 2 stages)

| Sequence length | Head dim | Mask | Before | After | Speedup | Median paired ratio [min, max] |
|---|---|---|---|---|---|---|
| 4096 | 64 | none | 231.9 | 504.2 | 2.17x | 2.174x [2.174, 2.177] |
| 4096 | 64 | causal | 119.1 | 312.8 | 2.63x | 2.625x [2.624, 2.629] |
| 4096 | 128 | none | 327.3 | 609.3 | 1.86x | 1.859x [1.856, 1.868] |
| 4096 | 128 | causal | 187.1 | 401.6 | 2.15x | 2.146x [2.138, 2.149] |
| 8192 | 128 | none | 333.5 | 631.9 | 1.90x | 1.895x [1.894, 1.896] |
| 8192 | 128 | causal | 227.9 | 463.2 | 2.03x | 2.031x [2.021, 2.098] |

### fp16 combined (1 stage / 8 waves -> 2 stages / 4 waves)

| Sequence length | Head dim | Mask | Before | After | Speedup | Median paired ratio [min, max] |
|---|---|---|---|---|---|---|
| 4096 | 64 | none | 237.9 | 504.2 | 2.12x | 2.117x [2.116, 2.125] |
| 4096 | 64 | causal | 102.7 | 312.8 | 3.04x | 3.045x [3.044, 3.049] |
| 4096 | 128 | none | 277.6 | 609.3 | 2.20x | 2.196x [2.188, 2.201] |
| 4096 | 128 | causal | 154.5 | 401.6 | 2.60x | 2.600x [2.595, 2.610] |
| 8192 | 128 | none | 285.5 | 631.9 | 2.21x | 2.214x [2.212, 2.222] |
| 8192 | 128 | causal | 193.1 | 463.2 | 2.40x | 2.400x [2.397, 2.402] |

