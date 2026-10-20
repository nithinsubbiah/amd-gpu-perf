# Resolving the one-stage, eight-wavefront baseline discrepancy

**The blog's baseline reproduces.** The corrected harness measured 240.2,
239.2, and 238.2 TFLOP/s in three fresh processes, a median of **239.2**, versus
the historical **239.1**. The earlier approximately 201 TFLOP/s result came
from the validation harness adding `torch.no_grad()`.

## Controlled experiments

Every run uses an idle MI350X GPU 6, with physical PCI identity and KFD process
isolation checked. Inputs are identical seeded fp16 tensors at
`B=1, H=16, S=4096, D=64`, without a mask. Inputs do not require gradients;
only forward execution is timed.

1. **Original script, original grad context:** restrict the sweep to this
   workload and retain all four configurations. Preserve the original timing
   loop and clock-ramp placement. Fresh compiler caches give **240.8 TFLOP/s**
   for the one-stage/eight-wavefront case.
2. **Original script with grad disabled:** change only the process grad mode
   and use another fresh cache. The baseline falls to **200.1 TFLOP/s**.
3. **Initial revised harness, fresh caches:** the original validation's
   grad-disabled execution gives **200.1 TFLOP/s** again. Its baseline has
   the same Triton compilation key as experiment 2.
4. **Corrected revised harness, three fresh processes:** restore the original
   grad-enabled context while retaining the stricter warmup and monitoring.
   The baseline returns to **239.2 TFLOP/s** at the process median. All three
   baseline kernels have the same compilation key as experiment 1.

The original script is byte-identical to the benchmark at blog commit
`e2f2049b24fd891b756c78cde4cb5b7298f54065`. The diagnostic changes only the
shape/mask/dtype sweep and intercepts `do_bench` returns to preserve all five
trial medians per configuration. It does not alter the original timing loop.
Background monitoring is active in all reported runs. The original script's
broad process warning includes its own PID; the enclosing monitor verifies
that no other process is mapped to the selected GPU.

## Why grad mode changes this forward kernel

The [PyTorch frontend](https://github.com/nithinsubbiah/pytorch/blob/37200109b999958ae0042b21b2da3eea48ffec4a/torch/nn/attention/flex_attention.py#L2163)
sets `OUTPUT_LOGSUMEXP` from `torch.is_grad_enabled()` when the caller does not
request auxiliary outputs. This applies even when Q/K/V do not require grads,
because captured buffers can require gradients.

[The generated-code diff](grad-mode-generated.diff) changes
`OUTPUT_LOGSUMEXP=True` to `False`. The compiler subsequently allocates:

| Execution context | Logsumexp output | VGPRs | Assembly-reported occupancy | Original-loop throughput |
|---|---|---:|---:|---:|
| Grad enabled | true | 128 | 4 | 240.8 TFLOP/s |
| Grad disabled | false | 130 | 3 | 200.1 TFLOP/s |

Both variants use eight wavefronts, one stage, 16,384 bytes of shared memory,
and no register spills. Disabling one output changes code generation and
crosses a register-occupancy threshold. The occupancy change is consistent
with the measured slowdown; these experiments do not isolate its contribution
from every other instruction-scheduling change.

Both execution contexts are valid. A comparison against the historical blog
must use its original grad-enabled context.

## Corrected configuration comparison

Medians across three fresh corrected-harness processes:

| Stages | Wavefronts | Blog TFLOP/s | Recheck TFLOP/s |
|---|---|---:|---:|
| 1 | 8 | 239.1 | 239.2 |
| 2 | 8 | 233.9 | 232.0 |
| 2 | 4 | 508.0 | 506.2 |
| 1 | 4 | not retained | 428.5 |

The combined gain is **2.12x**, matching the original blog's 2.12x. Pipelining
with eight wavefronts remains slightly negative in this execution context.
All 12 corrected measurements pass numerical and monitoring checks; each
retains five timing-trial medians, generated source, and compressed assembly.
These targeted four-configuration runs do not replace the earlier full
36-configuration campaign.

## Retained artifacts and replay

- `original-fresh-monitored.json`, `original-fresh-nograd.json`: original-loop
  trials, process observations, compiler metadata, and assembly paths.
- `current-fresh.jsonl`: initial revised harness with fresh caches and grad
  disabled.
- `current-matched-grad{1,2,3}.jsonl`: corrected, grad-enabled rechecks.
- [summary.json](summary.json): process medians and compilation-key check.
- [grad-enabled-kernel.py.gz](grad-enabled-kernel.py.gz) and
  [grad-disabled-kernel.py.gz](grad-disabled-kernel.py.gz): generated kernels.
- `source/`: compressed copies of the original benchmark and measured corrected
  harness. Compiler caches are ignored by Git; assembly exports are retained.

From the article directory, audit the retained matched runs without GPU work:

```bash
python data/validation-2026-10-19/baseline-recheck/analyze.py
```

The harness now exposes `--grad-mode enabled|disabled`, defaults to enabled,
and records the actual grad mode and generated logsumexp flag. Aggregation
checks these fields when present. The initial raw runs remain unchanged and
are explicitly identified as grad-disabled measurements in the parent report.
