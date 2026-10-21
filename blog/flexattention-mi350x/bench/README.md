# Reproducing the FlexAttention comparisons

Run these scripts from the article directory using Python 3.12 and a ROCm
PyTorch environment. Plotting needs matplotlib. The harness targets gfx950,
uses the default Triton template, and pins every compared configuration.

## Publication release environment

The article uses the [official PyTorch 2.14.1 ROCm campaign](../data/publication-2.14.1-2026-10-20/README.md):
PyTorch 2.14.1+rocm7.2, triton-rocm 3.8.0 and ROCm 7.2.1. Installation reports
retain the wheel URLs and SHA256 hashes. The installed sources contain the
gfx950 `(128, 64, 2, 4)` forward defaults for fp16 and bf16 at head dimensions
64 and 128. All twelve unforced-default checks and three complete
36-configuration processes passed, retaining 648 timing trials.

On CPython 3.12, Linux x86_64 with ROCm 7.2.1, create a fresh environment from
the article directory:

```bash
python3.12 -m venv .venv-publication
source .venv-publication/bin/activate
python -m pip install --require-hashes -r data/publication-2.14.1-2026-10-20/environment/requirements.lock
```

This installs the exact wheel artifacts used in the campaign. Keep old and new
configurations on this same build. The harness records the PyTorch source SHA,
HIP version, Triton distribution, GPU identity, relevant environment overrides,
active template hashes and protocol settings.

The campaign uses six trials per configuration to extend monitoring coverage;
pass `--reps 6` as below (the generic harness default remains three). Grad mode
is enabled by default. Inputs do not require gradients and only forward is
timed, but this context still emits logsumexp. `--grad-mode disabled` measures
a separate execution context and must be applied consistently across a campaign.

## Timing protocol

Use an exclusively allocated MI350X. Set `HIP_VISIBLE_DEVICES` to exactly one
GPU, or pass `--device N`. The harness resolves logical device 0 to its PCI bus address using HIP,
then uses that address to identify the device in `rocm-smi`; it does not assume
that HIP and SMI device indices are interchangeable. ROCm 7.2 `rocm-smi` and
read access to the GPU process and clock information are required.

The publication harness applies the following checks:

1. Compile each configuration and check its output against an eager
   `flex_attention` reference before timing. Require finite values and
   `torch.testing.assert_close`, with `(atol, rtol)=(0.003, 0.01)` for fp16 and
   `(0.02, 0.02)` for bf16. These tolerances apply to this fixed random-input
   sweep; other score modifiers and input distributions need their own checks.
2. After the correctness check, require three consecutive 0% utilization
   readings with no competing KFD processes on the selected physical GPU.
   Then ramp the GPU with eight seconds of dense matmul and warm up the actual
   attention configuration for three seconds, allowing its operating frequency
   to settle before timing. This issues ordinary GPU work and changes no clock
   or power settings. Both warmup durations are recorded in the run metadata.
3. Take six `do_bench` medians (`--reps 6`) per configuration, each with 200 ms of warmup
   and 500 ms of repetitions. Preserve every trial latency and the resulting
   median. The sweep covers 36 configurations, including all four fp16
   combinations of stage count and wavefront count.
4. Check for competing processes before each configuration. During timing,
   poll the selected GPU's process mapping, utilization, and SCLK, retaining the observations.
   Abort for observed competition or unavailable monitoring. Validate clocks
   using at least two observations inside the timing window with reported GPU
   utilization of at least 90%. Reject loaded SCLK below 1000 MHz and record
   its range and median. Dynamic frequency scaling can change instantaneous
   SCLK even when repeated kernel timings are stable; the default protocol
   therefore rejects a timing range above 5% of the median rather than imposing
   a SCLK range limit. An optional `--max-clock-spread-pct` enforces a stricter
   clock range. Before/after snapshots remain in the record
   for isolation checks; their idle clocks are excluded from the loaded-clock
   test. The floor and range limits are
   explicit command-line settings; keep them identical across a campaign.
5. Run three fresh processes, reversing configuration order in the second.
   Preserve all three JSONL files. Any failed correctness, monitoring, compile,
   or timing check aborts the process and leaves an incomplete run that the
   aggregator rejects.

Monitoring polls every 0.5 seconds between queries, plus checks at the start
and end of timing. It can miss brief interference and does not reserve a GPU;
an exclusive allocation is the primary isolation mechanism. SMI subprocess
queries run concurrently on the CPU and enqueue no GPU kernels, with the same
monitoring protocol for every configuration. Utilization is a reported sensor
value, so load qualification is an observation rather than an exact
reconstruction of GPU event intervals.

## Run and retain a campaign

From the article directory in Bash, with a new `results/` directory and the
release environment activated:

```bash
mkdir -p results
for run in 1 2 3; do
  order=()
  if [ "$run" = 2 ]; then order=(--reverse-order); fi
  HIP_VISIBLE_DEVICES=0 \
    TORCHINDUCTOR_CACHE_DIR="$PWD/results/cache-run$run/inductor" \
    TRITON_CACHE_DIR="$PWD/results/cache-run$run/triton" \
    python bench/bench_landed.py --reps 6 "${order[@]}" \
      --output "results/run$run.jsonl" || break
done
python bench/aggregate_landed.py 'results/run*.jsonl'
python bench/plot_landed.py 'results/run*.jsonl' results/figures
```

An output file must be new; the harness never overwrites an existing run.
The aggregator requires at least three complete runs with matching hardware,
software, protocol settings, and benchmark/monitor source hashes. It rejects
missing or duplicate configurations, inconsistent timing/throughput records,
and missing or failed isolation/clock evidence. Active FlexAttention Python
and template sources are also hashed in the environment record. Configuration order
may differ. It prints the ratio of median throughputs used by the article,
plus the median and observed range of paired process ratios for a fresh
campaign. Observed ranges are descriptive, not confidence intervals.
Process-private cache directory paths may differ between runs; all other
recorded compiler and backend environment overrides must match.

Invoking `aggregate_landed.py` or `plot_landed.py` without an input reads the
official PyTorch 2.14.1 summary CSV; the complete process samples are linked
from [the data index](../data/README.md). Rendering those charts uses no GPU.
Run `plot_mechanisms.py` to regenerate the two explanatory diagrams.

For a short hardware check, use `python bench/bench_landed.py --device N
--smoke --reps 6 --output results/smoke.jsonl`. This measures the first fp16 workload
with all four configurations. Its `smoke_complete` marker prevents it from
being aggregated as a full campaign. Generated kernel code is retained as
compressed Python source beside each JSONL run; the harness verifies that its
compiler stage and wavefront settings match the requested values.

Run the CPU validation checks with
`python -m unittest discover -s bench -p test_validation.py`.

## Separate memory-addressing ablation

`bench_addressing.py` reuses the same correctness, idle-device, warmup, and
timing checks for fp16 `B=1, H=16, S=4096, D=128`, with `(128, 64, 2, 4)` fixed.
It controls `TORCHINDUCTOR_EMIT_POINTER_RANGE_32` in fresh processes with separate
compiler caches. The two cases are unmasked and causal attention.

This ablation requires the template flag handling from
[pytorch#194136](https://github.com/pytorch/pytorch/pull/194136). Apply that change
to the historical source setup, or use a build containing it. The driver rejects
a disabled flag that still produces annotated template arguments.

For example:

```bash
python bench/bench_addressing.py --device 0 --pointer-range on --output results/address-on.jsonl
python bench/bench_addressing.py --device 0 --pointer-range off --output results/address-off.jsonl
```

Repeat the pair in three fresh processes per setting, alternating which setting
runs first. Inspect the generated annotation and compressed AMD assembly in
each output's artifact directory. The driver requires annotated kernels to use
buffer loads and unannotated kernels to switch to global loads; changing the
flag alone does not establish that the machine code changed. `ablation.json`
records instruction counts, artifact hashes, and the wrapper's source hash.
These two-case runs are separate from the 36-configuration campaign and cannot
be passed to `aggregate_landed.py`.

## Historical development environment

The earlier development campaign used ROCm 7.2.1, PyTorch
`2.14.0a0+gitb5460ee`, and Triton 3.8.0. See the
[environment record](../data/environment.json) for the source identifiers and
[summary provenance](../data/README.md) for the limits of the retained data.

To obtain the recorded PyTorch source in a fresh directory:

```bash
git clone https://github.com/nithinsubbiah/pytorch.git pytorch-flex
cd pytorch-flex
git fetch origin b5460ee3cad1ea1a9a175a3f95db31930d301eca
git checkout --detach b5460ee3cad1ea1a9a175a3f95db31930d301eca
git submodule sync --recursive
git submodule update --init --recursive
```

Use a separate virtual environment and the build dependencies described in
[PyTorch's source-build instructions](https://github.com/pytorch/pytorch#from-source).
The local ROCm build used the following configuration; run it from the PyTorch
source directory after installing those dependencies:

```bash
export ROCM_PATH=/opt/rocm-7.2.1
export HIP_PATH="$ROCM_PATH"
export PATH="$ROCM_PATH/bin:$PATH"
export CPLUS_INCLUDE_PATH="$ROCM_PATH/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
export C_INCLUDE_PATH="$ROCM_PATH/include${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}"
export PYTORCH_ROCM_ARCH=gfx950
export USE_ROCM=1 USE_CUDA=0
export USE_FLASH_ATTENTION=0 USE_MEM_EFF_ATTENTION=0 USE_ROCM_CK_SDPA=0
export USE_ROCM_CK_GEMM=0 USE_MSLK=0 USE_FBGEMM=0 USE_DISTRIBUTED=0
export BUILD_TEST=0 USE_KINETO=0
python tools/amd_build/build_amd.py
python -m pip install -e . --no-build-isolation -v
```

The historical logs did not retain a full Triton SHA. The recovered local
installation identifies `3.8.0+gitc8305ccc`, corresponding to
`c8305cccdeb7c2eb07441a5e696e7f9b17cd8e24` in
[triton-lang/triton](https://github.com/triton-lang/triton). To build that
identified compiler in the same virtual environment:

```bash
git clone https://github.com/triton-lang/triton.git triton-flex
cd triton-flex
git checkout --detach c8305cccdeb7c2eb07441a5e696e7f9b17cd8e24
python -m pip install -r python/requirements.txt
python -m pip install -e . --no-build-isolation
```

This provides a concrete source setup for a new campaign, without claiming
bit-for-bit reconstruction of the original environment. A newer ROCm PyTorch
build containing the upstream configuration changes can also measure the same
comparison. Keep all configurations on the same build and record its identity.
The current article uses the official release environment above.
