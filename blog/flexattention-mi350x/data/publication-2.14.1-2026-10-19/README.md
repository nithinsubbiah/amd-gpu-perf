# PyTorch 2.14.1 publication validation

The GPU campaign resumed on 2026-10-20 after verifying access to `/dev/kfd`
and `/dev/dri`, a working GPU kernel, the physical mapping of selected ordinal 6
to PCI `0000:e5:00.0`, and three consecutive idle observations with no competing
process. All twelve unforced release-default and reference-correctness checks
passed, followed by the four-configuration smoke check. The first full process
aborted after two accepted measurements because the next timing window contained
only one of the required two loaded-clock observations. This is an incomplete,
rejected campaign; all of its artifacts remain in place.

The [resume archive](history/20261020T001809Z/) preserves the prior campaign
files, including the original failed preflight, plus a SHA256 manifest of all
preexisting article files and the new GPU access/idle evidence. The launcher log
is `publication-20261020T001809Z.log`. The article's existing figures retain their
original development-build attribution.

The [failure analysis](failure-analysis.json) records utilization readings of
100%, 88%, and 46% during that window, no competing process, and a 0.26% trial
latency range. A [replacement campaign](../publication-2.14.1-2026-10-20/README.md)
uses six timing trials per point to extend monitoring coverage. Correctness,
isolation, loaded-clock, and timing-repeatability thresholds remain unchanged.
No measurements from this rejected attempt enter the replacement aggregate.
The replacement completed all 108 measurements and 648 timing trials and passed
its numerical, monitoring, source, and compiler-artifact audits.

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

## Resume the GPU campaign

The execution context must expose `/dev/kfd` and `/dev/dri` and permit GPU
process monitoring. Run the complete forward campaign with:

```bash
/home/nmeganat/flex/.venv-blog-2.14.1-rocm7.2/bin/python \
  /home/nmeganat/flex/blog-editorial/blog/flexattention-mi350x/data/publication-2.14.1-2026-10-19/run_publication.py \
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
and investigate the cause before starting a replacement campaign. The original
preflight failed before any GPU-run artifact was created; it is preserved in the
resume archive. The current attempt now has GPU-run artifacts, so rerunning the
top-level command would encounter existing output names.

## Remaining publication work

1. GPU measurements and compiler inspection are complete in the linked replacement
   campaign. Integrate its accepted results and revised static counts when
   updating the article's tables and diagrams.
2. Validate the addressing section on a separately pinned upstream build.
3. Regenerate the article's tables and charts from accepted measurements and
   update their software/protocol attribution.

CPU status: all twelve harness validation tests pass, dependency checks pass,
and Python lint passes. These checks do not substitute for GPU validation.
