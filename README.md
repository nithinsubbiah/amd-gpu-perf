# AMD GPU Performance Blog Drafts

Technical articles about AMD GPU performance, compiler behavior, and kernel
optimization. Each draft lives on its own branch so its prose, benchmark
artifacts, and review history remain independent.

## Active drafts

| Draft | Branch | Review |
|---|---|---|
| [Tuning PyTorch FlexAttention for AMD Instinct MI350X](https://github.com/nithinsubbiah/amd-gpu-perf/blob/blog/flexattention-mi350x/blog/flexattention-mi350x/README.md) | `blog/flexattention-mi350x` | [PR #3](https://github.com/nithinsubbiah/amd-gpu-perf/pull/3) |
| [Engineering a Faster FlashAttention Prefill Kernel on gfx1250](https://github.com/nithinsubbiah/amd-gpu-perf/blob/blog/gfx1250-mha-prefill/blog/gfx1250-mha-prefill/README.md) | `blog/gfx1250-mha-prefill` | [PR #2](https://github.com/nithinsubbiah/amd-gpu-perf/pull/2) |

## Branch model

`main` is intentionally neutral. A branch named `blog/<slug>` owns one article
and all material needed to review or reproduce it.

```text
main
  README.md
  .gitignore

blog/<slug>
  blog/<slug>/README.md
  blog/<slug>/images/       # optional
  blog/<slug>/bench/        # optional
  blog/<slug>/data/         # optional
```

Draft PRs provide line-level review without coupling unrelated articles. A
draft branch should be based directly on `main` and should not depend on
another draft branch.

## Evidence standards

Performance claims should include:

* exact hardware, source, and compiler revisions;
* workload shape, dtype, and FLOP accounting;
* correctness gates before timing;
* GPU exclusivity and clock-stability checks;
* control/candidate ordering and retained sample count; and
* explicit caveats when rows come from separate campaigns.

Static codegen improvements are useful evidence, but hardware timing remains
the performance acceptance gate.
