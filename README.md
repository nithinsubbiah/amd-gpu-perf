# AMD GPU Performance Blog Drafts

Technical articles about AMD GPU performance, compiler behavior, and kernel
optimization. Each article is developed on its own branch so its prose,
benchmark artifacts, and review history remain independent. Reviewed drafts
are squash-merged into `main`, which acts as the collected index.

## Articles

| Draft | Development branch | Review |
|---|---|---|
| [Tuning PyTorch FlexAttention for AMD Instinct MI350X](blog/flexattention-mi350x/README.md) | `blog/flexattention-mi350x` | [PR #3](https://github.com/nithinsubbiah/amd-gpu-perf/pull/3) |
| [Engineering a Faster FlashAttention Prefill Kernel on gfx1250](blog/gfx1250-mha-prefill/README.md) | `blog/gfx1250-mha-prefill` | [PR #2](https://github.com/nithinsubbiah/amd-gpu-perf/pull/2) |

## Branch model

`main` contains the collected drafts. New work starts from `main` on a branch
named `blog/<slug>`. That branch owns one article and all material needed to
review or reproduce it.

```text
main
  README.md
  .gitignore
  blog/<slug>/...

blog/<slug>
  blog/<slug>/README.md
  blog/<slug>/images/       # optional
  blog/<slug>/bench/        # optional
  blog/<slug>/data/         # optional
```

Draft PRs provide line-level review without coupling unrelated articles. Each
branch should be based directly on `main`, should not depend on another draft
branch, and is squash-merged when ready. The named branch can remain as the
article's focused development history.

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
