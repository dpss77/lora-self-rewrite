# ADR-0001: H200-only for cacheable runs and held-out evaluation

- Status: accepted
- Date: 2026-11-26

## Context

vLLM `bfloat16` kernels are not batch-size invariant, and the hardware/kernel
`cache_tag` separates H200 and H100 caches. The donor allowed an `H200,H100`
fallback; an H100 fallback made the first DeepScaleR cache tag unusable.

## Decision

Pin `gpu="H200"` for every cacheable experiment and for `eval_heldout.py`. Do not
mix GPU classes across shards or across a base/edited comparison.

## Consequences

- Cache signatures are valid only within one hardware/kernel tag.
- An H200 capacity shortage blocks a run; it does not silently degrade it.
- Non-cache-critical debugging may use another GPU only outside the scientific
  path and must be labeled as such.
