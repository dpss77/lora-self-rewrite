# ADR-0002: `enable_prefix_caching=False` always

- Status: accepted
- Date: 2026-11-26

## Context

`stage1_02` showed vLLM automatic prefix caching leaked base KV state into a
subsequent `LoRARequest`, contaminating LoRA outputs. The preflight now checks
`deterministic_repeat` and `base_between_stable`.

## Decision

Disable vLLM prefix caching in every engine, including the held-out evaluator.
Do not re-enable it for a speed win. If editor-prefill speed becomes a
bottleneck, investigate an adapter-aware cache or explicit reset API.

## Consequences

- Lower throughput and higher cost than a cached engine.
- Scientific results are not contaminated by cross-request KV reuse.
- Any proposal to re-enable it is a protocol violation until a new isolation
  proof exists.
