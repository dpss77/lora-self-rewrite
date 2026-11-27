# Changelog

## 0.1.0 — 2026-11-27

Initial public PoC snapshot.

- Rank-4 zero-initialized LoRA workspace over 128 text-backbone projection matrices.
- H200 vLLM engine with prefix caching disabled, exact-batch caches, and spawn-isolated
  activation reads.
- Second-Protocol session loop, resume/preemption handling, adapter validation, and held-out gate.
- Curated S01 (`math500`, null validation) and S02 (`DeepScaleR seed0`, withdrawn) records.
- Layered CPU test suite, contract checks, JSON Schemas, and CI workflow.
- CPU/Post-refactor H200 smoke is still pending; see `docs/04-operations/READINESS.md`.

