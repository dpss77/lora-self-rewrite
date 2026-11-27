# Start here

**Project:** zero-initialized rank-4 LoRA workspace adaptation of
*"Language Models Can Rewrite Their Own Weights"* (`arXiv:2611.21885v2`),
targeting `Qwen/Qwen3.5-27B` on a single H200.

## Current status

- **Stage 1 (`math500`) closed:** submitted edit, validation null
  (`+2.3% ± 3.9%`, one-sided `p=0.347`).
- **Stage 2 (`DeepScaleR seed0`) closed:** `WITHDRAW` after 20/20 resumed passes;
  no submitted adapter.
- **No compute budget is currently funded.** See
  `04-operations/COMPUTE_BUDGET.md`; do not rerun `ds_seed0`.
- This snapshot implements the CPU-only protocol layer, output contract,
  automation, tests, and a held-out evaluator (unrun).

## Map

| Area | Read |
| :--- | :--- |
| Source of truth / read order | `SOURCE-OF-TRUTH.md` |
| Paper reference (read-only) | `01-context/PAPER_SPEC_2611.21885v2.md` |
| Concept & closed milestones | `01-context/CONTEXT.md` |
| Module boundaries | `02-architecture/MODULE_MAP.md` |
| Non-negotiable invariants | `03-protocol/INVARIANTS.md` |
| Invariant-to-test map | `02-architecture/TEST_MAP.md` |
| Test layers & commands | `../tests/README.md` |
| Test strategy & CI tiers | `05-standards/STD-0005-test-strategy.md` |
| Running / finalizing runs | `04-operations/RUNBOOK.md` |
| Readiness / known issues | `04-operations/READINESS.md` |
| Compute budget / remaining roadmap | `04-operations/COMPUTE_BUDGET.md` |
| Release and discoverability metadata | `04-operations/DISCOVERABILITY.md` |
| Output/naming/status/analysis | `05-standards/` (STD-0001 to STD-0005) |
| Contributing / security / citation | `../CONTRIBUTING.md`, `../SECURITY.md`, `../CITATION.cff` |
| Experiment records | `06-experiments/` |
| Decisions | `07-decisions/` |
| Paper extracts | `reference/paper-extracts/` |

## Quick CPU checks (no GPU)

```bash
uv run --frozen --extra cpu --extra dev pytest -p no:cacheprovider -q
uv run --frozen --extra cpu --extra dev ruff check --no-cache src tests scripts modal_app.py
bash scripts/verify.sh
```

## One-line warning

Any command that reaches Modal/H200 costs money; see `04-operations/COMPUTE_BUDGET.md`.
