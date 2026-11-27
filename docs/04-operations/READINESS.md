# Readiness and known issues

**Status: CPU-only, pre-GPU.** No H200/Modal job is funded. The current Modal balance is
below one model load + vLLM startup; see `COMPUTE_BUDGET.md`.

## Closed results

- **S01-MATH500-R03:** submitted edit; reserved validation was statistically null
  (`+2.3% ± 3.9%`, one-sided `p=0.347`). No proven improvement.
- **S02-DS-SEED00:** DeepScaleR diagnosis was cut-limited (89.1% cuts, zero
  verbatim loops); the run ended in `WITHDRAW` after a preemption/resume, with no
  submitted adapter.

## Readiness gates

- [x] Project code separated from local migration/review/tooling history.
- [x] H200-only, prefix-caching-off, exact-batch cache, K=5 held-out slice.
- [x] CPU protocol/cache/resume/held-out-gate tests pass in the pinned test
      environment.
- [ ] GPU smoke (G2) — **not run; no budget**.
- [ ] Clean held-out evaluation — only after a submitting seed exists.
- [x] Local review and commit — prepared for publication.

## Known issues

- The migration audit's P0 protocol findings are addressed in code, but the GPU
  path has not been smoke-tested.
- The held-out evaluator has not been run against a submitted adapter.
- Compaction is a 48k project adaptation for the 65,536-token window, not the
  paper's 90k value.
- The sandbox is defence-in-depth, not a security boundary.
- Repository publication state is tracked outside this snapshot; check the remote.

See the ADRs in `docs/07-decisions/` and the protocol invariants in
`docs/03-protocol/INVARIANTS.md`.
