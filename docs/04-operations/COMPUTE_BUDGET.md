# Compute budget and remaining roadmap

**Status:** planning estimate, not a quote. As of 2026-11-27 the Modal balance is below one
H200 model load; no H200 job is currently funded. The measured `metrics.json` fields are ground
truth. Re-measure after the post-refactor G2 smoke before committing a sponsor budget.

## Billing assumptions

From the donor H200 telemetry and Modal's published rates:

| Item | Approx. rate |
| :--- | ---: |
| NVIDIA H200 SXM | $4.54 / GPU-hour |
| Host RAM (128 GiB) | $1.02 / hour |
| CPU orchestrator + container CPU | ~$0.35 / hour |
| **All-in active H200 session** | **~$5.90 / H200-hour** |

These are planning hypotheses. Modal pricing, model load, preemption, retries, and cache state can
change the real bill.

## Remaining work by tier

Tier estimates are **cumulative** planning ranges, not additive line items.

| Tier | Scope | Est. H200-hours | Est. cost |
| :--- | :--- | ---: | ---: |
| **Tier 1 — complete v1** | Post-refactor G2 smoke; 6–8 clean DeepScaleR Second-Protocol seeds; 490 × 8 held-out evaluation for submitted arms; retry/preemption buffer | **28–48 h** | **$165–$285** |
| **Tier 2 — v1 + v2 extensions** | Tier 1 plus 1D Gated DeltaNet `dt_bias`/`A_log` support, 6–8 matching seeds, held-out evaluation, multi-session chaining | **60–105 h** | **$355–$620** |
| **Tier 3 — optional full scope** | Tier 2 plus damaged-copy repair benchmark, additional rank/prompt ablations, and a larger buffer | **75–130 h** | **$440–$765** |

Per clean seed is roughly **1.1–1.9 H200-hours** (about **$6.50–$11**) after diagnosis and base
caches are warm; the first seed and the held-out base are more expensive. Do not treat these as
guarantees.

## What each tier unlocks

- **Tier 1:** the first honest multi-seed v1 efficacy test. It answers whether the rank-4 LoRA
  workspace reproduces the paper's positive sessions and produces a defensible held-out result.
- **Tier 2:** tests the paper's highest-gain mechanism (`dt_bias += 0.5`) and cross-session
  chaining; this is the closest comparison to the paper's complete Second Protocol.
- **Tier 3:** optional repair/ablation work; useful for a stronger paper, not required for the
  first v1 claim.

## What is already paid for

- Stage 1 (`S01-MATH500-R03`) full submit/validation loop (pre-refactor monolith).
- Stage 2 (`S02-DS-SEED00`) DeepScaleR diagnosis and one preempted/resumed seed
  (pre-refactor monolith).
- H200 diagnosis cache (`diag_deepscaler_256_4096`) and the corresponding cache tag.

These reduce the remaining cost but do **not** produce a positive efficacy claim. S02 withdrew;
the paper reports 3/8 Second-Protocol submissions and 5/8 withdrawals, so a withdrawal is not by
itself evidence of a broken method.

## What is not funded

- Post-refactor G2 smoke.
- Clean multi-seed v1 runs.
- Full 490 × 8 held-out evaluation.
- Any v2 `dt_bias`/`A_log` or multi-session work.

## How to help

- **Run compute:** follow [`RUNBOOK.md`](RUNBOOK.md), use H200 only, and contribute the run
  manifest, validation/held-out artifacts, and an experiment record through a GitHub issue or PR.
- **Sponsor compute:** open a GitHub issue with the tier, amount, and any constraints; do not send
  credentials. A Tier 1 allocation is the smallest useful request.
- **Contribute code:** the v2 `dt_bias`/`A_log` wrapper and multi-session chaining are the
  highest-value engineering extensions; start from
  [`PAPER_SPEC_2611.21885v2.md`](../01-context/PAPER_SPEC_2611.21885v2.md) and
  [`INVARIANTS.md`](../03-protocol/INVARIANTS.md).

## Guardrails

- H200-only; prefix caching off; exact-batch cache; held-out `[500:990]` (490) at 16,384 tokens.
- Do not weaken the validation/held-out gates to produce a sponsor-friendly result.
- Do not claim efficacy from a withdrawal, a null validation, or a diagnosis-only run.
- No secrets, raw logs, weights, adapters, or datasets in Git.
