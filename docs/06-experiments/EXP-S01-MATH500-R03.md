---
experiment_id: S01-MATH500-R03
legacy_id: stage1_03
status: submitted-null
decision: SUBMIT
dataset: math500
gpu: H200
validation: "64x2; +2.3% ± 3.9%; one-sided p=0.3473"
heldout: not-run
config: configs/experiments/S01-MATH500-R03.toml
run_ids: []
manifest: null
checksums: null
---

# EXP-S01-MATH500-R03

Legacy: `stage1_03`. Dataset: MATH-500. GPU: NVIDIA H200.

## Outcome

Completed 86 turns; `FINAL_SUBMIT`; reserved validation `64×2`:
Base `33.6%` (85 cut, 0 loop) vs Edited `35.9%` (83 cut, 0 loop),
`+2.3% ± 3.9%`, 95% CI `[-5.5%, +10.2%]`, one-sided sign-flip `p=0.3473`
(two-sided `0.6948`), `w2r=13`, `r2w=10`. Decision: **SUBMITTED** (model
judgment; no hard p-gate).

## Interpretation

Statistically null. The CI includes zero. Submitted edit: two rank-1 ablations at
`layers.24.linear_attn.out_proj` and `layers.32.linear_attn.out_proj`. The base is
cut-limited (66% of answers hit 4,096 tokens); cuts are non-looping long
derivations (`zlib 0.24–0.42`).

## Operational notes

- First clean run of `FINAL_SUBMIT` + validation + clustered stats + `val_base`
  + auto-snapshot + exact-batch cache under `enable_prefix_caching=False`.
- 9/11 generate calls were base-cache hits.
- `session_wall_s` 3939.86; `loop_gpu_fraction` 0.9892.

## Not evidence of improvement

Per `docs/04-operations/RUNBOOK.md`, do not run a full held-out eval for this
null edit.
