---
experiment_id: S02-DS-SEED00
legacy_id: ds_seed0
status: withdrawn
decision: WITHDRAW
dataset: deepscaler
gpu: H200
validation: not-run
heldout: not-run
config: configs/experiments/S02-DS-SEED00.toml
run_ids: []
manifest: null
checksums: null
---

# EXP-S02-DS-SEED00

Legacy: `ds_seed0`. Dataset: DeepScaleR (scrubbed). GPU: H200.

## Outcome

- Diagnosis: `acc=9.8%`, `cut=228/256` (89.1%), `looping_cut=0/228`,
  `acc_given_finished=89.3%`.
- Original run preempted at 19/20 (`ABORTED_ERROR`).
- Manual resume (`--no-reset-active-lora`) completed 20/20 and chose
  **`FINAL_WITHDRAW`**; no validation, no submitted adapter.
- Combined: 39 passes, 24 generate calls, 1,514 answers, ~5.88M tokens.

## Interpretation

DeepScaleR is cut/verbosity-limited, not loop-limited. The only repeated positive
(`c26`, `+3.3%` on 30 problems) was one flipped answer and two fewer cuts; it did
not replicate. Hidden-state cosine similarity between finished and length-limited
answers was > 0.97, so rank-1 late-layer ablations were insufficient.

## Operational defect to fix

Automatic restart failed because `ds_seed0` already had `WORK` state; the manual
resume reset the pass budget/transcript. Fix resume continuity before any
multi-seed work.

## Next

A fresh `ds_seed1` with a stronger intervention hypothesis; do not rerun
`ds_seed0`.
