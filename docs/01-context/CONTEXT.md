# Project context

## Concept

The paper has a model edit its own `bfloat16` weights to repair a diagnosed
failure. This project replaces destructive 54 GB checkpoint surgery with a
**zero-initialized rank-4 PEFT LoRA workspace** over the 128 text-backbone output
projections of Qwen3.5-27B:

```
W_eff = W_base - α u uᵀ W_base  ⇔  W_base + B A
B[:, k] = -α u,   A[k, :] = uᵀ W_base
```

Benefits: millisecond adapter hot-swaps, frozen base, structural protection of
embeddings/norms/LM head, portable `.safetensors` artifacts.

## Paper fidelity (Second Protocol / Table 5)

- `B=20` passes, `C=200` code runs, `T_max=160` turns, 3.0 h wall (adaptation),
  `max_tokens=4096` in-session.
- 4-way candidate comparison on identical seeds.
- Validation gate: 256×2 reserved problems at 4,096 tokens; model answers
  `SUBMIT`/`WITHDRAW`.
- Held-out: `heldout[500:990]` (490) × 8 samples at 16,384 tokens; one-sided
  problem-clustered sign-flip test, 100k resamples, `p<0.05`.

## Closed milestones

- `S01-MATH500-R03`: full 86-turn session, `FINAL_SUBMIT`, validation null
  (`+2.3% ± 3.9%`, `p=0.3473`). The model submitted anyway (allowed).
- `S02-DS-SEED00`: diagnosis `acc=9.8%`, `cut=228/256` (89.1%),
  `looping_cut=0/228`, `acc_given_finished=89.3%`; preempted at 19/20; resumed
  to 20/20; `FINAL_WITHDRAW`; no validation, no adapter.

## Scientific lesson

DeepScaleR failures are **long derivations**, not verbatim loops. Rank-1
late-layer ablations were insufficient (hidden-state cosine > 0.97). The next
seed needs a stronger intervention hypothesis, not another late-layer sweep.
