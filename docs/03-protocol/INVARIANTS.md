# Non-negotiable invariants

Violating any of these invalidates a run's scientific claim. Paper facts and v1 deviations
are separated in `docs/01-context/PAPER_SPEC_2611.21885v2.md`.

## Caching / hardware

1. `enable_prefix_caching=False` always. Prefix caching leaked base KV state into
   subsequent `LoRARequest` outputs (`stage1_02`).
2. Exact-batch base cache only. vLLM bf16 is not batch-size invariant; key on
   dataset mode, ordered pids, content hashes, `n_samples`, `max_tokens`,
   `cache_tag`, `GENERATION_CONFIG_HASH`, `cache_schema`.
3. H200-only for cacheable runs and held-out evaluation. No H100 fallback.
4. Single-writer Volume commits; GPU shards return rows in memory.

## Protocol

5. `B=20` passes, `C=200` code runs, `T_max=160` turns, 3.0 h wall (adaptation),
   `max_tokens=4096` in-session.
6. Never disclose turn count / turns remaining to the model.
7. Never add a hard `p<0.05` block on `FINAL_SUBMIT`.
8. Reserved validation does not consume the 20-pass budget.
9. `generate` ≤128 answers/pass in the Second Protocol; up to 4 candidates on
   identical seeds.
10. `read` ≤256 texts, ≤8,192 tokens; `answer_mean | every_k | last_k | all`.
11. Grading: `</think>` closed and final `\boxed{}` passes math-verify.
12. Loop: cut at limit AND last 6,000 units `zlib <0.15` (do not weaken).

## Edits / sandbox

13. Edits only in `lora_A`/`lora_B` output projections; do not edit embeddings,
    final norm, or LM head; reject pure scalar multiples of a whole tensor.
14. Sandbox: stateful CPU, gradients off, ≤600 s, 10,000-char stdout,
    no network/subprocess/model loaders.
15. HF extraction is spawn-isolated; never load HF in the main process.

## Evaluation

16. Held-out slice fixed at `heldout[500:990]` (490 problems) × 8 samples at
    16,384 tokens; one-sided 100k sign-flip, `p<0.05`.
17. Never evaluate a `WITHDRAW` run (no adapter) and never reuse held-out data to
    choose edits.
