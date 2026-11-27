# G2 smoke checklist — first H200 validation

**Status:** prepared, **NOT RUN**. This is the Phase-2 GPU smoke gate.
It requires (a) a recorded `APPROVE_FOR_GPU_SMOKE` review and (b) explicit
operator budget approval (**G2**). No command here may run before both.

**Budget reality (2026-11-26):** Modal balance **$0.50**; H200 all-in burn
~**$5.90/h**. $0.50 is ~5 minutes — less than a model load + vLLM startup.
Do **not** start this smoke until the account is funded (smoke estimate
~$15–$27; one seed ~$6.50–$9). Add budget first.

## Preconditions

- [ ] G1 reviewer verdict recorded: `APPROVE_FOR_GPU_SMOKE`.
- [ ] Operator budget approval recorded (amount + cap).
- [ ] Locked decisions in force: `COMPACT_TOKEN_LIMIT=48_000`,
      `WALL_LIMIT_HOURS=3.0`, `LOOP_METRIC_MODE="chars"` (exact donor
      expression: `text[-6000:].encode(...)` then divide by its byte length).
- [ ] H200 pin present in code; no H100 fallback.
- [ ] Modal names: `qwen35-27b-lora-self-rewrite` / `self-rewrite-workspace`;
      held-out app `qwen35-27b-heldout-eval`.
- [ ] Secrets via `HF_SECRET_NAME`; nothing committed.
- [ ] No commit/push (G4).

## Checks (each must pass; stop on first failure)

1. **LoRA apply changes outputs.** A deliberate non-zero edit changes logprobs /
   outputs vs base on the same seed.
2. **Prefix-cache isolation.** `enable_prefix_caching=False`;
   `deterministic_repeat=True`, `base_between_stable=True`. Note
   `deterministic_repeat` currently only warns — treat a warn as a failure here.
3. **Seed bound accepted.** `SamplingParams(seed=...)` accepts the `2**31`-bounded
   `derive_sample_seed` value (B3).
4. **Real registry.** `init_empty_lora` against the real checkpoint: 128 targets,
   real `shape`, absolute `shard_path` openable by
   `sandbox_runtime.load_base_weight`.
5. **Sleep/wake extraction.** One `sleep(level=1)` → spawn HF worker → `.pt`
   write → `wake_up()`; host RAM < 128 GiB; no OOM.
6. **Content-addressed LoRA id.** Same adapter bytes → same id; changed bytes →
   a new id (B5b).
7. **Mid-pass kill + resume.** Kill mid-pass, resume: `passes_used` includes the
   in-flight pass; `code_runs_used` survives; `transcript.jsonl` appended
   (T2/T2b/T2c are CPU-verified only).
8. **Mid-run adapter change.** During a multi-shard held-out smoke, a later shard
   aborts when the adapter bytes change (B5b guard).
9. **Provenance recorded.** Write resolved `model_revision` and the resolved
   `math_verify` version into the run manifest.
10. **Cache compatibility.** Session `cache_tag` == `a370202bd599` on H200 and
    `GENERATION_CONFIG_HASH` == `e7bbf912129e`.

## Stop conditions

- Any check fails → stop, preserve `metrics.json` and logs, report. Do not spend
  into a seed run.
- Unexpected cost burn or repeated reload failures → stop.

## Artifacts to preserve

- `metrics.json` (with `gpu_container_billed_s_est`), `transcript.json`,
  `cache_tag`, `model_revision`, resolved `math_verify` version, Modal app/run
  IDs, and the exact checklist results.

## After the smoke

- Reviewer verifies the checklist evidence before any seed run.
- G3: challenger recommendation before a new seed if the intervention hypothesis
  remains contested.
- Full held-out gate additionally needs the pre-full-run items in
  `REVIEW_REPORT_G1.md` (I1, `cut` definition, `HELDOUT.json` schema, loop-cache
  key, `math_verify` pin, immutability, 490 check).
