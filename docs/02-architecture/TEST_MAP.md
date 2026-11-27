# Test map

This map connects protocol invariants and modules to the tests that protect
them. Layer rules, markers, and commands live in `tests/README.md`; the strategy
and CI tiers live in `docs/05-standards/STD-0005-test-strategy.md`.

## Invariant -> test

Invariant numbers refer to `docs/03-protocol/INVARIANTS.md`.

| Invariant | Test file | Behaviour pinned |
| :--- | :--- | :--- |
| 1. Prefix caching off | `tests/unit/test_engine_identity.py`, `tests/contract/test_heldout_gate.py` | Engine/eval config hash changes when `enable_prefix_caching` changes. |
| 2. Exact-batch base cache | `tests/contract/test_caches.py`, `tests/contract/test_eval_common.py` | Ordered pids, problem hashes, answers, `n_samples`, `max_tokens`, tag, generation hash, schema, label schema, shard count, and model revision all scope the signature. |
| 2. Cache invalidation on label changes | `tests/contract/test_caches.py` | Loop-mode, cut-definition, grader, schema, and stale-token changes force a miss or regrade. |
| 3. H200-only cache identity | `tests/contract/test_cache_tag.py` | The recorded H200 fingerprint fields and pinned donor tag. |
| 4. Single-writer volume / shard rows | `tests/contract/test_eval_common.py`, `tests/integration/test_eval_heldout.py` | Shard identity checks cache tag, adapter bytes, and eval-engine hash; reload retries transient failures; GPU shards return rows in memory and only the orchestrator writes/commits results. |
| 5. Locked budgets | `tests/contract/test_constants.py`, `tests/unit/test_run_config.py` | Pass/code/turn/wall/token/compaction values and non-protocol budget rejection. |
| 6. Never disclose turn count | `tests/integration/test_resume.py` | Resumed/fresh editor prompts contain no "Begin Turn N" disclosure. |
| 7. No hard p-value block on submit | `tests/integration/test_resume.py` | A `FINAL_SUBMIT` path writes provenance without a p-value gate, even when validation is unfavorable. |
| 8. Reserved validation is off-budget | `tests/unit/test_run_config.py`, `tests/integration/test_resume.py` | Reserved split and protocol validation keep the 20-pass adaptation budget distinct; the real `FINAL_SUBMIT` validation path spends no pass and records metrics/transcript separately. |
| 9. Generate answer budget | `tests/integration/test_resume.py` | A generate action above the session answer/token budget is rejected before spending a pass. |
| 10. Read limits | `tests/unit/test_run_config.py` | `read_max_length` is hard-capped at 8,192. |
| 11. Closed thinking + boxed grading | `tests/contract/test_protocol_shared.py`, `tests/contract/test_donor_parity.py` | Unclosed thinking is never correct; golden grading cases match the donor. |
| 12. Loop definition | `tests/contract/test_protocol_shared.py`, `tests/contract/test_donor_parity.py`, `tests/contract/test_caches.py` | Cut-gated chars-mode zlib threshold, bytes-mode divergence, and re-derived loop labels. |
| 13. Edit whitelist / adapter shape | `tests/unit/test_adapter.py`, `tests/integration/test_resume.py` | Unknown, missing, wrong-shape, non-finite, wrong-dtype, and out-of-scope keys are rejected; submit refuses an out-of-scope adapter. |
| 13. Target registry | `tests/unit/test_model_layout.py` | Exactly 128 targets, 3:1 self-attention schedule, MTP/visual exclusion, PEFT keys, consumer schema. |
| 14. Sandbox restrictions | `tests/unit/test_sandbox.py` | Forbidden imports/attributes/network/model loaders, relative imports, and safe introspection. |
| 15. Spawn-isolated HF | `tests/integration/test_hf_spawn_isolation.py` | HF/PEFT/torch stay out of the parent process; hidden-state reads route through a spawned `_hf_extract_worker` and the orchestrator commits the returned payload. |
| 16. Fixed held-out slice and stats | `tests/unit/test_datasets.py`, `tests/contract/test_heldout_gate.py`, `tests/contract/test_eval_common.py`, `tests/contract/test_donor_parity.py`, `tests/integration/test_eval_heldout.py` | `heldout[500:990]`, 490-problem pool, 8-sample protocol, whole-problem shards, golden paired statistics, and base/edited running under one eval-engine hash. |
| 17. No evaluation of WITHDRAW / no held-out leakage | `tests/contract/test_heldout_gate.py`, `tests/integration/test_eval_heldout.py` | Missing/non-SUBMIT provenance, wrong directory, zero adapter, and adapter-digest mismatch are rejected before any shard; base-only is allowed without an adapter. |
| Provenance and run artifacts | `tests/unit/test_manifest.py`, `tests/integration/test_scripts.py` | Manifest schema/status, run ids, checksums, terminal re-seal refusal, artifact completeness, index generation. |
| Artifact JSON schemas | `tests/contract/test_artifact_schemas.py` | Real script and fake-engine `manifest`, `status`, `metrics`, `pass`, `validation`, and `heldout` payloads conform to `schemas/`; missing required fields are rejected. |
| Path and session safety | `tests/unit/test_paths.py` | Session-id validation, traversal rejection, and contained held-out/session paths. |

## Module -> test

| Module | Test file(s) |
| :--- | :--- |
| `adapter.py` | `tests/unit/test_adapter.py` |
| `caches.py` | `tests/contract/test_caches.py` |
| `constants.py` | `tests/contract/test_constants.py` |
| `datasets.py` | `tests/unit/test_datasets.py` |
| `engine_identity.py` | `tests/unit/test_engine_identity.py` |
| `eval_common.py` | `tests/contract/test_eval_common.py` |
| `eval_heldout.py` | `tests/integration/test_eval_heldout.py`, `tests/contract/test_eval_common.py`, `tests/contract/test_heldout_gate.py` |
| `heldout_gate.py` | `tests/contract/test_heldout_gate.py` |
| `hf_worker.py`, `engine.py` | `tests/integration/test_hf_spawn_isolation.py` |
| `instrumentation.py` | `tests/contract/test_cache_tag.py` |
| `manifest.py` | `tests/unit/test_manifest.py` |
| `model_layout.py` | `tests/unit/test_model_layout.py` |
| `paths.py` | `tests/unit/test_paths.py` |
| `protocol_shared.py` | `tests/contract/test_protocol_shared.py`, `tests/contract/test_donor_parity.py` |
| `resume.py`, `session.py` | `tests/integration/test_resume.py` |
| `run_config.py` | `tests/unit/test_run_config.py`, `tests/integration/test_resume.py` |
| `sandbox.py` | `tests/unit/test_sandbox.py` |
| `seeds.py` | `tests/contract/test_seeds.py`, `tests/contract/test_donor_parity.py` |
| `stats.py` | `tests/unit/test_stats.py`, `tests/contract/test_donor_parity.py` |
| `scripts/` | `tests/integration/test_scripts.py` |
| `schemas/` | `tests/contract/test_artifact_schemas.py` |

## Change-impact matrix

Update the listed tests **and** the listed public docs whenever the change alters
an observable identity.

| Change | Add or update | Also update | Notes |
| :--- | :--- | :--- | :--- |
| Base-cache signature dimension | `tests/contract/test_caches.py`, `tests/contract/test_eval_common.py` | `docs/03-protocol/INVARIANTS.md` #2 | Add a "changed dimension changes the signature" assertion. Bump `BASE_CACHE_SCHEMA` when stored rows become incompatible. |
| Diagnosis/validation cache identity | `tests/contract/test_caches.py` | `docs/03-protocol/INVARIANTS.md` #2, #11, #12 | Re-grade from text on load; update `GRADER_IDENTITY` and label schema together. |
| Cache tag / fingerprint fields | `tests/contract/test_cache_tag.py` | `docs/07-decisions/ADR-0002-prefix-caching-off.md`, invariant #2 | Never edit the pinned donor tag without recorded H200 evidence. Every pinned field must affect the tag. |
| Generation config hash | `tests/contract/test_cache_tag.py`, `tests/contract/test_donor_parity.py` | Invariant #2 | Treat as part of the cache namespace, not a free-floating constant. |
| Prompt suffix / chat template / thinking mode | `tests/contract/test_protocol_shared.py`, `tests/contract/test_donor_parity.py` | `docs/03-protocol/INVARIANTS.md` #11 | Keep the donor golden prompt case and the old-tokenizer fallback. |
| Grading rule or math-verify version | `tests/contract/test_protocol_shared.py`, `tests/contract/test_donor_parity.py`, `tests/contract/test_constants.py` | Invariant #11, `docs/05-standards/STD-0001-output-contract.md` if labels change | Bump `GRADER_IDENTITY` and the cache schema so paid text is re-graded. |
| Loop metric, tail size, or zlib ratio | `tests/contract/test_protocol_shared.py`, `tests/contract/test_donor_parity.py`, `tests/contract/test_caches.py` | `docs/07-decisions/ADR-0005-loop-metric-chars.md`, invariant #12 | Do not weaken the cut gate. Bump the label schema. |
| Adapter whitelist, dtype, shape, or config | `tests/unit/test_adapter.py`, `tests/integration/test_resume.py` | `docs/03-protocol/INVARIANTS.md` #13 | Keep reject cases for each violation; keep the submit provenance test. |
| Held-out gate, protocol shape, or provenance | `tests/contract/test_heldout_gate.py`, `tests/contract/test_eval_common.py`, `tests/integration/test_eval_heldout.py` | `docs/07-decisions/ADR-0003-heldout-k5.md`, invariants #16 and #17 | A non-SUBMIT, zero, wrong-directory, or mismatched-digest adapter must fail; base-only is allowed without one. |
| Dataset revisions, split counts, or held-out slice | `tests/unit/test_datasets.py`, `tests/integration/test_resume.py` | `docs/03-protocol/INVARIANTS.md` #16 | Update pinned revisions only with a deliberate dataset decision; keep the content hash test. |
| Resume counters, transcript merge, or wall budget | `tests/integration/test_resume.py` | `docs/04-operations/RUNBOOK.md`, invariant #6 | Keep jsonl-vs-snapshot, in-flight pass, read-pass, and cumulative-wall cases. |
| Engine config, compaction, or max-model length | `tests/unit/test_engine_identity.py`, `tests/contract/test_constants.py`, `tests/contract/test_heldout_gate.py` | `docs/07-decisions/ADR-0001-h200-only.md`, `ADR-0004-compaction-48k.md`, invariants #1 and #5 | Every numeric/mode must affect the config hash. |
| Manifest schema, run ids, or checksums | `tests/unit/test_manifest.py`, `tests/integration/test_scripts.py`, `tests/contract/test_artifact_schemas.py` | `docs/05-standards/STD-0001-output-contract.md`, `STD-0002-naming.md`, `STD-0003-run-status.md` | Keep terminal re-seal refusal and unlisted-artifact detection. Update `schemas/run_manifest.schema.json` with the validator. |
| Artifact shape (`metrics`, `pass`, `validation`, `heldout`, `status`) | `tests/contract/test_artifact_schemas.py` | `docs/05-standards/STD-0001-output-contract.md` | Change the producer and its schema together; keep at least one missing-required-field rejection per schema. |
| Session path or validation rules | `tests/unit/test_paths.py` | `docs/03-protocol/INVARIANTS.md` #14 | Keep traversal and sibling-path rejection. |
| Sandbox policy | `tests/unit/test_sandbox.py` | Invariant #14 | Add both an allowed case and the forbidden case. |
| Target registry / model layout | `tests/unit/test_model_layout.py` | Invariant #13 | Keep the 128-target count, 3:1 schedule, and alias-collision checks. |
| Shipped run config or protocol defaults | `tests/unit/test_run_config.py`, `tests/integration/test_resume.py` | `docs/05-standards/` and `configs/experiments/` provenance | Unknown keys must remain rejected, not dropped. |

## Known gaps

These invariants are enforced in runtime code or operations but do not have a
dedicated CPU test today:

- Physical H200 availability and actual GPU selection (the cache tag pins the
  environment, but tests cannot prove the hardware).
- An optional property-testing or mutation-testing gate.

Marking a gap here is not permission to weaken an existing test. Add a targeted
test when touching the relevant module.
