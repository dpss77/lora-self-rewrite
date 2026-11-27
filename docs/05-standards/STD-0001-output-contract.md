# Output & artifact contract v1

Status: **finalized implementation baseline**. Enforced by
`src/lora_self_rewrite/manifest.py` and `scripts/`. Protocol values are locked in the
ADRs; do not change them without a new decision record.

## Principles

1. Code and stable docs live in Git. Raw run outputs do not.
2. One run = one immutable directory, one manifest, one human summary.
3. Raw evidence and interpretation are separate; never rewrite raw logs.
4. Every result is addressable by a stable experiment ID and run ID.
5. Machine-readable first (metrics/manifest), human-readable second.
6. No anonymous files (`analysis.md`, `output.md`, `terminal_output.md`).
7. One writer commits run state; GPU shards return rows in memory.
8. No secrets, weights, adapters, datasets, or caches in Git.

## Repository layout

```
lora-self-rewrite/
  modal_app.py                 # local Modal entrypoint
  src/lora_self_rewrite/       # package (pure protocol layer + GPU port)
  scripts/                     # run automation
  tests/                       # CPU characterization tests
  configs/experiments/         # experiment TOMLs
  docs/
    00-START-HERE.md
    01-context/
    02-architecture/
    03-protocol/
    04-operations/
    05-standards/              # STD-0001..STD-0005
    06-experiments/            # curated records + generated INDEX.md
    07-decisions/              # ADRs
    reference/                 # paper extracts and external references
  runs/                        # gitignored; raw/curated local run evidence
  artifacts/                   # gitignored; weights/adapters/datasets only
```

`runs/` and `artifacts/` are never committed. Local agent/tooling files are
gitignored and are not part of the public project.

## Run directory tree

```
runs/<run_id>/
  00_RUN_MANIFEST.json
  01_CONFIG.toml
  02_PROMPT.md
  03_DATASET/{DATASET_MANIFEST.json,DATASET_META.json}
  10_AGENT/{AGENT_NOTES.md,AGENT_TRANSCRIPT.jsonl,PASSES/PASS-XXXX.json}
  20_METRICS/{METRICS.json,PASS_LOG.jsonl,COST_LEDGER.json}
  30_VALIDATION/{VALIDATION.json,VALIDATION_ROWS.jsonl}
  40_HELDOUT/{HELDOUT.json,HELDOUT_ROWS.jsonl}
  50_ANALYSIS/{SUMMARY.md,ANALYSIS.md,TABLES.md}
  60_LOGS/{ORCHESTRATOR.log,ENGINE.log,PREFLIGHT.log}
  70_ARTIFACTS/{ADAPTERS/,CACHE_POINTERS.json}
  80_FINAL/{STATUS.json,CHECKSUMS.sha256}
```

## Manifest

`src/lora_self_rewrite/manifest.py` validates the minimum schema. Required
top-level fields: `schema_version`, `run_id`, `experiment_id`, `status`,
`created_at_utc`, `code{repo_sha,file_sha256}`, `model{id,revision,dtype}`,
`dataset{name,splits}`, `protocol{max_tokens,passes_budget,code_budget,
turns_budget,wall_limit_hours}`, `hardware{gpu_requested}`, `results{decision,
submitted_lora}`. Terminal statuses require `finished_at_utc`.

Write/update atomically. Commit the final status to both the Volume and the local
collected copy; verify after finalization.

## JSON Schemas

`schemas/` publishes draft-2020-12 JSON Schemas for the core machine-readable
artifacts:

| Schema | Artifact |
| :--- | :--- |
| `schemas/run_manifest.schema.json` | `00_RUN_MANIFEST.json` |
| `schemas/status.schema.json` | `80_FINAL/STATUS.json` |
| `schemas/metrics.schema.json` | `20_METRICS/METRICS.json` |
| `schemas/validation.schema.json` | Reserved-validation stats (`val_stats` / `validation_stats`) |
| `schemas/heldout.schema.json` | `40_HELDOUT/HELDOUT.json` |
| `schemas/pass.schema.json` | `10_AGENT/PASSES/PASS-*.json` / session `pass_<n>.json` |

`tests/contract/test_artifact_schemas.py` validates real `scripts/` output and
real fake-engine session/held-out output against these schemas, so the schemas
cannot drift from `session.py`, `eval_heldout.py`, or the run scripts. Change a
producer and the corresponding schema together.

The schemas intentionally do not enumerate optional diagnostics: per-turn timings,
GPU-memory snapshots, engine timing blocks, package-version extras, and row
`finish_reason` remain allowed without a schema bump. `COST_LEDGER.json` and
`CACHE_POINTERS.json` are not schema'd because no current writer emits them;
cache files are internal, identity-keyed caches rather than run artifacts.

## Automation

| Script | Purpose |
| :--- | :--- |
| `scripts/new_run.py` | allocate run id; write pre-run manifest + tree |
| `scripts/finalize_run.py` | enforce transition; write STATUS + checksums |
| `scripts/summarize_run.py` | generate SUMMARY/TABLES/ANALYSIS from JSON |
| `scripts/index_runs.py` | generate `docs/06-experiments/INDEX.md` |
| `scripts/check_artifacts.py` | validate schema, required files, checksums |
| `scripts/build_dataset.py` | deterministic DeepScaleR manifest |

## Git rules

- Commit: code, tests, configs, docs, manifests/schemas, small curated reports.
- Never commit: `runs/`, `artifacts/`, weights, adapters, datasets, caches, logs,
  `.env`, Volume contents, raw transcripts, local operator tooling.
- Pin model/dataset revisions and `uv.lock`.
- Generate `INDEX.md`; do not hand-maintain it.
