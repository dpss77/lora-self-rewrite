# Naming

## IDs

- **Experiment ID:** `S<stage>-<DATASET>-<ARM>`
  - `S01-MATH500-R03`, `S02-DS-SEED00`, `EVAL-HELDOUT-490`
- **Run ID:** `<UTC timestamp>_<experiment-id>_<gpu>_<code-sha7>`
  - `20261126T061343Z_S02-DS-SEED00_H200_fe71f74`
- **Run directory:** `runs/<run_id>/`
- **Docs:** `EXP-<experiment-id>.md`, `ADR-<number>-<slug>.md`,
  `STD-<number>-<slug>.md`, `OPS-<slug>.md`, `REF-<slug>.md`.

## Rules

- Uppercase IDs/prefixes for artifact files; lower-kebab-case folders.
- No local times in IDs; filenames use UTC. Display local time only in prose.
- Inside a run, numeric prefixes enforce order and names are stable.
- Do not reuse old filenames; keep any historical alias mapping local-only.

## Legacy aliases

| Legacy | New |
| :--- | :--- |
| `stage1_03` | `S01-MATH500-R03` |
| `ds_seed0` | `S02-DS-SEED00` |
| `v06` | `S01`/`S02` experiment series |
