# Analysis template

Every run gets exactly two human files (plus generated tables):

## `50_ANALYSIS/SUMMARY.md` (one page)

- Status / decision
- One-paragraph TL;DR
- Key table: base, edited, delta, cut, p, n
- What failed / what remains
- Links to manifest, validation, held-out, logs

## `50_ANALYSIS/ANALYSIS.md` (detailed)

- Context and exact protocol (link `01_CONFIG.toml`)
- Evidence table (passes/batches/cache provenance)
- Statistical interpretation and limitations
- Operational issues (preemption/resume/cache)
- What it means for the next experiment
- Provenance and file checksums

## Rules

- Never mix raw logs into analysis prose; quote or link.
- Do not edit the model-generated `AGENT_NOTES.md`; quote or link it.
- `TABLES.md` is generated from JSON by `scripts/summarize_run.py`.
- The DB/`verification_report` discipline applies to work-item reviews; run
  analyses use this file contract.
