# Source of truth

**Purpose:** route contributors, reviewers, and coding agents to the current authority for each
topic. Read this before changing protocol, cache identity, experiment records, run artifacts, or
public claims.

This index is a router, not a second copy of the facts. Each fact has one owner. If the owner and a
summary disagree, fix the conflict; do not silently choose a side.

## Three kinds of source

| Kind | Answers | Where it lives | Rule |
| :--- | :--- | :--- | :--- |
| **Durable target** | What is intended and ratified? | Paper spec, invariants, ADRs, standards | **Trust first.** |
| **Live evidence** | What is true for the current code or an executed run? | Source, tests, run manifest, metrics, checksums, environment fingerprint | **Verify now; evidence expires.** |
| **Provenance** | How did we get here? | `internal/reviews/`, migration notes, chat history, un-promoted research | **Not authority.** Cite for *why*, not *what*. |

The raw paper is third-party provenance; the committed
[`PAPER_SPEC_2611.21885v2.md`](01-context/PAPER_SPEC_2611.21885v2.md) is the parsed project
reference. Raw HTML/PDF stays local under ignored `reference/paper/`.

## Authority map

| Topic | Canonical owner | Verify with | Do not treat as current authority |
| :--- | :--- | :--- | :--- |
| Paper facts and paper-vs-v1 gaps | [`docs/01-context/PAPER_SPEC_2611.21885v2.md`](01-context/PAPER_SPEC_2611.21885v2.md) | PDF/HTML for disputed wording; extracted references | chat summaries, old review notes, code comments |
| Current status, gates, budget | [`docs/04-operations/READINESS.md`](04-operations/READINESS.md), [`docs/04-operations/COMPUTE_BUDGET.md`](04-operations/COMPUTE_BUDGET.md), [`docs/00-START-HERE.md`](00-START-HERE.md) | latest run manifest; funded-run record | README prose or a remembered Modal balance |
| Project intent and closed milestones | [`docs/01-context/CONTEXT.md`](01-context/CONTEXT.md) | experiment records | review reports |
| Module boundaries and data flow | [`docs/02-architecture/MODULE_MAP.md`](02-architecture/MODULE_MAP.md) | source imports and tests | mermaid diagrams alone |
| Protocol invariants and budgets | [`docs/03-protocol/INVARIANTS.md`](03-protocol/INVARIANTS.md), [`docs/07-decisions/`](07-decisions/) | code + contract tests | README summaries |
| Scientific behavior | `src/` + `tests/` | run the CPU suite; inspect the relevant contract test | docs claiming behavior not exercised by tests |
| Cache identity and labels | `src/lora_self_rewrite/caches.py`, schema/tests | cache contract tests | comments or old cache files |
| Adapter/edit surface | `src/lora_self_rewrite/adapter.py`, registry | adapter negative tests + submit provenance | system-prompt wording alone |
| Held-out gate | [`docs/07-decisions/ADR-0003-heldout-k5.md`](07-decisions/ADR-0003-heldout-k5.md) + `heldout_gate.py` | gate tests and `HELDOUT.json` | a held-out claim without provenance |
| Run/artifact contract | [`docs/05-standards/STD-0001-output-contract.md`](05-standards/STD-0001-output-contract.md), [`schemas/`](../schemas/) | `check_artifacts.py` + schema tests | loose `analysis.md`/`terminal_output.md` |
| Experiment result | [`docs/06-experiments/`](06-experiments/) | run manifest, checksums, validation/held-out artifacts | raw logs or reviewer prose |
| Test strategy and coverage map | [`docs/05-standards/STD-0005-test-strategy.md`](05-standards/STD-0005-test-strategy.md), [`docs/02-architecture/TEST_MAP.md`](02-architecture/TEST_MAP.md) | `tests/README.md` and pytest markers | test-count badges |
| Naming and run IDs | [`docs/05-standards/STD-0002-naming.md`](05-standards/STD-0002-naming.md) | manifest/script validation | ad hoc filenames |
| Authorship and citation | [`AUTHORS.md`](../AUTHORS.md), [`CITATION.cff`](../CITATION.cff), [`LICENSE`](../LICENSE) | Git history; acknowledgements | LLM-generated claims of authorship |
| AI assistance | [`ACKNOWLEDGEMENTS.md`](../ACKNOWLEDGEMENTS.md) | development history | listing tools as authors |
| Local migration/tooling history | `internal/` (ignored); `.opsami/`, `.claude/`, `.agents/` (ignored) | operator only | anything in `internal/` |
| Agent/tool instruction files | `.opsami/`, `.claude/`, `.agents/`, `AGENTS.md`, `CLAUDE.md` (all ignored; local only) | local operator tooling | never public; use this source-of-truth page for project instructions |

## Mandatory read order

For a new contributor or agent:

1. [`README.md`](../README.md) — what the project is and what is not proven.
2. This file — where authority lives.
3. [`PAPER_SPEC_2611.21885v2.md`](01-context/PAPER_SPEC_2611.21885v2.md) — paper truth vs v1 gaps.
4. [`INVARIANTS.md`](03-protocol/INVARIANTS.md) — non-negotiable project rules.
5. [`READINESS.md`](04-operations/READINESS.md) — current gates and budget.
6. The relevant module/test map/experiment record before editing.

## Before changing a fact

- **Paper value:** edit `PAPER_SPEC` only for a verified paper correction; record any v1 deviation in
  the gap table and an ADR. Never change paper truth to match code.
- **Protocol value:** update the invariant, the matching ADR/decision, cache identity if affected,
  and the contract tests together.
- **Behavior:** update source and the lowest-layer test that fails without the change; update
  `TEST_MAP.md`.
- **Run artifact:** update `STD-0001`, the JSON Schema, and the schema test together.
- **Result/claim:** add/update an experiment record and link its manifest, checksums, validation,
  and (only if eligible) held-out artifact. Do not claim efficacy from a withdrawal or null result.
- **Author/citation:** edit `AUTHORS.md`/`CITATION.cff`/`LICENSE`; do not use commit history as a
  substitute for current authorship.
- **Internal/research note:** never promote by copy-paste; move the fact into a durable owner and
  link provenance.

## Conflict rules

1. **Paper vs code:** the paper spec says what the paper says; the code/tests say what v1 does. Record
   the delta explicitly.
2. **Docs vs code:** current source + passing tests are behavioral truth; stale docs are a bug.
3. **Manifest vs prose:** the manifest/checksums win for an executed run; curated prose must link to it.
4. **Review vs decision:** review notes are provenance; a durable ADR/standard is authority.
5. **Run vs cache:** a cached row is evidence only under its exact cache identity and hardware tag.

## Maintenance

- Every new durable document must be added to the authority map.
- Every renamed/moved doc must update this map and the Markdown links.
- `docs/00-START-HERE.md`, `README.md`, and `CONTRIBUTING.md` should link here.
- `internal/` is deliberately not linked from public docs; it is local provenance only.
- No `AGENTS.md` or other agent/tool instruction file is committed. A coding agent should use the
  mandatory read order above and this page as its project handoff. Tool-specific files stay local
  and ignored.
