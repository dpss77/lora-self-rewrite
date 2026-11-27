# Contributing

Before editing, read [`docs/SOURCE-OF-TRUTH.md`](docs/SOURCE-OF-TRUTH.md) and follow its
mandatory read order. One concern per change; keep diffs scoped, preserve the protocol
invariants, and leave the repository in a CPU-verifiable state. Agent/tool instruction files
(`AGENTS.md`, `CLAUDE.md`, `.agents/`, `.claude/`, `.opsami/`) are local-only and must not be
committed; the source-of-truth page is the public handoff.

## Environment

Use `uv` only. Do not install project dependencies globally, with `pip`, or into
another repository's virtual environment.

```bash
uv sync --frozen --extra cpu --extra dev
```

Add `--extra gpu` or `--extra serve` only when the task actually requires the
heavy runtime; the default CPU profile intentionally excludes vLLM, Modal, and
the GPU stack. See [`README.md`](README.md#quickstart-cpu) for the full
quickstart.

## Tests, lint, and format

The suite is layered: `unit`, `contract`, and `integration` run by default;
`hardware` is manual and excluded from CI. Run the lowest layer that reproduces
a change, then the full CPU gate before handing off.

```bash
uv run --frozen --extra cpu --extra dev pytest -p no:cacheprovider -q
uv run --frozen --extra cpu --extra dev pytest -q -m unit
uv run --frozen --extra cpu --extra dev pytest -q -m contract
uv run --frozen --extra cpu --extra dev pytest -q -m integration
uv run --frozen --extra cpu --extra dev ruff check --no-cache src tests scripts modal_app.py
uv run --frozen --extra cpu --extra dev ruff format --check --no-cache src tests scripts modal_app.py
./scripts/verify.sh
```

Layer rules, markers, and fakes live in [`tests/README.md`](tests/README.md).
The strategy and CI tiers are in
[`docs/05-standards/STD-0005-test-strategy.md`](docs/05-standards/STD-0005-test-strategy.md);
the invariant-to-test mapping is in
[`docs/02-architecture/TEST_MAP.md`](docs/02-architecture/TEST_MAP.md).

## Quality bar

- Prefer a real call-site test over a mock. Fake only at a real boundary
  (engine, Modal wiring, tokenizer, network/HF loader) and never replace the
  function or contract under test.
- Do not weaken protocol invariants to make a change pass: no relaxed
  thresholds, removed negative cases, or new skips. If an invariant must
  change, update the matching decision record, invariant, cache identity, and
  tests together.
- Keep changelog and decision records current. This repository has no separate
  changelog; user-visible changes belong in the relevant docs, and
  behaviour-affecting changes need an ADR under
  [`docs/07-decisions/`](docs/07-decisions/).
- GPU/Modal work follows the budget-gated runbook. Hardware tests are explicit and never
  belong in CI or default local commands.
- Never commit secrets, weights, adapters, datasets, raw logs, or caches; see
  [`SECURITY.md`](SECURITY.md).

## Documentation

Standards use `STD-<number>-<slug>.md`; decisions use `ADR-<number>-<slug>.md`
(see [`docs/05-standards/STD-0002-naming.md`](docs/05-standards/STD-0002-naming.md)).
When adding or renaming a standard, update every relative link and the maps in
[`README.md`](README.md) and [`docs/00-START-HERE.md`](docs/00-START-HERE.md).
