# Test strategy

This document defines the test layers, markers, boundary fakes, golden-value
policy, and CI tiers. The module-to-test mapping lives in
`docs/02-architecture/TEST_MAP.md`; directory rules and commands live in
`tests/README.md`.

## Why a layered suite

The suite follows a **test trophy** shape rather than a pure pyramid:

- **Unit** tests are the broad base for pure protocol code: constants, seeds,
  statistics, paths, sandbox rules, manifest validation, dataset splitting, and
  model-layout discovery.
- **Contract** tests are the thick middle. They pin the externally observable
  identities that make runs comparable: donor parity, cache keys and tags,
  prompt rendering, grading, loop detection, held-out protocol gates, and
  source wiring that cannot be imported without a heavy runtime.
- **Integration** tests are narrower. They drive the real session/resume and
  script orchestration with fakes installed only at process boundaries.
- **Hardware** smoke is a thin tip. It is manual, explicitly approved, and never
  part of default collection or CI.

The trophy shape is deliberate: cache-key and prompt regressions are cheap to
catch in contract tests and expensive to discover after a paid run. A pure
pyramid would over-test helpers and under-test the identities that make the
science reproducible.

## Layer contracts

### Unit

- Pure CPU code only. No network, GPU, Modal, vLLM, or long-lived external
  process.
- File I/O is limited to `tmp_path`.
- Deterministic: seeded randomness, fixed clocks or injected timestamps.
- May monkeypatch a module-local dependency to avoid a heavy import, but must
  not replace the behaviour under test.

### Contract

- Pins an externally observable contract, not an implementation detail.
- Includes donor golden values, cache signature/tag identity, prompt and grading
  outputs, loop thresholds, protocol constants, and held-out gate decisions.
- May read source text to pin a wiring invariant that cannot be exercised on CPU
  without importing Modal/vLLM/torch. These are narrow assertions, not general
  source snapshots.
- Changing a pinned value requires a deliberate protocol decision and matching
  doc/invariant update. Do not update a golden value only to make a test pass.

### Integration

- Exercises the real orchestration function (`session`, `resume`, automation
  scripts, evaluator helpers) with fakes at the boundary.
- Fakes are allowed for the engine, Modal app/volume, tokenizer, package
  versions, and network/HF loaders.
- No GPU, no Modal call, no network. Filesystem work stays under `tmp_path`
  except the script-automation tests, which use the repository's shipped
  configs read-only and write only to temporary run directories.
- An integration test may use `subprocess` to prove the public CLI contract of a
  script; it must not start a server or a GPU job.

### Hardware

- H200 smoke only. See the policy below and `tests/hardware/README.md`.

## Markers

All markers are registered in `pyproject.toml`, and collection uses
`--strict-markers` plus `--strict-config`.

| Marker | Meaning | Default |
| :--- | :--- | :--- |
| `unit` | Pure CPU unit test | included |
| `contract` | Donor/cache/prompt/golden/protocol contract | included |
| `integration` | Fake-boundary orchestration test | included |
| `hardware` | H200 smoke test | excluded by default |
| `slow` | Long CPU test, opt in explicitly | included unless filtered |

Default `addopts` contains `-m "not hardware"` so a future hardware test cannot
run accidentally on a laptop or in CI. An explicit `-m hardware` on the command
line overrides the default expression.

## Fakes at boundaries

Use the smallest fake that crosses a real boundary:

- **Tokenizer:** a fake implementing `apply_chat_template`, including an older
  signature without `enable_thinking`.
- **Engine:** a fake with remote-shaped methods (`generate_batch`,
  `editor_step`, `get_environment`, `read_hidden_states`) that returns
  deterministic rows and raises on unexpected calls.
- **Modal wiring:** stand-in modules for `app`, `image`, `vol`, `WORKSPACE_DIR`,
  secrets, and safe commit/reload helpers.
- **Volume:** a stub that records `commit`/`reload` calls and can raise a
  transient failure.
- **Adapters/cache rows:** small factories that write the minimal files or rows
  a boundary expects.

Do not fake the function under test. Do not replace cache identity, grading, or
protocol gates with a constant: those are the contracts.

## Golden values and properties

The contract layer uses two complementary strategies:

- **Golden values** captured from the donor-parity port: seed recipe, prompt
  suffix, grading examples, loop detection, McNemar/bootstrap/sign-flip outputs,
  cache tag, and pinned protocol constants.
- **Property-style assertions** written without a property-testing dependency:
  every cache dimension must change the signature; sharding must cover every
  problem exactly once; resume counters must never under-count durable state;
  a non-submit or zero adapter must be rejected.

When adding a cache dimension or changing a prompt, add both the positive golden
case and the negative sensitivity assertion. An identity test that only checks
"same input gives same output" is insufficient.

## Coverage and mutation policy

- Line coverage is **not** a CI gate. The suite is judged by invariant coverage:
  does each risk row in `TEST_MAP.md` have a test that would fail if the
  invariant broke?
- Coverage reporting may be added later as an optional developer tool; do not
  add a new heavy dependency or a coverage threshold without an explicit
  decision.
- Mutation testing is not enabled in this phase. If introduced, it must be an
  optional local tool, never a required CI dependency, and it must start with
  the cache, prompt, adapter, held-out gate, and resume modules.
- Prefer a new targeted assertion over raising a global tolerance or skipping a
  flaky deterministic test.

## CI tiers

| Tier | Where | Command | Runs |
| :--- | :--- | :--- | :--- |
| Unit | CI and local | `pytest -m unit` | pure CPU tests |
| Contract | CI and local | `pytest -m contract` | donor/cache/prompt/golden contracts |
| Integration | CI and local | `pytest -m integration` | fake-boundary orchestration |
| Full CPU | local/verify | `pytest` (default) | all non-hardware layers |
| Hardware smoke | manual only | `pytest -m hardware tests/hardware` | approved H200 smoke |

CI runs the three CPU layers as separate steps, then the existing ruff and wheel
smoke checks. `scripts/verify.sh` runs the same three layers explicitly before
lint, format check, and an in-memory compile.

## Hardware-smoke policy

- No hardware test runs by default, in CI, or in `scripts/verify.sh`.
- Adding a hardware test requires explicit operator approval, a real H200
  environment, and the `hardware` marker.
- A smoke test covers startup plus a minimal generation/evaluation path. It does
  not replace the contract and integration tiers and must not assert scientific
  results on a tiny sample.
- Do not put commands that launch Modal/H200 jobs in CI or default local test
  commands. Cost approval lives in the operations runbook, not in tests.

## Rules for changes

- Add or update the lowest-layer test that can reproduce the change.
- Preserve existing assertions. Do not weaken a threshold, remove a negative
  case, or add a skip to make a change pass.
- If you change cache keys, prompts/generation config, adapter validation, the
  held-out gate, resume state, datasets, or artifacts, update the matching
  change-impact row in `docs/02-architecture/TEST_MAP.md`.
- Keep tests deterministic and network-free. Seed all randomness.
