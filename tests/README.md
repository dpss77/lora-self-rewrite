# Tests

The suite is layered so a slow or hardware-specific failure cannot hide a pure
protocol regression. Every test module declares exactly one primary layer marker:

| Layer | Directory | Marker | What belongs here |
| :--- | :--- | :--- | :--- |
| Unit | `tests/unit/` | `unit` | Pure functions and small modules. Temporary files only; no network, GPU, Modal, vLLM, or subprocesses. |
| Contract | `tests/contract/` | `contract` | Donor parity, cache keys/tags, prompt and grading golden values, protocol constants, held-out gate, and source-wiring checks that pin an invariant. |
| Integration | `tests/integration/` | `integration` | Real orchestration code driven with fakes at process boundaries: fake engine, fake Modal wiring, script automation, session/resume flows. |
| Hardware | `tests/hardware/` | `hardware` | H200 smoke only. No default test lives here. Requires operator approval. |

`tests/support/` holds shared fakes and factories when a second layer needs the
same boundary double. Do not import one test module from another.

## Running

The package is importable through `pythonpath = ["src"]`; nested test
directories work without `__init__.py` files.

```bash
# All CPU tiers (hardware is excluded by default)
uv run --frozen --extra cpu --extra dev pytest -p no:cacheprovider -q

# One tier at a time
uv run --frozen --extra cpu --extra dev pytest -q -m unit
uv run --frozen --extra cpu --extra dev pytest -q -m contract
uv run --frozen --extra cpu --extra dev pytest -q -m integration

# Approved H200 smoke only
uv run --frozen --extra cpu --extra dev pytest -q -m hardware tests/hardware
```

`unit`, `contract`, `integration`, `hardware`, and `slow` are registered in
`pyproject.toml`. `--strict-markers` and `--strict-config` are on, so an
unregistered marker or a typo fails collection.

## Adding tests

1. Choose the lowest layer that can still reproduce the bug or pin the change.
2. Use a unique filename `test_<module-or-invariant>.py`; avoid duplicate
   basenames across layers so pytest import mode stays unambiguous.
3. Add `pytestmark = pytest.mark.<layer>` after the imports.
4. Keep assertions strict. Do not add skips, loose tolerances, or `pytest.xfail`
   to hide a real regression.
5. Keep tests deterministic and network-free. Seed randomness explicitly and
   use `tmp_path` for file I/O.
6. Put a reusable fake or row factory in `tests/support/`, not in two test
   modules.
7. If a change affects an invariant in `docs/03-protocol/INVARIANTS.md`, update
   the matching row in `docs/02-architecture/TEST_MAP.md`.

## Change-impact map

`docs/02-architecture/TEST_MAP.md` maps each protocol invariant and module to
the test file and behaviour that protects it. Start there when changing cache
keys, prompts, adapter validation, the held-out gate, resume logic, datasets, or
artifact automation.
