## What changed

<!-- One or two sentences. -->

## Why

<!-- Link the invariant/issue/experiment record. -->

## Verification

- [ ] `uv sync --frozen --extra cpu --extra dev`
- [ ] `uv run pytest -p no:cacheprovider -q`
- [ ] `uv run ruff check --no-cache src tests scripts modal_app.py`
- [ ] `uv run ruff format --check --no-cache src tests scripts modal_app.py`
- [ ] `./scripts/verify.sh`
- [ ] I did not change a protocol invariant, cache identity, grading/loop definition, or the
      held-out slice without a decision record and tests.

## Protocol / scientific impact

<!-- Does this change a cache key, prompt, adapter gate, held-out gate, or public claim? -->
