# Hardware smoke tests

This directory is reserved for **H200 smoke tests only**. There is no default
hardware test in this repository, and CI never runs this tier.

## Policy

- Hardware smoke tests require explicit operator approval and a real H200
  environment. Do not add one as part of ordinary CPU work.
- Mark every test in this directory with `@pytest.mark.hardware` (or a
  module-level `pytestmark = pytest.mark.hardware`).
- Default collection excludes `hardware` through `addopts = -m "not hardware"`.
- Run an approved smoke test explicitly:

  ```bash
  uv run --frozen --extra cpu --extra dev pytest -q -m hardware tests/hardware
  ```

- A smoke test proves startup and a tiny generation path. It is not a substitute
  for the CPU contract and integration tiers.
- Keep secrets out of test code. Use the same Modal secret and environment
  policy as the real runbook.

See `docs/05-standards/STD-0005-test-strategy.md` for the full hardware-smoke policy.
