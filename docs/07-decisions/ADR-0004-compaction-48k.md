# ADR-0004: Compaction limit is 48k for the current 65,536-token window

- Status: accepted
- Date: 2026-11-26

## Context

The donor used a 48,000-token compaction threshold with a 65,536-token model
window. The paper describes a 90k threshold, but that requires a larger model
window and a separate validation pass.

## Decision

Keep `COMPACT_TOKEN_LIMIT=48_000` for the current 65,536-token configuration.
Do not raise it to 90k without a separate change that sets `max_model_len` to at
least 114,688 and revalidates the protocol.

## Consequences

- The current implementation is a deliberate project adaptation, not exact paper
  parity for compaction.
- Changing the threshold changes context pressure and may change behavior; treat
  it as a protocol change.
