# ADR-0003: Held-out evaluation uses K=5 shards

- Status: accepted
- Date: 2026-11-26

## Context

The held-out slice is fixed at 490 problems × 8 samples at 16,384 tokens. A
sharded evaluator needs a stable shard count so base-cache signatures and
single-writer result assembly remain reproducible.

## Decision

Use `K=5` shards, 98 problems per shard, H200-only. Retune only after a
one-shard smoke test; any change requires a new decision record and new cache
keys.

## Consequences

- Base and edited arms must use the same fixed shard list.
- A shard count change invalidates existing held-out base caches.
- The single CPU orchestrator, not GPU shards, writes and commits results.
