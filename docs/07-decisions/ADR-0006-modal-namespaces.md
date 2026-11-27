# ADR-0006: Reuse the existing Modal app and volume names

- Status: accepted
- Date: 2026-11-26

## Context

Existing diagnosis/validation caches and run artifacts live on the Modal volume.
Creating new names would lose those caches; reusing names risks session
collisions if IDs are not managed carefully.

## Decision

Reuse `qwen35-27b-lora-self-rewrite` and `self-rewrite-workspace`. Give the
held-out evaluator its own app name, `qwen35-27b-heldout-eval`, mounting the same
volume. Keep session IDs validated and distinct.

## Consequences

- Existing paid diagnosis caches remain usable after re-grading/migration.
- Session hygiene and preemption resume must be correct to avoid collisions.
- Cache tags and model/dataset revisions remain part of every run manifest.
