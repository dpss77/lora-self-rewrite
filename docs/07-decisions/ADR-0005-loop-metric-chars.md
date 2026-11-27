# ADR-0005: Loop metric uses 6,000 characters, not bytes

- Status: accepted
- Date: 2026-11-26

## Context

The donor expression slices `text[-6000:]` characters, encodes the result to
UTF-8, compresses it with zlib level 6, and divides by the byte length. A
byte-slicing variant exists but is a non-donor diagnostic.

## Decision

Default `LOOP_METRIC_MODE="chars"` and preserve the exact donor expression:
`text[-6000:]` → UTF-8 → zlib level 6 → `len(compressed) / len(tail_bytes) < 0.15`.

## Consequences

- Multibyte text is handled exactly as in the donor and earlier runs.
- The byte-slicing variant remains only behind an explicit diagnostic flag.
- Changing the definition invalidates cached loop labels.
