"""Deterministic seed derivation (PAPER: Table 5).

A ``(problem_id, sample_idx)`` pair must produce the same seed with and without
an edit, across every pass, session, and shard. The donor derived it as the
integer value of ``sha256(f"{pid}:{sample_idx}")`` truncated to 31 bits; we keep
that exact recipe so in-session evidence and the held-out gate draw identical
seeds.
"""

from __future__ import annotations

import hashlib

_SEED_MODULUS = 2**31


def derive_sample_seed(problem_id: int | str, sample_idx: int) -> int:
    """Return the deterministic vLLM sampling seed for one answer.

    Matches the donor: the 256-bit SHA-256 digest reduced ``% (2**31)``. The
    bound matters because ``SamplingParams(seed=...)`` is backed by a uint64/uint32
    generator; a full 256-bit digest is not a valid seed.
    """
    digest = hashlib.sha256(f"{problem_id}:{int(sample_idx)}".encode()).hexdigest()
    return int(digest, 16) % _SEED_MODULUS
