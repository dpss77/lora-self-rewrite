"""Locked held-out protocol gate (pure; no Modal/vLLM/torch imports).

Non-smoke held-out runs must use exactly the approved configuration
(490 problems x 8 samples, 16,384 tokens, K=5) and a non-zero adapter that was
submitted from ``WORK/submitted_lora`` with a SUBMIT decision. A base-only
precompute is allowed without an adapter. Any deliberate smoke deviation must be
explicit via ``--dev-override``.
"""

from __future__ import annotations

import importlib.metadata
import os

from .adapter import (
    adapter_stats,
    read_submit_provenance,
)
from .caches import content_addressed_lora_key
from .constants import (
    HELDOUT_K_SHARDS,
    HELDOUT_MAX_TOKENS,
    HELDOUT_N_PROBLEMS,
    HELDOUT_N_SAMPLES,
)
from .engine_identity import engine_config as _engine_config
from .engine_identity import engine_config_hash as _engine_config_hash

__all__ = [
    "SUBMIT_DECISION",
    "adapter_stats",
    "available_package_versions",
    "eval_engine_config",
    "eval_engine_config_hash",
    "validate_adapter_nonzero",
    "validate_heldout_adapter",
    "validate_heldout_run_args",
    "validate_submit_provenance",
]

SUBMIT_DECISION = "SUBMIT"

_PACKAGES = (
    "torch",
    "vllm",
    "transformers",
    "peft",
    "accelerate",
    "safetensors",
    "math-verify",
    "datasets",
    "huggingface_hub",
)


def validate_heldout_run_args(
    *,
    n_problems: int,
    n_samples: int,
    max_tokens: int,
    k_shards: int,
    smoke_problems: int | None,
    dev_override: bool,
) -> None:
    """Reject any non-protocol held-out shape unless ``--dev-override`` is set."""
    if dev_override:
        if n_problems < 1:
            raise ValueError("Held-out run needs at least one problem.")
        if smoke_problems is not None and int(smoke_problems) < 1:
            raise ValueError("smoke_problems must be >= 1.")
        return

    if smoke_problems is not None:
        raise ValueError(
            "smoke_problems requires an explicit --dev-override. Refusing to "
            "silently produce a non-protocol held-out artifact."
        )

    expected = {
        "n_problems": HELDOUT_N_PROBLEMS,
        "n_samples": HELDOUT_N_SAMPLES,
        "max_tokens": HELDOUT_MAX_TOKENS,
        "k_shards": HELDOUT_K_SHARDS,
    }
    actual = {
        "n_problems": int(n_problems),
        "n_samples": int(n_samples),
        "max_tokens": int(max_tokens),
        "k_shards": int(k_shards),
    }
    mismatches = [
        f"{key}={actual[key]!r} (expected {expected[key]!r})"
        for key in expected
        if actual[key] != expected[key]
    ]
    if mismatches:
        raise ValueError(
            "Non-protocol held-out configuration: "
            + ", ".join(mismatches)
            + ". Pass --dev-override only for a deliberate smoke run."
        )


def eval_engine_config(
    *,
    model_id: str,
    model_revision: str | None,
    dtype: str,
    max_model_len: int,
    max_num_seqs: int,
    max_loras: int,
    max_lora_rank: int,
    gpu_memory_utilization: float,
    enable_prefix_caching: bool,
    trust_remote_code: bool,
    generation_config_hash: str,
    sleep_mode: bool = False,
) -> dict:
    """Identity of the held-out generation engine (shared by base and edited)."""
    return _engine_config(
        model_id=model_id,
        model_revision=model_revision,
        dtype=dtype,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        max_loras=max_loras,
        max_lora_rank=max_lora_rank,
        gpu_memory_utilization=gpu_memory_utilization,
        enable_sleep_mode=sleep_mode,
        enable_prefix_caching=enable_prefix_caching,
        trust_remote_code=trust_remote_code,
        generation_config_hash=generation_config_hash,
    )


def eval_engine_config_hash(config: dict) -> str:
    """Stable hash covering every engine numeric recorded in the config."""
    return _engine_config_hash(config)


def available_package_versions(packages: tuple[str, ...] = _PACKAGES) -> dict:
    """Best-effort resolved versions for provenance (missing -> ``None``)."""
    versions: dict[str, str | None] = {}
    for name in packages:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def validate_adapter_nonzero(stats: dict) -> None:
    """Fail when the submitted adapter has no non-zero elements."""
    if int(stats.get("nonzero_elements") or 0) <= 0:
        raise ValueError("Submitted adapter is all zeros; refusing held-out evaluation.")


def validate_submit_provenance(adapter_dir: str, provenance: dict | None = None) -> dict:
    """Require a SUBMIT decision recorded for the submitted adapter."""
    if os.path.basename(os.path.normpath(adapter_dir)) != "submitted_lora":
        raise ValueError(
            "Held-out adapter must be the session's WORK/submitted_lora directory, "
            f"got {adapter_dir!r}."
        )
    record = provenance if provenance is not None else read_submit_provenance(adapter_dir)
    if record.get("decision") != SUBMIT_DECISION:
        raise ValueError(
            f"Held-out adapter provenance is not a SUBMIT decision: {record.get('decision')!r}."
        )
    expected_sha = record.get("adapter_sha256")
    if expected_sha:
        actual_sha = content_addressed_lora_key(adapter_dir)
        if actual_sha != expected_sha:
            raise ValueError(
                "Held-out adapter bytes do not match the SUBMIT provenance digest "
                f"(provenance={expected_sha!r}, actual={actual_sha!r})."
            )
    return record


def validate_heldout_adapter(
    adapter_dir: str | None,
    *,
    base_only: bool = False,
    provenance: dict | None = None,
    stats: dict | None = None,
) -> dict | None:
    """Gate the held-out adapter path: base-only allowed, otherwise SUBMIT + non-zero."""
    if base_only:
        if adapter_dir is not None:
            raise ValueError("--base-only cannot be combined with an adapter.")
        return None
    if not adapter_dir:
        raise ValueError(
            "Non-smoke held-out evaluation requires a submitted adapter "
            "(or pass --base-only to precompute the base arm)."
        )
    record = validate_submit_provenance(adapter_dir, provenance=provenance)
    adapter_stats_value = stats if stats is not None else adapter_stats(adapter_dir)
    validate_adapter_nonzero(adapter_stats_value)
    return record
