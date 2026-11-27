"""Pure engine-configuration identity and preflight invariants."""

from __future__ import annotations

import hashlib
import json

__all__ = [
    "ENGINE_CONFIG_FIELDS",
    "engine_config",
    "engine_config_hash",
    "require_deterministic_repeat",
    "require_pinned_revision",
]

ENGINE_CONFIG_FIELDS = (
    "model_id",
    "model_revision",
    "dtype",
    "max_model_len",
    "max_num_seqs",
    "max_loras",
    "max_lora_rank",
    "gpu_memory_utilization",
    "enable_sleep_mode",
    "enable_prefix_caching",
    "trust_remote_code",
    "generation_config_hash",
)


def engine_config(
    *,
    model_id: str,
    model_revision: str | None,
    dtype: str,
    max_model_len: int,
    max_num_seqs: int,
    max_loras: int,
    max_lora_rank: int,
    gpu_memory_utilization: float,
    enable_sleep_mode: bool,
    enable_prefix_caching: bool,
    trust_remote_code: bool,
    generation_config_hash: str,
) -> dict:
    """Identity of the generation engine, shared by session and held-out caches."""
    return {
        "model_id": model_id,
        "model_revision": model_revision or "unknown",
        "dtype": dtype,
        "max_model_len": int(max_model_len),
        "max_num_seqs": int(max_num_seqs),
        "max_loras": int(max_loras),
        "max_lora_rank": int(max_lora_rank),
        "gpu_memory_utilization": float(gpu_memory_utilization),
        "enable_sleep_mode": bool(enable_sleep_mode),
        "enable_prefix_caching": bool(enable_prefix_caching),
        "trust_remote_code": bool(trust_remote_code),
        "generation_config_hash": generation_config_hash,
    }


def engine_config_hash(config: dict) -> str:
    """Stable hash covering every recorded engine numeric/mode."""
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]


def require_pinned_revision(value: str | None, *, label: str = "model_revision") -> str:
    """Reject an unknown/blank revision for a cacheable or scientific run."""
    if not value or str(value).strip().lower() in {"unknown", "none", "null"}:
        raise ValueError(f"{label} must be pinned to a non-unknown revision for a cacheable run.")
    return str(value)


def require_deterministic_repeat(deterministic_repeat: bool) -> None:
    """A non-deterministic stack invalidates the frozen-base cache premise."""
    if not deterministic_repeat:
        raise RuntimeError(
            "FAIL-FAST: identical temperature=0 LoRA calls were not deterministic; "
            "the frozen-base cache cannot be trusted. This stack is non-cacheable."
        )
