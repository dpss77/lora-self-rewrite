"""Strict LoRA adapter validation shared by the sandbox and session boundaries.

The protocol only permits rank-r edits over the exact registry
``*.lora_A.weight`` / ``*.lora_B.weight`` keys. This module enforces the key
whitelist, shapes/rank, finite bfloat16 values, and an immutable canonical
``adapter_config.json`` at every write/load/submit boundary.

A whole-tensor scalar multiple of ``W_base`` is structurally unreachable in a
rank-4 LoRA (it would need full rank), so no cosine/scalar-multiple check against
``W_base`` is performed; the key/shape/rank whitelist is the enforcement point.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping

from .caches import content_addressed_lora_key
from .constants import MODEL_ID

__all__ = [
    "ADAPTER_CONFIG_NAME",
    "ADAPTER_MODEL_NAME",
    "DEFAULT_RANK",
    "SUBMIT_PROVENANCE_NAME",
    "adapter_stats",
    "adapter_stats_from_tensors",
    "canonical_adapter_config",
    "expected_lora_shapes",
    "load_adapter_tensors",
    "load_and_validate_adapter",
    "read_adapter_config",
    "read_submit_provenance",
    "validate_adapter_config",
    "validate_lora_tensors",
    "write_submit_provenance",
]

ADAPTER_MODEL_NAME = "adapter_model.safetensors"
ADAPTER_CONFIG_NAME = "adapter_config.json"
SUBMIT_PROVENANCE_NAME = "submit_provenance.json"
DEFAULT_RANK = 4

_TARGET_MODULES = ["o_proj", "out_proj", "down_proj"]


def canonical_adapter_config(rank: int = DEFAULT_RANK, model_id: str = MODEL_ID) -> dict:
    """Return the one permitted adapter config for this protocol."""
    return {
        "peft_type": "LORA",
        "base_model_name_or_path": model_id,
        "r": int(rank),
        "lora_alpha": int(rank),
        "lora_dropout": 0.0,
        "target_modules": list(_TARGET_MODULES),
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }


def validate_adapter_config(
    config: object, *, rank: int = DEFAULT_RANK, model_id: str = MODEL_ID
) -> dict:
    """Require an exact canonical config; reject unknown/missing/changed fields."""
    if not isinstance(config, dict):
        raise ValueError("adapter_config.json must be a JSON object.")
    expected = canonical_adapter_config(rank=rank, model_id=model_id)
    missing = sorted(set(expected) - set(config))
    unknown = sorted(set(config) - set(expected))
    changed = sorted(key for key in expected if key in config and config[key] != expected[key])
    if missing or unknown or changed:
        raise ValueError(
            "adapter_config.json is immutable and must match the canonical config "
            f"(missing={missing}, unknown={unknown}, changed={changed})."
        )
    return config


def read_adapter_config(
    lora_dir: str, *, rank: int = DEFAULT_RANK, model_id: str = MODEL_ID
) -> dict:
    """Read and validate ``adapter_config.json`` from an adapter directory."""
    path = os.path.join(lora_dir, ADAPTER_CONFIG_NAME)
    try:
        with open(path) as f:
            config = json.load(f)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Unreadable adapter_config.json at {path}: {error}") from error
    return validate_adapter_config(config, rank=rank, model_id=model_id)


def expected_lora_shapes(
    module_registry: Mapping[str, Mapping[str, object]], rank: int
) -> dict[str, tuple[int, int]]:
    """Return the exact ``key -> shape`` whitelist for the registry at ``rank``."""
    rank = int(rank)
    if rank < 1:
        raise ValueError(f"LoRA rank must be >= 1, got {rank}.")
    unique = {info["module_path"]: info for info in module_registry.values()}
    if not unique:
        raise ValueError("Module registry is empty; cannot validate LoRA tensors.")
    shapes: dict[str, tuple[int, int]] = {}
    for info in unique.values():
        prefix = info.get("peft_prefix")
        shape = info.get("shape")
        if not prefix:
            raise ValueError(f"Registry entry has no peft_prefix: {info!r}")
        if not isinstance(shape, (tuple, list)) or len(shape) != 2:
            raise ValueError(f"Registry entry has no 2-D shape: {info!r}")
        out_features, in_features = int(shape[0]), int(shape[1])
        if out_features <= 0 or in_features <= 0:
            raise ValueError(f"Invalid registry shape for {prefix}: {shape!r}")
        shapes[f"{prefix}.lora_A.weight"] = (rank, in_features)
        shapes[f"{prefix}.lora_B.weight"] = (out_features, rank)
    return shapes


def _tensor_shape(tensor: object) -> tuple[int, ...]:
    try:
        return tuple(int(dim) for dim in tensor.shape)  # type: ignore[attr-defined]
    except (AttributeError, TypeError, ValueError) as error:
        raise TypeError(f"Expected a tensor-like value, got {type(tensor)!r}.") from error


def validate_lora_tensors(
    tensors_dict: Mapping[str, object],
    module_registry: Mapping[str, Mapping[str, object]],
    *,
    rank: int = DEFAULT_RANK,
    require_bfloat16: bool = True,
) -> None:
    """Enforce the exact registry key/shape/rank/finiteness whitelist."""
    if not isinstance(tensors_dict, Mapping):
        raise TypeError("LoRA tensors must be a mapping of key -> tensor.")
    expected = expected_lora_shapes(module_registry, rank)
    got = set(tensors_dict)
    unknown = sorted(got - set(expected))
    missing = sorted(set(expected) - got)
    if unknown or missing:
        raise ValueError(
            "LoRA tensor key whitelist violation: "
            f"unknown={unknown[:8]}, missing={missing[:8]} "
            f"({len(unknown)} unknown, {len(missing)} missing)."
        )

    import torch

    for key, shape in expected.items():
        tensor = tensors_dict[key]
        actual = _tensor_shape(tensor)
        if actual != shape:
            raise ValueError(
                f"LoRA tensor shape violation for {key}: expected {shape}, got {actual}."
            )
        if require_bfloat16:
            if getattr(tensor, "dtype", None) != torch.bfloat16:
                raise ValueError(
                    f"LoRA tensor dtype violation for {key}: expected "
                    f"torch.bfloat16, got {getattr(tensor, 'dtype', None)!r}."
                )
        elif not torch.is_floating_point(tensor):  # type: ignore[arg-type]
            raise TypeError(
                f"LoRA tensor dtype violation for {key}: expected a floating "
                f"tensor, got {getattr(tensor, 'dtype', None)!r}."
            )
        if not bool(torch.isfinite(tensor).all().item()):  # type: ignore[arg-type]
            raise ValueError(f"Non-finite values in LoRA tensor {key}.")


def load_adapter_tensors(lora_dir: str) -> dict:
    """Load raw adapter tensors (heavy import kept lazy for CPU callers)."""
    from safetensors.torch import load_file

    path = os.path.join(lora_dir, ADAPTER_MODEL_NAME)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing LoRA weights: {path}")
    return load_file(path)


def load_and_validate_adapter(
    lora_dir: str,
    module_registry: Mapping[str, Mapping[str, object]],
    *,
    rank: int | None = None,
    model_id: str = MODEL_ID,
) -> tuple[dict, dict]:
    """Load an adapter directory and validate config + tensors together."""
    config = read_adapter_config(
        lora_dir, rank=rank if rank is not None else DEFAULT_RANK, model_id=model_id
    )
    tensors = load_adapter_tensors(lora_dir)
    validate_lora_tensors(tensors, module_registry, rank=int(config["r"]), require_bfloat16=True)
    return tensors, config


def _nonzero_count(tensor: object) -> int:
    value = (tensor != 0).sum()  # type: ignore[operator]
    return int(value.item()) if hasattr(value, "item") else int(value)


def adapter_stats_from_tensors(tensors_dict: Mapping[str, object]) -> dict:
    """Summarize tensor count/size and non-zero elements without importing torch."""
    nonzero = 0
    elements = 0
    for tensor in tensors_dict.values():
        nonzero += _nonzero_count(tensor)
        elements += int(tensor.numel())  # type: ignore[attr-defined]
    return {
        "n_tensors": len(tensors_dict),
        "n_elements": elements,
        "nonzero_elements": nonzero,
    }


def adapter_stats(lora_dir: str) -> dict:
    """Summarize a saved adapter's weights (used by the held-out non-zero gate)."""
    return adapter_stats_from_tensors(load_adapter_tensors(lora_dir))


def write_submit_provenance(
    lora_dir: str,
    *,
    decision: str,
    session_id: str | None = None,
    model_id: str = MODEL_ID,
    extra: Mapping[str, object] | None = None,
) -> dict:
    """Write a durable submission decision sidecar next to the adapter files."""
    provenance = {
        "schema_version": 1,
        "decision": str(decision),
        "session_id": session_id,
        "model_id": model_id,
        "adapter_sha256": content_addressed_lora_key(lora_dir),
        "written_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if extra:
        provenance.update(dict(extra))
    path = os.path.join(lora_dir, SUBMIT_PROVENANCE_NAME)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(provenance, f, indent=2, default=str)
    os.replace(tmp, path)
    return provenance


def read_submit_provenance(lora_dir: str) -> dict:
    """Read the submission sidecar; fail closed when it is missing/corrupt."""
    path = os.path.join(lora_dir, SUBMIT_PROVENANCE_NAME)
    try:
        with open(path) as f:
            provenance = json.load(f)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Missing or unreadable SUBMIT provenance at {path}: {error}") from error
    if not isinstance(provenance, dict):
        raise ValueError(f"SUBMIT provenance at {path} must be a JSON object.")
    return provenance
