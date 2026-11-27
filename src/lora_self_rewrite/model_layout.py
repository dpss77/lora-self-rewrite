"""Qwen3.5-27B text-backbone target discovery and PEFT key construction.

The pure helpers (``text_layer_index``, ``peft_key``, ``build_registry``) are
importable without torch/safetensors; ``init_empty_lora`` is the GPU/CPU-side
initializer and imports its heavy dependencies lazily.

Layout invariant: 64 text layers under ``model.language_model.layers.{0..63}``
with a 3:1 hybrid schedule -> ``self_attn.o_proj`` on 16 layers, ``linear_attn.
out_proj`` on 48, ``mlp.down_proj`` on 64 = **128 exact target modules**.
Exclude ``visual``/``vision`` and the ``mtp.layers.0.*`` auxiliary head.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable

from .constants import MODEL_ID, MODEL_REVISION

__all__ = [
    "TARGET_SUFFIXES",
    "TEXT_LAYER_PREFIXES",
    "text_layer_index",
    "module_path",
    "peft_prefix",
    "short_alias",
    "build_registry",
    "get_model_index",
    "init_empty_lora",
]

TARGET_SUFFIXES = (
    ".self_attn.o_proj.weight",
    ".linear_attn.out_proj.weight",
    ".mlp.down_proj.weight",
)

TEXT_LAYER_PREFIXES = (
    "model.language_model.layers.",
    "language_model.layers.",
    "model.layers.",
    "layers.",
)


def text_layer_index(key: str) -> int | None:
    """Return the text-backbone layer index for a weight key, or ``None``.

    Excluding ``mtp`` is load-bearing: the Multi-Token Prediction head under
    ``mtp.layers.0.*`` would otherwise collide with real layer 0 aliases.
    """
    if ".mtp." in key or key.startswith("mtp."):
        return None
    for prefix in TEXT_LAYER_PREFIXES:
        if key.startswith(prefix):
            head = key[len(prefix) :].split(".", 1)[0]
            if head.isdigit():
                return int(head)
    return None


def module_path(weight_key: str) -> str:
    """Strip the trailing ``.weight`` from a canonical weight key."""
    if not weight_key.endswith(".weight"):
        raise ValueError(f"Not a weight key: {weight_key!r}")
    return weight_key[: -len(".weight")]


def peft_prefix(weight_key: str) -> str:
    """Canonical PEFT prefix for a base weight key."""
    return f"base_model.model.{module_path(weight_key)}"


def short_alias(weight_key: str) -> str:
    """Model-facing short alias (``layers.<i>.<proj>``)."""
    return f"layers.{module_path(weight_key).split('layers.', 1)[-1]}"


def build_registry(
    weight_map: dict[str, str],
    num_layers: int | None = None,
    shape_resolver: Callable[[str], tuple[int, int]] | None = None,
) -> dict:
    """Build the module registry from a safetensors ``weight_map`` (pure).

    Each entry carries the full consumer schema: ``layer``, ``module_path``,
    ``peft_prefix``, ``full_key``, ``shard_path`` (the map's shard filename) and
    ``shape``. ``shape`` is ``None`` unless a ``shape_resolver`` is supplied;
    ``init_empty_lora`` supplies one so consumers can index ``shape``/``shard_path``
    exactly as the donor did.

    Returns a dict keyed by both the full module path and the short alias. Raises
    on short-alias collisions and on an unexpected target count.
    """
    registry: dict[str, dict] = {}
    per_layer: dict[int, int] = {}
    for full_key, shard_file in weight_map.items():
        if "visual" in full_key or "vision" in full_key:
            continue
        if not full_key.endswith(TARGET_SUFFIXES):
            continue
        layer_num = text_layer_index(full_key)
        if layer_num is None:
            continue
        if num_layers is not None and not (0 <= layer_num < num_layers):
            continue
        mp = module_path(full_key)
        alias = short_alias(full_key)
        existing = registry.get(alias)
        if existing is not None and existing["module_path"] != mp:
            raise ValueError(
                f"Short alias collision for '{alias}': '{existing['module_path']}' vs '{mp}'."
            )
        info = {
            "layer": layer_num,
            "module_path": mp,
            "peft_prefix": peft_prefix(full_key),
            "full_key": full_key,
            "shape": shape_resolver(full_key) if shape_resolver is not None else None,
            "shard_path": shard_file,
        }
        registry[mp] = info
        registry[alias] = info
        per_layer[layer_num] = per_layer.get(layer_num, 0) + 1

    unique = {v["module_path"]: v for v in registry.values()}
    unique_count = len(unique)
    if not all(count == 2 for count in per_layer.values()):
        raise ValueError(f"Expected exactly 2 target projections per layer, got {per_layer}.")
    if num_layers is not None:
        if set(per_layer) != set(range(num_layers)):
            raise ValueError(
                f"Text backbone layers mismatch: found {sorted(per_layer)} but "
                f"config.json says {num_layers} layers."
            )
        if unique_count != 2 * num_layers:
            raise ValueError(
                f"Expected {2 * num_layers} text-backbone output projections, got "
                f"{unique_count}. Auxiliary/mtp layers probably leaked into the registry."
            )
        expected_self_layers = {i for i in range(num_layers) if i % 4 == 3}
        observed_self_layers = {
            info["layer"]
            for info in unique.values()
            if info["module_path"].endswith(".self_attn.o_proj")
        }
        counts = {
            "self_attn.o_proj": sum(
                info["module_path"].endswith(".self_attn.o_proj") for info in unique.values()
            ),
            "linear_attn.out_proj": sum(
                info["module_path"].endswith(".linear_attn.out_proj") for info in unique.values()
            ),
            "mlp.down_proj": sum(
                info["module_path"].endswith(".mlp.down_proj") for info in unique.values()
            ),
        }
        expected_counts = {
            "self_attn.o_proj": len(expected_self_layers),
            "linear_attn.out_proj": num_layers - len(expected_self_layers),
            "mlp.down_proj": num_layers,
        }
        if observed_self_layers != expected_self_layers:
            raise ValueError(
                "Expected self_attn.o_proj only on layers "
                f"{sorted(expected_self_layers)}, got {sorted(observed_self_layers)}."
            )
        if counts != expected_counts:
            raise ValueError(f"Expected the exact 3:1 layout {expected_counts}, got {counts}.")
    return registry


def get_model_index(
    model_id: str = MODEL_ID, revision: str = MODEL_REVISION
) -> tuple[str, dict[str, str]]:
    """Download (or reuse) the pinned snapshot and return its weight map (GPU-side)."""
    from huggingface_hub import snapshot_download

    model_dir = snapshot_download(
        repo_id=model_id,
        revision=revision,
        allow_patterns=["*.json", "*.safetensors"],
    )
    with open(os.path.join(model_dir, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    return model_dir, weight_map


def _read_num_layers(model_dir: str) -> int | None:
    try:
        with open(os.path.join(model_dir, "config.json")) as f:
            cfg = json.load(f)
        text_cfg = cfg.get("text_config", cfg)
        raw = text_cfg.get("num_hidden_layers") or cfg.get("num_hidden_layers")
        return int(raw) if raw is not None else None
    except (OSError, json.JSONDecodeError, TypeError, ValueError, AttributeError) as e:
        print(f"[init] could not read num_hidden_layers from config.json: {e}")
        return None


def init_empty_lora(
    lora_dir: str,
    rank: int = 4,
    reset: bool = True,
    commit: Callable[[], None] | None = None,
    model_id: str = MODEL_ID,
    model_revision: str = MODEL_REVISION,
) -> dict[str, dict]:
    """Initialize a zeroed rank-r PEFT LoRA over all 128 target projections.

    ``commit`` is an optional zero-argument callback (e.g. ``vol.commit``) invoked
    after writing a fresh adapter so the GPU sees it before any edit/test call.
    """
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    if commit is None:
        try:
            from .modal_wiring import vol as _vol

            commit = _vol.commit
        except Exception:  # noqa: BLE001 - CPU-only tests have no Modal app
            commit = None

    os.makedirs(lora_dir, exist_ok=True)
    model_dir, weight_map = get_model_index(model_id, model_revision)
    num_layers = _read_num_layers(model_dir)

    target_keys = []
    for full_key, shard_file in weight_map.items():
        if "visual" in full_key or "vision" in full_key:
            continue
        if not full_key.endswith(TARGET_SUFFIXES):
            continue
        layer_num = text_layer_index(full_key)
        if layer_num is None:
            continue
        if num_layers is not None and not (0 <= layer_num < num_layers):
            continue
        target_keys.append((full_key, shard_file))

    shape_cache: dict[str, tuple[int, int]] = {}

    def _resolve_shape(full_key: str) -> tuple[int, int]:
        shard_path = os.path.join(model_dir, weight_map[full_key])
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            shape = tuple(f.get_slice(full_key).get_shape())
        shape_cache[full_key] = shape
        return shape

    registry = build_registry(
        {k: v for k, v in target_keys}, num_layers, shape_resolver=_resolve_shape
    )
    # sandbox_runtime opens ``info["shard_path"]`` directly, so store the absolute
    # path here rather than the bare shard filename kept by the pure builder.
    for info in {v["module_path"]: v for v in registry.values()}.values():
        info["shard_path"] = os.path.join(model_dir, weight_map[info["full_key"]])

    lora_tensors = {}
    for full_key, _shard_file in sorted(target_keys):
        out_features, in_features = shape_cache[full_key]
        prefix = peft_prefix(full_key)
        lora_tensors[f"{prefix}.lora_A.weight"] = torch.zeros(
            (rank, in_features), dtype=torch.bfloat16
        )
        lora_tensors[f"{prefix}.lora_B.weight"] = torch.zeros(
            (out_features, rank), dtype=torch.bfloat16
        )

    adapter_model_path = os.path.join(lora_dir, "adapter_model.safetensors")
    adapter_config_path = os.path.join(lora_dir, "adapter_config.json")
    if reset or not os.path.exists(adapter_model_path):
        adapter_config = {
            "peft_type": "LORA",
            "base_model_name_or_path": model_id,
            "r": rank,
            "lora_alpha": rank,
            "lora_dropout": 0.0,
            "target_modules": ["o_proj", "out_proj", "down_proj"],
            "bias": "none",
            "task_type": "CAUSAL_LM",
        }
        with open(adapter_config_path, "w") as f:
            json.dump(adapter_config, f, indent=2)
        save_file(lora_tensors, adapter_model_path)
        if commit is not None:
            commit()
        print(
            f"Initialized {len({v['module_path'] for v in registry.values()})} "
            f"exact LoRA modules (rank={rank})."
        )
    else:
        print(f"Kept existing active LoRA at {lora_dir}.")
    return registry
