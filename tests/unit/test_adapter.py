import json
from pathlib import Path

import pytest
import torch

import lora_self_rewrite.adapter as adapter
from lora_self_rewrite.adapter import (
    adapter_stats_from_tensors,
    canonical_adapter_config,
    expected_lora_shapes,
    load_and_validate_adapter,
    validate_adapter_config,
    validate_lora_tensors,
)

pytestmark = pytest.mark.unit

RANK = 2


def _registry() -> dict[str, dict]:
    return {
        "layers.0.mlp.down_proj": {
            "module_path": "layers.0.mlp.down_proj",
            "peft_prefix": "base_model.model.model.layers.0.mlp.down_proj",
            "shape": (6, 4),
        },
        "layers.1.self_attn.o_proj": {
            "module_path": "layers.1.self_attn.o_proj",
            "peft_prefix": "base_model.model.model.layers.1.self_attn.o_proj",
            "shape": (5, 3),
        },
    }


def _tensors(registry: dict[str, dict], rank: int = RANK) -> dict:
    return {
        key: torch.zeros(shape, dtype=torch.bfloat16)
        for key, shape in expected_lora_shapes(registry, rank).items()
    }


def _first_a(tensors: dict) -> str:
    return next(key for key in tensors if key.endswith(".lora_A.weight"))


def test_valid_registry_adapter_passes():
    registry = _registry()
    validate_lora_tensors(_tensors(registry), registry, rank=RANK)


def test_rejects_unknown_key():
    registry = _registry()
    tensors = _tensors(registry)
    tensors["base_model.model.lm_head.lora_A.weight"] = torch.zeros((RANK, 4), dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="whitelist violation"):
        validate_lora_tensors(tensors, registry, rank=RANK)


def test_rejects_missing_key():
    registry = _registry()
    tensors = _tensors(registry)
    tensors.pop(_first_a(tensors))
    with pytest.raises(ValueError, match="whitelist violation"):
        validate_lora_tensors(tensors, registry, rank=RANK)


def test_rejects_wrong_shape():
    registry = _registry()
    tensors = _tensors(registry)
    key = _first_a(tensors)
    tensors[key] = torch.zeros((RANK + 1, 4), dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="shape violation"):
        validate_lora_tensors(tensors, registry, rank=RANK)


def test_rejects_non_finite_values():
    registry = _registry()
    tensors = _tensors(registry)
    key = _first_a(tensors)
    tensors[key][0, 0] = float("nan")
    with pytest.raises(ValueError, match="Non-finite"):
        validate_lora_tensors(tensors, registry, rank=RANK)


def test_rejects_wrong_dtype_for_saved_adapter():
    registry = _registry()
    tensors = _tensors(registry)
    key = _first_a(tensors)
    tensors[key] = tensors[key].float()
    with pytest.raises(ValueError, match="dtype violation"):
        validate_lora_tensors(tensors, registry, rank=RANK)


def test_float_inputs_allowed_when_save_path_casts_to_bf16():
    registry = _registry()
    tensors = {key: value.float() for key, value in _tensors(registry).items()}
    validate_lora_tensors(tensors, registry, rank=RANK, require_bfloat16=False)
    converted = {key: value.to(torch.bfloat16) for key, value in tensors.items()}
    validate_lora_tensors(converted, registry, rank=RANK)


def test_adapter_config_is_immutable():
    config = canonical_adapter_config(rank=RANK)
    assert validate_adapter_config(config, rank=RANK) is config
    with pytest.raises(ValueError, match="immutable"):
        validate_adapter_config({**config, "r": RANK + 1}, rank=RANK)
    with pytest.raises(ValueError, match="immutable"):
        validate_adapter_config({**config, "extra": True}, rank=RANK)
    incomplete = dict(config)
    incomplete.pop("target_modules")
    with pytest.raises(ValueError, match="immutable"):
        validate_adapter_config(incomplete, rank=RANK)


def test_load_and_validate_adapter_checks_saved_tensors(tmp_path, monkeypatch):
    registry = _registry()
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps(canonical_adapter_config(rank=RANK))
    )
    good = _tensors(registry)
    monkeypatch.setattr(adapter, "load_adapter_tensors", lambda _dir: good)
    tensors, config = load_and_validate_adapter(str(adapter_dir), registry, rank=RANK)
    assert config["r"] == RANK
    assert set(tensors) == set(good)

    out_of_scope = dict(good)
    out_of_scope["base_model.model.lm_head.lora_B.weight"] = torch.zeros(
        (4, RANK), dtype=torch.bfloat16
    )
    monkeypatch.setattr(adapter, "load_adapter_tensors", lambda _dir: out_of_scope)
    with pytest.raises(ValueError, match="whitelist violation"):
        load_and_validate_adapter(str(adapter_dir), registry, rank=RANK)


def test_adapter_stats_detects_zero_and_nonzero():
    registry = _registry()
    tensors = _tensors(registry)
    assert adapter_stats_from_tensors(tensors)["nonzero_elements"] == 0
    tensors[_first_a(tensors)][0, 0] = 1.0
    assert adapter_stats_from_tensors(tensors)["nonzero_elements"] == 1


def test_sandbox_and_session_wire_strict_adapter_validation():
    src = Path(__file__).resolve().parents[2] / "src" / "lora_self_rewrite"
    sandbox_src = (src / "sandbox_runtime.py").read_text()
    assert "validate_lora_tensors(" in sandbox_src
    assert "load_and_validate_adapter(" in sandbox_src
    assert "adapter_config.json is immutable" in sandbox_src
    session_src = (src / "session.py").read_text()
    assert "load_and_validate_adapter(lora_dir, module_registry)" in session_src
    assert 'decision="SUBMIT"' in session_src
