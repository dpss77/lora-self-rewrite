import json
import os
import sys
import types

import pytest

from lora_self_rewrite import model_layout
from lora_self_rewrite.model_layout import (
    build_registry,
    init_empty_lora,
    module_path,
    peft_prefix,
    short_alias,
    text_layer_index,
)

pytestmark = pytest.mark.unit


def _key(i, proj):
    return f"model.language_model.layers.{i}.{proj}.weight"


def _weight_map(num_layers=64):
    wm = {}
    for i in range(num_layers):
        if i % 4 == 3:
            wm[_key(i, "self_attn.o_proj")] = "shard.safetensors"
        else:
            wm[_key(i, "linear_attn.out_proj")] = "shard.safetensors"
        wm[_key(i, "mlp.down_proj")] = "shard.safetensors"
    # Distractors that must never enter the registry.
    wm["model.visual.blocks.0.attn.qkv.weight"] = "shard.safetensors"
    wm["mtp.layers.0.self_attn.o_proj.weight"] = "shard.safetensors"
    wm["model.language_model.layers.0.input_layernorm.weight"] = "shard.safetensors"
    return wm


def test_text_layer_index_basic_and_mtp_exclusion():
    assert text_layer_index(_key(3, "self_attn.o_proj")) == 3
    assert text_layer_index(_key(63, "mlp.down_proj")) == 63
    assert text_layer_index("mtp.layers.0.self_attn.o_proj.weight") is None
    assert text_layer_index("model.visual.blocks.0.attn.qkv.weight") is None
    assert text_layer_index("lm_head.weight") is None


def test_key_helpers():
    k = _key(5, "mlp.down_proj")
    assert module_path(k) == "model.language_model.layers.5.mlp.down_proj"
    assert peft_prefix(k) == "base_model.model.model.language_model.layers.5.mlp.down_proj"
    assert short_alias(k) == "layers.5.mlp.down_proj"


def test_build_registry_has_128_targets_and_3to1_schedule():
    reg = build_registry(_weight_map(), num_layers=64)
    unique = {v["module_path"] for v in reg.values()}
    assert len(unique) == 128
    # 16 self_attn.o_proj + 48 linear_attn.out_proj + 64 mlp.down_proj
    assert sum("self_attn.o_proj" in m for m in unique) == 16
    assert sum("linear_attn.out_proj" in m for m in unique) == 48
    assert sum("mlp.down_proj" in m for m in unique) == 64
    assert all("mtp" not in m and "visual" not in m for m in unique)
    assert "layers.0.linear_attn.out_proj" in reg
    assert "layers.3.self_attn.o_proj" in reg


def test_build_registry_rejects_wrong_target_count():
    wm = _weight_map(num_layers=4)
    wm.pop(_key(0, "mlp.down_proj"))
    with pytest.raises(ValueError):
        build_registry(wm, num_layers=4)


def test_build_registry_rejects_alias_collision():
    # Two distinct prefixes mapping to the same short alias.
    wm = {
        "model.language_model.layers.0.mlp.down_proj.weight": "a",
        "layers.0.mlp.down_proj.weight": "b",
    }
    with pytest.raises(ValueError):
        build_registry(wm, num_layers=None)


CONSUMER_KEYS = {
    "layer",
    "module_path",
    "peft_prefix",
    "full_key",
    "shape",
    "shard_path",
}


def test_registry_entries_expose_consumer_schema():
    reg = build_registry(_weight_map(), num_layers=64)
    assert reg
    for info in reg.values():
        assert set(info) == CONSUMER_KEYS


def test_build_registry_shape_resolver_and_shard_path():
    reg = build_registry(
        _weight_map(),
        num_layers=64,
        shape_resolver=lambda _key: (32, 64),
    )
    for info in {v["module_path"]: v for v in reg.values()}.values():
        assert info["shape"] == (32, 64)
        assert info["shard_path"] == "shard.safetensors"


def test_init_empty_lora_populates_consumer_schema(tmp_path, monkeypatch):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    weight_map = _weight_map(num_layers=4)
    (model_dir / "config.json").write_text(json.dumps({"text_config": {"num_hidden_layers": 4}}))
    shapes = {key: (16, 8) for key in weight_map}

    class _Slice:
        def __init__(self, shape):
            self._shape = shape

        def get_shape(self):
            return self._shape

    class _Handle:
        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def get_slice(self, key):
            return _Slice(shapes[key])

    safetensors = types.ModuleType("safetensors")
    safetensors.safe_open = lambda *_a, **_k: _Handle()
    safetensors.__path__ = []
    safetensors_torch = types.ModuleType("safetensors.torch")
    safetensors_torch.save_file = lambda *_a, **_k: None
    monkeypatch.setitem(sys.modules, "safetensors", safetensors)
    monkeypatch.setitem(sys.modules, "safetensors.torch", safetensors_torch)
    monkeypatch.setattr(
        model_layout,
        "get_model_index",
        lambda _model_id, _revision=None: (str(model_dir), weight_map),
    )

    reg = init_empty_lora(
        str(tmp_path / "lora"),
        rank=4,
        reset=True,
        commit=lambda: None,
        model_id="fake",
    )

    unique = {v["module_path"]: v for v in reg.values()}
    assert len(unique) == 8  # 4 layers x 2 projections
    for info in unique.values():
        assert set(info) == CONSUMER_KEYS
        # Consumer-style indexing (sandbox_runtime.py:90,149).
        assert info["shape"][0] == 16
        assert info["shard_path"] == os.path.join(str(model_dir), "shard.safetensors")
    # Both the full path and the short alias must resolve to the same entry.
    sample = next(iter(unique.values()))
    assert reg[sample["module_path"]] is sample
    assert reg[short_alias(sample["full_key"])] is sample


def test_build_registry_rejects_wrong_self_attention_layer_schedule():
    wm = _weight_map(num_layers=64)
    # Move the self-attention projection from layer 3 to layer 0 while keeping
    # two targets per layer; counts stay 16/48/64 but the schedule is malformed.
    wm.pop(_key(0, "linear_attn.out_proj"))
    wm.pop(_key(3, "self_attn.o_proj"))
    wm[_key(0, "self_attn.o_proj")] = "shard.safetensors"
    wm[_key(3, "linear_attn.out_proj")] = "shard.safetensors"
    with pytest.raises(ValueError, match="self_attn.o_proj only on layers"):
        build_registry(wm, num_layers=64)


def test_build_registry_rejects_wrong_suffix_counts():
    wm = _weight_map(num_layers=64)
    # Keep two targets on layer 0 but replace one out_proj with a second down_proj.
    wm.pop(_key(0, "linear_attn.out_proj"))
    wm["model.language_model.layers.0.mlp.extra_down_proj.weight"] = "shard.safetensors"
    # The extra key is outside TARGET_SUFFIXES, so the layer now has one target.
    with pytest.raises(ValueError, match="exactly 2 target projections"):
        build_registry(wm, num_layers=64)

    wm2 = _weight_map(num_layers=64)
    wm2.pop(_key(1, "linear_attn.out_proj"))
    wm2[_key(1, "mlp.extra.down_proj")] = "shard.safetensors"
    with pytest.raises(ValueError, match="exactly 2 target projections"):
        build_registry(wm2, num_layers=64)
