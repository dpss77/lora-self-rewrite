from pathlib import Path

import pytest

from lora_self_rewrite.constants import MODEL_REVISION
from lora_self_rewrite.engine_identity import (
    ENGINE_CONFIG_FIELDS,
    engine_config,
    engine_config_hash,
    require_deterministic_repeat,
    require_pinned_revision,
)

pytestmark = pytest.mark.unit

SRC = Path(__file__).resolve().parents[2] / "src" / "lora_self_rewrite"


def _config(**overrides):
    values = {
        "model_id": "Qwen/Qwen3.5-27B",
        "model_revision": MODEL_REVISION,
        "dtype": "bfloat16",
        "max_model_len": 65536,
        "max_num_seqs": 128,
        "max_loras": 5,
        "max_lora_rank": 16,
        "gpu_memory_utilization": 0.88,
        "enable_sleep_mode": True,
        "enable_prefix_caching": False,
        "trust_remote_code": True,
        "generation_config_hash": "gen",
    }
    values.update(overrides)
    return engine_config(**values)


def test_engine_config_hash_covers_every_protocol_numeric_and_mode():
    config = _config()
    assert set(config) == set(ENGINE_CONFIG_FIELDS)
    digest = engine_config_hash(config)
    assert digest == engine_config_hash(dict(config))
    for field, value in (
        ("max_num_seqs", 64),
        ("max_loras", 2),
        ("max_lora_rank", 8),
        ("max_model_len", 32768),
        ("gpu_memory_utilization", 0.8),
        ("enable_sleep_mode", False),
        ("dtype", "float16"),
        ("enable_prefix_caching", True),
    ):
        assert engine_config_hash(_config(**{field: value})) != digest


def test_non_deterministic_repeat_is_a_hard_failure():
    require_deterministic_repeat(True)
    with pytest.raises(RuntimeError, match="non-cacheable"):
        require_deterministic_repeat(False)
    source = (SRC / "engine.py").read_text()
    assert "require_deterministic_repeat(deterministic_repeat)" in source
    assert "WARNING: identical LoRA calls differ" not in source


def test_unknown_revision_is_rejected_for_cacheable_runs():
    assert require_pinned_revision(MODEL_REVISION) == MODEL_REVISION
    for value in (None, "", "unknown", "UNKNOWN"):
        with pytest.raises(ValueError, match="pinned"):
            require_pinned_revision(value)
