from pathlib import Path

import pytest

from lora_self_rewrite.constants import GENERATION_CONFIG_HASH, METRICS_VERSION
from lora_self_rewrite.instrumentation import (
    CACHE_FINGERPRINT_FIELDS,
    cache_fingerprint_from_env,
    cache_tag_from_fingerprint,
)

pytestmark = pytest.mark.contract

SRC = Path(__file__).resolve().parents[2] / "src" / "lora_self_rewrite"

# Recorded donor H200 environment (terminal_output_stage1_01.md:380 + ds_seed0).
# Includes decoy keys that are present in the full fingerprint but must not scope
# the cache namespace.
DONOR_FINGERPRINT = {
    "gpu": "NVIDIA H200",
    "gpu_capability": [9, 0],
    "torch": "2.14.0+cu130",
    "torch_cuda": "13.0",
    "triton": "3.8.0",
    "vllm": "0.34.0",
    "transformers": "5.22.2",
    "peft": "0.22.0",
    "model_id": "Qwen/Qwen3.5-27B",
    "model_revision": "81a287ca484a5e71148f5beb7f367bb5286d70f9",
    "model_dtype": "bfloat16",
    "generation_config_hash": "e7bbf912129e",
    "python": "3.11.14",
    "platform": "Linux",
    "accelerate": "1.0.0",
    "file_sha256": "deadbeef",
}

EXPECTED_FIELDS = {
    "gpu",
    "gpu_capability",
    "torch",
    "torch_cuda",
    "triton",
    "vllm",
    "transformers",
    "peft",
    "model_id",
    "model_revision",
    "model_dtype",
    "generation_config_hash",
}


def _donor_cache_fp():
    return cache_fingerprint_from_env(DONOR_FINGERPRINT, METRICS_VERSION)


def test_donor_h200_cache_tag_is_a370202bd599():
    assert cache_tag_from_fingerprint(_donor_cache_fp()) == "a370202bd599"


def test_generation_config_hash_matches_donor():
    assert GENERATION_CONFIG_HASH == "e7bbf912129e"


def test_fingerprint_field_list_and_schema_are_pinned():
    assert set(CACHE_FINGERPRINT_FIELDS) == EXPECTED_FIELDS
    assert METRICS_VERSION == 1
    donor_fp = _donor_cache_fp()
    assert donor_fp["cache_schema"] == 1
    # Every pinned dimension must affect the tag.
    for field in EXPECTED_FIELDS:
        changed = dict(donor_fp)
        changed[field] = "changed"
        assert cache_tag_from_fingerprint(changed) != "a370202bd599"
    assert cache_tag_from_fingerprint({**donor_fp, "cache_schema": 2}) != "a370202bd599"


def test_session_uses_the_shared_helper():
    src = (SRC / "session.py").read_text()
    assert "cache_fingerprint_from_env(gpu_fp, METRICS_VERSION)" in src
    assert "cache_tag_from_fingerprint(cache_fp)" in src
