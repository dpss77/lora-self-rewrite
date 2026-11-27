import json
from pathlib import Path

import pytest

from lora_self_rewrite.constants import (
    GENERATION_CONFIG_HASH,
    HELDOUT_K_SHARDS,
    HELDOUT_MAX_TOKENS,
    HELDOUT_N_PROBLEMS,
    HELDOUT_N_SAMPLES,
    MODEL_ID,
)
from lora_self_rewrite.eval_common import assert_shard_identity
from lora_self_rewrite.heldout_gate import (
    available_package_versions,
    eval_engine_config,
    eval_engine_config_hash,
    validate_adapter_nonzero,
    validate_heldout_adapter,
    validate_heldout_run_args,
    validate_submit_provenance,
)

pytestmark = pytest.mark.contract

SRC = Path(__file__).resolve().parents[2] / "src" / "lora_self_rewrite"

PROTOCOL = {
    "n_problems": HELDOUT_N_PROBLEMS,
    "n_samples": HELDOUT_N_SAMPLES,
    "max_tokens": HELDOUT_MAX_TOKENS,
    "k_shards": HELDOUT_K_SHARDS,
    "smoke_problems": None,
    "dev_override": False,
}


def test_protocol_config_accepted():
    validate_heldout_run_args(**PROTOCOL)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("n_samples", 1),
        ("max_tokens", 8_192),
        ("k_shards", 1),
        ("n_problems", 100),
    ],
)
def test_non_protocol_config_requires_dev_override(field, value):
    with pytest.raises(ValueError, match="--dev-override"):
        validate_heldout_run_args(**{**PROTOCOL, field: value})
    # The same value is only acceptable as an explicit smoke deviation.
    validate_heldout_run_args(**{**PROTOCOL, field: value, "dev_override": True})


def test_smoke_problems_requires_dev_override():
    with pytest.raises(ValueError, match="--dev-override"):
        validate_heldout_run_args(**{**PROTOCOL, "smoke_problems": 10})
    validate_heldout_run_args(
        **{
            **PROTOCOL,
            "n_problems": 10,
            "smoke_problems": 10,
            "dev_override": True,
        }
    )


def test_zero_adapter_is_rejected():
    with pytest.raises(ValueError, match="all zeros"):
        validate_adapter_nonzero({"nonzero_elements": 0})
    with pytest.raises(ValueError, match="all zeros"):
        validate_adapter_nonzero({})
    validate_adapter_nonzero({"nonzero_elements": 1})


def _submitted_dir(tmp_path: Path) -> Path:
    path = tmp_path / "submitted_lora"
    path.mkdir()
    return path


def _write_provenance(path: Path, decision: str) -> None:
    (path / "submit_provenance.json").write_text(
        json.dumps({"schema_version": 1, "decision": decision})
    )


def test_missing_submit_provenance_is_rejected(tmp_path):
    submitted = _submitted_dir(tmp_path)
    with pytest.raises(ValueError, match="SUBMIT provenance"):
        validate_submit_provenance(str(submitted))


def test_non_submit_decision_is_rejected(tmp_path):
    submitted = _submitted_dir(tmp_path)
    _write_provenance(submitted, "WITHDRAW")
    with pytest.raises(ValueError, match="not a SUBMIT"):
        validate_submit_provenance(str(submitted))


def test_submit_provenance_is_accepted(tmp_path):
    submitted = _submitted_dir(tmp_path)
    _write_provenance(submitted, "SUBMIT")
    record = validate_submit_provenance(str(submitted))
    assert record["decision"] == "SUBMIT"


def test_adapter_must_come_from_submitted_lora(tmp_path):
    active = tmp_path / "active_lora"
    active.mkdir()
    with pytest.raises(ValueError, match="submitted_lora"):
        validate_heldout_adapter(
            str(active),
            provenance={"decision": "SUBMIT"},
            stats={"nonzero_elements": 1},
        )


def test_heldout_adapter_requires_nonzero_and_submit(tmp_path):
    submitted = _submitted_dir(tmp_path)
    with pytest.raises(ValueError, match="all zeros"):
        validate_heldout_adapter(
            str(submitted),
            provenance={"decision": "SUBMIT"},
            stats={"nonzero_elements": 0},
        )
    record = validate_heldout_adapter(
        str(submitted),
        provenance={"decision": "SUBMIT"},
        stats={"nonzero_elements": 4},
    )
    assert record["decision"] == "SUBMIT"


def test_base_only_is_allowed_without_adapter():
    assert validate_heldout_adapter(None, base_only=True) is None
    with pytest.raises(ValueError, match="cannot be combined"):
        validate_heldout_adapter(
            "/tmp/submitted_lora",
            base_only=True,
            provenance={"decision": "SUBMIT"},
            stats={"nonzero_elements": 1},
        )


def _engine_config(**overrides):
    values = {
        "model_id": MODEL_ID,
        "model_revision": "rev1",
        "dtype": "bfloat16",
        "max_model_len": 32768,
        "max_num_seqs": 64,
        "max_loras": 2,
        "max_lora_rank": 16,
        "gpu_memory_utilization": 0.88,
        "enable_prefix_caching": False,
        "trust_remote_code": True,
        "generation_config_hash": GENERATION_CONFIG_HASH,
    }
    values.update(overrides)
    return eval_engine_config(**values)


def test_eval_engine_config_hash_is_stable_and_sensitive():
    config = _engine_config()
    digest = eval_engine_config_hash(config)
    assert digest == eval_engine_config_hash(dict(config))
    assert eval_engine_config_hash(_engine_config(max_num_seqs=32)) != digest
    assert eval_engine_config_hash(_engine_config(enable_prefix_caching=True)) != digest


def test_assert_shard_identity_checks_eval_engine_hash():
    digest = eval_engine_config_hash(_engine_config())
    assert_shard_identity(
        {"cache_tag": "tag", "eval_engine_config_hash": digest},
        expected_cache_tag="tag",
        expected_eval_engine_config_hash=digest,
    )
    with pytest.raises(RuntimeError, match="eval engine config drift"):
        assert_shard_identity(
            {"cache_tag": "tag", "eval_engine_config_hash": "other"},
            expected_cache_tag="tag",
            expected_eval_engine_config_hash=digest,
        )


def test_available_package_versions_returns_requested_keys():
    versions = available_package_versions(("torch", "definitely-not-a-package"))
    assert set(versions) == {"torch", "definitely-not-a-package"}
    assert versions["torch"] is not None
    assert versions["definitely-not-a-package"] is None


def test_eval_heldout_wires_gate_secrets_and_bootstrap():
    source = (SRC / "eval_heldout.py").read_text()
    assert '__package__ = "lora_self_rewrite"' in source
    assert "secrets=_SECRETS" in source
    assert "validate_heldout_run_args(" in source
    assert "validate_heldout_adapter(" in source
    assert "expected_eval_engine_config_hash=engine_config_hash" in source
    assert "dev_override" in source
    assert "base_only" in source


def test_provenance_digest_must_match_adapter_bytes(tmp_path):
    submitted = _submitted_dir(tmp_path)
    (submitted / "adapter_model.safetensors").write_bytes(b"weights")
    (submitted / "adapter_config.json").write_text("{}")
    (submitted / "submit_provenance.json").write_text(
        json.dumps({"decision": "SUBMIT", "adapter_sha256": "deadbeef"})
    )
    with pytest.raises(ValueError, match="do not match"):
        validate_submit_provenance(str(submitted))
