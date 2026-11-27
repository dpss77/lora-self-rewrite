from pathlib import Path

import pytest

from lora_self_rewrite.constants import (
    CODE_BUDGET,
    IN_SESSION_MAX_TOKENS,
    MAX_READ_LENGTH,
    MODEL_REVISION,
    PASS_BUDGET,
    TURN_BUDGET,
    WALL_LIMIT_HOURS,
)
from lora_self_rewrite.run_config import (
    cap_read_length,
    load_run_config,
    resolve_session_config,
    validate_protocol_session,
    validate_run_config,
)

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs" / "experiments"


def test_shipped_session_configs_are_protocol_and_fully_resolved():
    for name in ("S01-MATH500-R03.toml", "S02-DS-SEED00.toml"):
        resolved = resolve_session_config(load_run_config(CONFIGS / name))
        assert resolved["model_revision"] == MODEL_REVISION
        assert resolved["max_tokens"] == IN_SESSION_MAX_TOKENS
        assert resolved["max_passes"] == PASS_BUDGET
        assert resolved["max_code_runs"] == CODE_BUDGET
        assert resolved["max_turns"] == TURN_BUDGET
        assert resolved["wall_limit_hours"] == WALL_LIMIT_HOURS
        assert resolved["max_gen_answers"] == 128
        assert resolved["read_max_length"] == MAX_READ_LENGTH


def test_unknown_config_keys_are_rejected_not_dropped():
    with pytest.raises(ValueError, match="Unknown top-level"):
        validate_run_config({"experiment_id": "S", "dataset_name": "math500", "surprise": 1})
    with pytest.raises(ValueError, match="Unknown protocol"):
        validate_run_config(
            {
                "experiment_id": "S",
                "dataset_name": "math500",
                "protocol": {"max_tokens": 4096, "surprise": 1},
            }
        )


def test_non_protocol_budgets_require_explicit_deviation():
    protocol = dict(
        max_tokens=IN_SESSION_MAX_TOKENS,
        max_passes=PASS_BUDGET,
        max_code_runs=CODE_BUDGET,
        max_turns=TURN_BUDGET,
        wall_limit_hours=WALL_LIMIT_HOURS,
    )
    assert validate_protocol_session(**protocol, deviation=None) == []
    with pytest.raises(ValueError, match="Non-protocol session budget"):
        validate_protocol_session(**{**protocol, "max_turns": 20}, deviation=None)
    deviations = validate_protocol_session(
        **{**protocol, "max_turns": 20}, deviation="smoke: 20 turns"
    )
    assert deviations and "max_turns=20" in deviations[0]


def test_read_length_is_hard_capped_at_8192():
    assert cap_read_length(None) == MAX_READ_LENGTH
    assert cap_read_length(4096) == 4096
    assert cap_read_length(16384) == MAX_READ_LENGTH
    with pytest.raises(ValueError):
        cap_read_length(0)


def test_modal_app_defaults_to_protocol_turn_budget():
    source = (ROOT / "modal_app.py").read_text()
    assert "max_turns: int = TURN_BUDGET" in source
    assert "deviation: str | None = None" in source
    assert "config_path=config" in source
