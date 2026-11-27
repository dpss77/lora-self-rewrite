import re
from pathlib import Path

import pytest

from lora_self_rewrite import constants
from lora_self_rewrite.constants import (
    COMPACT_TOKEN_LIMIT,
    MAX_MODEL_LEN,
    WALL_LIMIT_HOURS,
    validate_compaction_budget,
)

pytestmark = pytest.mark.contract

SRC = Path(__file__).resolve().parents[2] / "src" / "lora_self_rewrite"


def test_locked_decision_values():
    assert WALL_LIMIT_HOURS == 3.0
    assert COMPACT_TOKEN_LIMIT == 48_000
    assert MAX_MODEL_LEN == 65_536
    assert COMPACT_TOKEN_LIMIT < MAX_MODEL_LEN
    # D1 explicitly holds LOOP_METRIC_MODE.
    assert constants.LOOP_METRIC_MODE == "chars"


def test_compaction_budget_guard():
    validate_compaction_budget()  # locked defaults fit
    validate_compaction_budget(48_000, 65_536)
    validate_compaction_budget(90_000, 114_688)
    with pytest.raises(ValueError, match="114_688"):
        validate_compaction_budget(90_000, 65_536)
    with pytest.raises(ValueError, match="must be <"):
        validate_compaction_budget(65_536, 65_536)


def test_engine_startup_uses_locked_constants():
    src = (SRC / "engine.py").read_text()
    assert "validate_compaction_budget(COMPACT_TOKEN_LIMIT, MAX_MODEL_LEN)" in src
    assert "max_model_len=MAX_MODEL_LEN" in src


def test_heldout_app_name_is_separate():
    src = (SRC / "eval_heldout.py").read_text()
    assert 'APP_NAME = "qwen35-27b-heldout-eval"' in src
    assert constants.MODAL_VOLUME_NAME == "self-rewrite-workspace"


def test_session_guards_before_gpu_startup():
    src = (SRC / "session.py").read_text()
    assert "validate_compaction_budget(COMPACT_TOKEN_LIMIT, MAX_MODEL_LEN)" in src


def test_gpu_images_pin_same_math_verify_version():
    versions = []
    for name in ("modal_wiring.py", "eval_heldout.py"):
        src = (SRC / name).read_text()
        assert '"math-verify"' not in src, f"{name} has an unpinned math-verify"
        match = re.search(r'"math-verify==([0-9][0-9.]*)"', src)
        assert match, f"{name} does not pin math-verify"
        versions.append(match.group(1))
    assert versions == ["0.9.0", "0.9.0"]
