import hashlib
from pathlib import Path

import pytest

from lora_self_rewrite.seeds import derive_sample_seed

pytestmark = pytest.mark.contract

SRC = Path(__file__).resolve().parents[2] / "src" / "lora_self_rewrite"


def _donor_seed(problem_id, sample_idx):
    """Verbatim donor recipe: full digest reduced to 31 bits."""
    return int(hashlib.sha256(f"{problem_id}:{int(sample_idx)}".encode()).hexdigest(), 16) % (2**31)


def test_seed_is_deterministic_and_distinct():
    assert derive_sample_seed(1, 0) == derive_sample_seed(1, 0)
    assert derive_sample_seed(1, 0) != derive_sample_seed(1, 1)
    assert derive_sample_seed(1, 0) != derive_sample_seed(2, 0)
    assert derive_sample_seed("1", 0) == derive_sample_seed(1, 0)


def test_seed_is_bounded_and_matches_donor_expression():
    pairs = [(0, 0), (1, 0), (1, 1), (42, 3), ("abc", 2), (999, 7)]
    for problem_id, sample_idx in pairs:
        seed = derive_sample_seed(problem_id, sample_idx)
        assert 0 <= seed < 2**31
        assert seed == _donor_seed(problem_id, sample_idx)


def test_known_donor_values():
    # Pins the recipe independent of the helper above.
    assert derive_sample_seed(1, 0) == 1192650141
    assert derive_sample_seed(0, 0) == 171998802
    assert derive_sample_seed(42, 3) == 1209547345
    assert derive_sample_seed("abc", 2) == 940776825


def test_engine_calls_shared_seed_recipe():
    src = (SRC / "engine.py").read_text()
    assert "from .seeds import derive_sample_seed" in src
    assert 'derive_sample_seed(p["id"], s_idx)' in src
    # The 256-bit inline recipe must not reappear.
    assert "% (2**31)" not in src


def test_heldout_calls_shared_seed_recipe():
    src = (SRC / "eval_heldout.py").read_text()
    assert "from .seeds import derive_sample_seed" in src
    assert 'derive_sample_seed(problem["id"], sample_idx)' in src
