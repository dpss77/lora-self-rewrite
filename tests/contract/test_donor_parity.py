"""Self-contained characterization tests for the donor-parity CPU layer.

The canonical donor ``modal_app.py`` cannot be imported (Modal at module scope),
and CI must not depend on a local donor tree. These tests therefore pin the
scientific behaviour with explicit golden values captured from the donor-parity
port. They always run: no donor path and no skip.

For an optional extra check, set ``ALLO_DONOR_PATH`` to a donor snapshot. The
test then verifies its SHA-256 against the pinned digest instead of trusting an
arbitrary file.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from lora_self_rewrite.caches import norm_question
from lora_self_rewrite.protocol_shared import (
    format_math_prompt,
    grade_answer,
    is_looping_answer,
    strip_thinking,
)
from lora_self_rewrite.seeds import derive_sample_seed
from lora_self_rewrite.stats import (
    bootstrap_ci,
    compute_paired_stats,
    mcnemar_exact,
    signflip_p,
)

pytestmark = pytest.mark.contract

DONOR_SHA256 = "fe71f74b0cebe933c155e03f59941c62ce402d92ca31ac2ce4e038c9b23dcbc2"


def test_optional_donor_snapshot_matches_pinned_sha():
    donor = os.environ.get("ALLO_DONOR_PATH")
    if not donor:
        return
    path = Path(donor)
    assert path.is_file(), f"ALLO_DONOR_PATH does not exist: {path}"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == DONOR_SHA256


def test_norm_question_matches_golden_values():
    cases = {
        "  A   B\nC ": "a b c",
        "x": "x",
        "  ": "",
        "MiXeD\tCase": "mixed case",
    }
    for raw, expected in cases.items():
        assert norm_question(raw) == expected


def test_strip_thinking_matches_golden_values():
    cases = {
        "<think>a</think>b": "b",
        "no think": "no think",
        "<think>only</think>": "",
        "a</think>b</think>c": "c",
    }
    for raw, expected in cases.items():
        assert strip_thinking(raw) == expected


def _multibyte_loop_sample() -> str:
    """Deterministic CJK text whose char-tail and byte-tail compress differently."""
    prefix = "".join(chr(0x4E00 + (i * 37) % 0x1000) for i in range(2500))
    return prefix + "数" * 4000


def test_loop_detection_matches_golden_chars_mode():
    samples = [
        "abc" * 3000,
        "short",
        "",
        "x" * 5000,
        "The answer is 42. " * 500,
        _multibyte_loop_sample(),
    ]
    expected = [
        (True, False),
        (False, False),
        (False, False),
        (True, False),
        (True, False),
        (False, False),
    ]
    for text, (cut_true, cut_false) in zip(samples, expected, strict=True):
        assert is_looping_answer(text, True, mode="chars") is cut_true
        assert is_looping_answer(text, False, mode="chars") is cut_false


def test_loop_bytes_mode_diverges_on_multibyte():
    # The non-donor "bytes" diagnostic mode must disagree here, proving the
    # chars-mode golden values above are not vacuous.
    text = _multibyte_loop_sample()
    assert is_looping_answer(text, True, mode="bytes") is True
    assert is_looping_answer(text, True, mode="chars") is False


def test_mcnemar_matches_golden_values():
    cases = {
        (0, 0): 1.0,
        (0, 5): 0.0625,
        (5, 0): 0.0625,
        (3, 7): 0.34375,
        (10, 2): 0.03857421875,
    }
    for (wrong_to_right, right_to_wrong), expected in cases.items():
        assert mcnemar_exact(wrong_to_right, right_to_wrong) == expected


def test_bootstrap_and_signflip_match_golden_values():
    diffs = [1.0, -1.0, 2.0, 0.0, 1.0, -2.0, 0.5]
    assert bootstrap_ci(diffs, resamples=200) == pytest.approx(
        (-0.8571428571428571, 1.1428571428571428)
    )
    ints = [1, 0, 1, -1, 1, 1, 0]
    assert signflip_p(ints, resamples=500, exact_int=True) == pytest.approx(
        (0.18962075848303392, 0.35728542914171657)
    )


def test_compute_paired_stats_matches_golden_values():
    base = [
        {"id": 1, "sample_idx": 0, "correct": True, "cut": 0, "loop": 0},
        {"id": 1, "sample_idx": 1, "correct": True, "cut": 1, "loop": 0},
        {"id": 2, "sample_idx": 0, "correct": False, "cut": 1, "loop": 0},
        {"id": 2, "sample_idx": 1, "correct": False, "cut": 1, "loop": 0},
        {"id": 3, "sample_idx": 0, "correct": False, "cut": 0, "loop": 0},
        {"id": 3, "sample_idx": 1, "correct": True, "cut": 1, "loop": 0},
    ]
    edited = [
        {"id": 1, "sample_idx": 0, "correct": True, "cut": 0, "loop": 0},
        {"id": 1, "sample_idx": 1, "correct": True, "cut": 1, "loop": 0},
        {"id": 2, "sample_idx": 0, "correct": True, "cut": 0, "loop": 0},
        {"id": 2, "sample_idx": 1, "correct": True, "cut": 1, "loop": 0},
        {"id": 3, "sample_idx": 0, "correct": False, "cut": 0, "loop": 0},
        {"id": 3, "sample_idx": 1, "correct": True, "cut": 1, "loop": 0},
    ]
    ours = compute_paired_stats(base, edited, bootstrap_resamples=200, signflip_resamples=500)
    expected = {
        "n": 6,
        "n_answers": 6,
        "base_acc": 0.5,
        "base_cut": 4,
        "base_loop": 0,
        "n_problems": 3,
        "samples_per_problem": 2,
        "edited_acc": 0.8333333333333334,
        "delta": 0.3333333333333333,
        "se": 0.33333333333333337,
        "ci_low": 0.0,
        "ci_high": 0.6666666666666666,
        "signflip_p_one_sided": 0.500998003992016,
        "signflip_p_two_sided": 1.0,
        "mcnemar_p": None,
        "wrong_to_right": 2,
        "right_to_wrong": 0,
        "edited_cut": 3,
        "edited_loop": 0,
    }
    assert ours.keys() == expected.keys()
    for key, value in expected.items():
        if isinstance(value, float):
            assert ours[key] == pytest.approx(value), key
        else:
            assert ours[key] == value, key


class _StubTokenizer:
    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
    ):
        return f"THINK={enable_thinking}|" + "|".join(
            f"{m['role']}:{m['content']}" for m in messages
        )


class _OldTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "OLD|" + "|".join(f"{m['role']}:{m['content']}" for m in messages)


def test_format_math_prompt_matches_golden_values():
    suffix = "\nPlease reason step by step, and put your final answer within \\boxed{}."
    cases = {
        "What is 2+2?": "What is 2+2?",
        "Prove the claim.": "Prove the claim.",
        "": "",
    }
    for problem, rendered in cases.items():
        assert format_math_prompt(_StubTokenizer(), problem) == (
            f"THINK=True|user:{rendered}{suffix}"
        )
        assert format_math_prompt(_OldTokenizer(), problem) == (f"OLD|user:{rendered}{suffix}")


def test_grade_answer_matches_golden_values():
    cases = [
        ("<think>x</think>\\boxed{4}", "4", True),
        ("<think>x</think>\\boxed{5}", "4", False),
        ("\\boxed{4}", "4", False),
        ("<think>x</think>\\boxed{42}", "42", True),
        ("<think>x</think>no box 42", "42", True),
        ("<think>x</think>\\boxed{\\frac{1}{2}}", "0.5", True),
    ]
    for text, ground_truth, expected in cases:
        assert grade_answer(text, ground_truth) is expected


def test_seed_matches_golden_donor_recipe():
    cases = {
        (0, 0): 171998802,
        (1, 0): 1192650141,
        (42, 3): 1209547345,
        ("abc", 2): 940776825,
        (999, 7): 678715533,
    }
    for (problem_id, sample_idx), expected in cases.items():
        assert derive_sample_seed(problem_id, sample_idx) == expected
        assert 0 <= expected < 2**31
