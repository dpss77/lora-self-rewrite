import pytest

from lora_self_rewrite.protocol_shared import (
    apply_chat,
    final_boxed_text,
    format_math_prompt,
    grade_answer,
    is_looping_answer,
    strip_thinking,
)

pytestmark = pytest.mark.contract


class FakeTokenizer:
    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
    ):
        return f"THINK={enable_thinking}|" + "|".join(m["content"] for m in messages)


def test_format_math_prompt_has_table5_suffix_and_thinking():
    out = format_math_prompt(FakeTokenizer(), "What is 2+2?")
    assert "THINK=True" in out
    assert "Please reason step by step" in out
    assert "\\boxed{}" in out


def test_apply_chat_falls_back_without_thinking_kwarg():
    class OldTokenizer:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            return "OLD"

    assert apply_chat(OldTokenizer(), [{"role": "user", "content": "x"}]) == "OLD"


def test_strip_thinking_keeps_only_post_think():
    text = "<think>secret FINAL_SUBMIT</think> hello"
    assert strip_thinking(text) == " hello"
    assert strip_thinking("no think") == "no think"


def test_final_boxed_text_requires_closed_thinking():
    assert final_boxed_text("no think") == ""
    assert final_boxed_text("<think>x</think>answer") == "answer"


def test_looping_requires_cut_and_repetition():
    import random
    import string

    repetitive = "abc" * 3000
    rng = random.Random(0)
    randomish = "".join(rng.choices(string.ascii_letters + string.digits, k=6000))
    assert is_looping_answer(repetitive, cut=True) is True
    assert is_looping_answer(repetitive, cut=False) is False
    assert is_looping_answer(randomish, cut=True) is False
    assert is_looping_answer("short", cut=True) is False
    assert is_looping_answer("", cut=True) is False


def test_looping_bytes_mode_matches_chars_for_ascii():
    repetitive = "abc" * 3000
    assert is_looping_answer(repetitive, True, mode="bytes") == is_looping_answer(
        repetitive, True, mode="chars"
    )
    with pytest.raises(ValueError):
        is_looping_answer(repetitive, True, mode="nope")


def test_grade_answer_rejects_unclosed_thinking():
    assert grade_answer("\\boxed{4}", "4") is False


def test_grade_answer_accepts_boxed_match_when_math_verify_available():
    pytest.importorskip("math_verify")
    assert grade_answer("<think>x</think>\\boxed{4}", "4") is True
    assert grade_answer("<think>x</think>\\boxed{5}", "4") is False
