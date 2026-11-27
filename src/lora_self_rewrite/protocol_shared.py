"""Pure, decorator-free protocol layer shared by the session and held-out eval.

Nothing here imports Modal, vLLM, torch, or transformers at module import time.
That is load-bearing: the same code must run on the CPU orchestrator, inside the
GPU container, and in plain pytest on a laptop, with byte-identical behaviour.
"""

from __future__ import annotations

import zlib

from .constants import (
    GENERATION_CONFIG_HASH,
    GENERATION_SAMPLING,
    LOOP_METRIC_MODE,
    LOOP_TAIL_UNITS,
    LOOP_ZLIB_RATIO,
)

__all__ = [
    "GENERATION_SAMPLING",
    "GENERATION_CONFIG_HASH",
    "format_math_prompt",
    "apply_chat",
    "strip_thinking",
    "grade_answer",
    "is_looping_answer",
    "final_boxed_text",
]


def format_math_prompt(tokenizer, problem_text: str) -> str:
    """Apply the Table 5 prompt suffix and request Qwen3.5 thinking mode."""
    msg = [
        {
            "role": "user",
            "content": (
                f"{problem_text}\n"
                "Please reason step by step, and put your final answer within \\boxed{}."
            ),
        }
    ]
    return apply_chat(tokenizer, msg)


def apply_chat(tokenizer, messages: list[dict]) -> str:
    """Render messages with thinking enabled, falling back for older tokenizers."""
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def strip_thinking(text: str) -> str:
    """Return only the post-``</think>`` portion when present.

    Blocks/keywords discussed inside the thinking span must never pollute tool
    parsing or SUBMIT/WITHDRAW confirmation.
    """
    if "</think>" in text:
        return text.split("</think>")[-1]
    return text


def final_boxed_text(text: str) -> str:
    """Post-thinking text used for grading (the ``</think>`` must be closed)."""
    if "</think>" not in text:
        return ""
    return text.split("</think>")[-1]


def grade_answer(response_text: str, ground_truth: str) -> bool:
    """Table 5 grading: ``</think>`` closed and final ``\\boxed{}`` matches reference.

    ``math_verify`` is imported lazily so this module is importable without it.
    Any parse/verify failure is a non-correct answer, never an exception.
    """
    if "</think>" not in response_text:
        return False
    from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify

    final_part = response_text.split("</think>")[-1]
    try:
        gt_parsed = parse(
            f"${ground_truth}$",
            extraction_config=[
                LatexExtractionConfig(boxed_match_priority=0, try_extract_without_anchor=True),
                ExprExtractionConfig(),
            ],
        )
        pred_parsed = parse(
            final_part,
            extraction_config=[
                LatexExtractionConfig(boxed_match_priority=0),
                ExprExtractionConfig(),
            ],
        )
        return bool(gt_parsed and pred_parsed and verify(gt_parsed, pred_parsed))
    except Exception:  # noqa: BLE001 - grading must never crash a run
        return False


def is_looping_answer(
    text: str,
    cut: bool,
    *,
    mode: str = LOOP_METRIC_MODE,
    tail_units: int = LOOP_TAIL_UNITS,
    ratio: float = LOOP_ZLIB_RATIO,
) -> bool:
    """Table 5 loop definition.

    Loop iff the answer was cut at the token limit AND the last ``tail_units``
    units compress with zlib to less than ``ratio`` of their size.

    ``mode="chars"`` matches the paper and the donor: slice ``tail_units``
    characters, then UTF-8 encode. ``mode="bytes"`` is a non-donor diagnostic
    variant that slices the UTF-8 byte tail instead.
    """
    if not cut or not text:
        return False
    if mode == "chars":
        tail = text[-tail_units:].encode("utf-8", errors="ignore")
    elif mode == "bytes":
        tail = text.encode("utf-8", errors="ignore")[-tail_units:]
    else:
        raise ValueError(f"Unknown loop metric mode {mode!r}")
    if len(tail) < 200:
        return False
    compressed = zlib.compress(tail, level=6)
    return (len(compressed) / len(tail)) < ratio
