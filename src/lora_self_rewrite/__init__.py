"""LoRA self-rewrite harness (arXiv:2611.21885v2 adaptation).

The package is split so that the *protocol-critical* logic is plain, importable
Python with no Modal/vLLM/torch dependency at import time. GPU-only code lives in
``engine``/``session``/``hf_worker`` and imports its heavy dependencies lazily or
only inside the Modal image.
"""

from .constants import (
    CODE_BUDGET,
    HELDOUT_MAX_TOKENS,
    HELDOUT_N_PROBLEMS,
    HELDOUT_N_SAMPLES,
    IN_SESSION_MAX_TOKENS,
    MODEL_ID,
    PASS_BUDGET,
    TURN_BUDGET,
    WALL_LIMIT_HOURS,
)

__all__ = [
    "MODEL_ID",
    "PASS_BUDGET",
    "CODE_BUDGET",
    "TURN_BUDGET",
    "WALL_LIMIT_HOURS",
    "IN_SESSION_MAX_TOKENS",
    "HELDOUT_MAX_TOKENS",
    "HELDOUT_N_PROBLEMS",
    "HELDOUT_N_SAMPLES",
]
