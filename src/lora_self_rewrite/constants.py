"""Single source of truth for protocol constants and tuning knobs.

Values labeled ``PAPER`` come from arXiv:2611.21885v2 (Second Protocol / Table 5).
Values labeled ``DEVIATION`` are deliberate project choices; see
``docs/07-decisions/`` and ``docs/03-protocol/INVARIANTS.md``. Do not silently
change a protocol constant: cache keys and run manifests embed it.
"""

from __future__ import annotations

import hashlib
import json

# ---------------------------------------------------------------------------
# Model / infrastructure
# ---------------------------------------------------------------------------
MODEL_ID = "Qwen/Qwen3.5-27B"
# Pinned Hugging Face snapshot verified on the H200 runs. Every model load and
# cache identity must use this exact revision; "unknown" is not cacheable.
MODEL_REVISION = "81a287ca484a5e71148f5beb7f367bb5286d70f9"
WORKSPACE_DIR = "/workspace"

# Cacheable runs must pin one GPU class. The package default is H200 only; the
# legacy donor allowed an H100 fallback, which invalidated cache tags.
GPU_SPEC = ["H200"]

# Modal volume names (decision: reuse unless the operator renames them).
MODAL_APP_NAME = "qwen35-27b-lora-self-rewrite"
MODAL_VOLUME_NAME = "self-rewrite-workspace"

# Container-local JIT caches: never put POSIX-locked caches on a shared volume.
JIT_CACHE_DIR = "/tmp/jit_cache"
VLLM_CACHE_DIR = "/tmp/vllm_cache"

# ---------------------------------------------------------------------------
# Second Protocol budgets (PAPER: Sec. E.9 / Table 4)
# ---------------------------------------------------------------------------
PASS_BUDGET = 20  # B: forward passes
CODE_BUDGET = 200  # C: stateful CPU code runs
TURN_BUDGET = 160  # T_max: turns (never disclosed to the model)
WALL_LIMIT_HOURS = 3.0  # Second Protocol paper parity (Sec. E.9); diagnosis not counted
IN_SESSION_MAX_TOKENS = 4096  # PAPER: second-protocol answer budget
MAX_GEN_ANSWERS = 128  # per-pass generated-answer cap
MAX_READ_LENGTH = 8192  # per-read token budget (Table 5)

# Editor / sandbox limits
EDITOR_MAX_TOKENS = 8192
CODE_TIMEOUT_S = 600
CODE_STDOUT_LIMIT = 10_000
MAX_READ_TEXTS = 256
MAX_READ_LAYERS = 8
MAX_READ_BYTES = 4 * 1024**3
WORKER_TIMEOUT_S = 1800
SCALEDOWN_WINDOW_S = 240

# History compaction. Pure-refactor value: 48k, which fits max_model_len=65_536.
# The 90k paper value is only safe when max_model_len >= 114_688 (a separate
# post-refactor change); see ADR-0004. COMPACT_TOKEN_LIMIT must
# stay below MAX_MODEL_LEN, which validate_compaction_budget() enforces at startup.
COMPACT_TOKEN_LIMIT = 48_000
MAX_MODEL_LEN = 65_536
NOTES_MAX_CHARS = 36_000  # ~9-10k tokens; prevents compaction from looping


def validate_compaction_budget(
    compact_token_limit: int = COMPACT_TOKEN_LIMIT,
    max_model_len: int = MAX_MODEL_LEN,
) -> None:
    """Fail fast when the compaction limit cannot fit the model context window."""
    if compact_token_limit >= max_model_len:
        raise ValueError(
            f"COMPACT_TOKEN_LIMIT ({compact_token_limit}) must be < "
            f"MAX_MODEL_LEN ({max_model_len}); a 90_000 compaction limit requires "
            "max_model_len >= 114_688."
        )


# ---------------------------------------------------------------------------
# Sampling / grading (PAPER: Table 5)
# ---------------------------------------------------------------------------
GENERATION_SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20}
GENERATION_CONFIG_HASH = hashlib.sha256(
    json.dumps(GENERATION_SAMPLING, sort_keys=True).encode()
).hexdigest()[:12]

# Looping definition: cut at token limit AND last 6,000 units compress to <15%.
LOOP_TAIL_UNITS = 6_000
LOOP_ZLIB_RATIO = 0.15
# PAPER and the donor both slice the last 6,000 *characters* then UTF-8 encode.
# Default is that paper/donor value; "bytes" is a non-donor diagnostic mode.
LOOP_METRIC_MODE = "chars"  # "chars" (paper/donor) | "bytes" (diagnostic)

# ---------------------------------------------------------------------------
# Held-out evaluation (PAPER: Table 5)
# ---------------------------------------------------------------------------
HELDOUT_SLICE = (500, 990)  # fixed once; heldout[500:990]
HELDOUT_N_PROBLEMS = 490
HELDOUT_N_SAMPLES = 8
HELDOUT_MAX_TOKENS = 16_384
HELDOUT_K_SHARDS = 5
HELDOUT_GPU = "H200"
HELDOUT_PREFIX_CACHING = False
HELDOUT_SIGNFLIP_RESAMPLES = 100_000
HELDOUT_PRIMARY_ALPHA = 0.05

# ---------------------------------------------------------------------------
# Reproducibility / caches
# ---------------------------------------------------------------------------
METRICS_VERSION = 1
# Bumped to 3 for the engine-config hash (max_num_seqs/max_loras/max_lora_rank/
# max_model_len/gpu_memory_utilization/sleep/dtype/prefix-caching) in the base
# cache identity; no cache existed under schema 2 with that identity.
BASE_CACHE_SCHEMA = 3
DEEPSCALER_SCRAPER_VERSION = 2
DEEPSCALER_SPLITS = {"practice": (0, 3000), "reserved": (3000, 3300), "heldout": (3300, 4300)}
DEEPSCALER_SHUFFLE_SEED = 42

# ---------------------------------------------------------------------------
# Run status vocabulary (docs/05-standards/STD-0003-run-status.md)
# ---------------------------------------------------------------------------
STATUS_PLANNED = "planned"
STATUS_RUNNING = "running"
STATUS_PREEMPTED = "preempted"
STATUS_FAILED = "failed"
STATUS_WITHDRAWN = "withdrawn"
STATUS_SUBMITTED = "submitted"
STATUS_VALIDATED_NULL = "validated-null"
STATUS_VALIDATED_POSITIVE = "validated-positive"
STATUS_ARCHIVED = "archived"
STATUSES = (
    STATUS_PLANNED,
    STATUS_RUNNING,
    STATUS_PREEMPTED,
    STATUS_FAILED,
    STATUS_WITHDRAWN,
    STATUS_SUBMITTED,
    STATUS_VALIDATED_NULL,
    STATUS_VALIDATED_POSITIVE,
    STATUS_ARCHIVED,
)
TERMINAL_STATUSES = (
    STATUS_WITHDRAWN,
    STATUS_SUBMITTED,
    STATUS_VALIDATED_NULL,
    STATUS_VALIDATED_POSITIVE,
    STATUS_ARCHIVED,
    STATUS_FAILED,
)
