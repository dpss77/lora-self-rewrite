"""Resolved session configuration and protocol-budget enforcement.

Configs are the explicit source of truth when a path is supplied. Everything the
session actually uses is resolved here and recorded in metrics/summary, so no TOML
key is silently dropped.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from .constants import (
    CODE_BUDGET,
    IN_SESSION_MAX_TOKENS,
    MAX_GEN_ANSWERS,
    MAX_READ_LENGTH,
    MODEL_ID,
    MODEL_REVISION,
    PASS_BUDGET,
    TURN_BUDGET,
    WALL_LIMIT_HOURS,
)

__all__ = [
    "ALLOWED_PROTOCOL_KEYS",
    "ALLOWED_TOP_LEVEL_KEYS",
    "cap_read_length",
    "load_run_config",
    "protocol_deviations",
    "resolve_session_config",
    "validate_protocol_session",
    "validate_run_config",
]

ALLOWED_TOP_LEVEL_KEYS = {
    "experiment_id",
    "stage",
    "arm",
    "dataset_name",
    "model_id",
    "model_revision",
    "protocol",
    "splits",
    "notes",
}

ALLOWED_PROTOCOL_KEYS = {
    "max_tokens",
    "passes_budget",
    "code_budget",
    "turns_budget",
    "wall_limit_hours",
    "max_gen_answers",
    "read_max_length",
    "signflip_resamples",
    "diag_n",
    "val_n",
    "val_samples",
}


def load_run_config(path: str | Path) -> dict:
    """Load and validate one experiment TOML."""
    with open(path, "rb") as f:
        config = tomllib.load(f)
    validate_run_config(config)
    return config


def validate_run_config(config: dict) -> None:
    """Reject malformed/unknown keys instead of silently discarding them."""
    if not isinstance(config, dict):
        raise ValueError("Experiment config must be a TOML table.")
    unknown = sorted(set(config) - ALLOWED_TOP_LEVEL_KEYS)
    if unknown:
        raise ValueError(f"Unknown top-level config keys: {unknown}.")
    protocol = config.get("protocol", {})
    if not isinstance(protocol, dict):
        raise ValueError("config.protocol must be a TOML table.")
    unknown_protocol = sorted(set(protocol) - ALLOWED_PROTOCOL_KEYS)
    if unknown_protocol:
        raise ValueError(f"Unknown protocol config keys: {unknown_protocol}.")
    splits = config.get("splits", {})
    if not isinstance(splits, dict):
        raise ValueError("config.splits must be a TOML table.")
    if not isinstance(config.get("experiment_id"), str) or not config["experiment_id"]:
        raise ValueError("config.experiment_id must be a non-empty string.")
    if not isinstance(config.get("dataset_name"), str) or not config["dataset_name"]:
        raise ValueError("config.dataset_name must be a non-empty string.")


def resolve_session_config(config: dict) -> dict:
    """Return every session field a config controls, with explicit defaults."""
    validate_run_config(config)
    protocol = config.get("protocol", {})
    return {
        "experiment_id": config["experiment_id"],
        "dataset_mode": config["dataset_name"],
        "model_id": config.get("model_id", MODEL_ID),
        "model_revision": config.get("model_revision", MODEL_REVISION),
        "max_tokens": int(protocol.get("max_tokens", IN_SESSION_MAX_TOKENS)),
        "max_passes": int(protocol.get("passes_budget", PASS_BUDGET)),
        "max_code_runs": int(protocol.get("code_budget", CODE_BUDGET)),
        "max_turns": int(protocol.get("turns_budget", TURN_BUDGET)),
        "wall_limit_hours": float(protocol.get("wall_limit_hours", WALL_LIMIT_HOURS)),
        "max_gen_answers": int(protocol.get("max_gen_answers", MAX_GEN_ANSWERS)),
        "read_max_length": int(protocol.get("read_max_length", MAX_READ_LENGTH)),
        "diag_n": protocol.get("diag_n"),
        "val_n": protocol.get("val_n"),
        "val_samples": protocol.get("val_samples"),
        "signflip_resamples": protocol.get("signflip_resamples"),
    }


def cap_read_length(requested: int | None = None) -> int:
    """Cap any read request at the protocol's 8,192-token budget."""
    value = MAX_READ_LENGTH if requested is None else int(requested)
    if value < 1:
        raise ValueError(f"read_max_length must be >= 1, got {value}.")
    return min(value, MAX_READ_LENGTH)


def protocol_deviations(
    *,
    max_tokens: int,
    max_passes: int,
    max_code_runs: int,
    max_turns: int,
    wall_limit_hours: float,
) -> list[str]:
    """List non-protocol session budget overrides."""
    expected = {
        "max_tokens": IN_SESSION_MAX_TOKENS,
        "max_passes": PASS_BUDGET,
        "max_code_runs": CODE_BUDGET,
        "max_turns": TURN_BUDGET,
        "wall_limit_hours": WALL_LIMIT_HOURS,
    }
    actual = {
        "max_tokens": int(max_tokens),
        "max_passes": int(max_passes),
        "max_code_runs": int(max_code_runs),
        "max_turns": int(max_turns),
        "wall_limit_hours": float(wall_limit_hours),
    }
    return [
        f"{key}={actual[key]!r} (protocol {expected[key]!r})"
        for key in expected
        if actual[key] != expected[key]
    ]


def validate_protocol_session(
    *,
    max_tokens: int,
    max_passes: int,
    max_code_runs: int,
    max_turns: int,
    wall_limit_hours: float,
    deviation: str | None,
) -> list[str]:
    """Require an explicit recorded deviation for any non-protocol smoke run."""
    deviations = protocol_deviations(
        max_tokens=max_tokens,
        max_passes=max_passes,
        max_code_runs=max_code_runs,
        max_turns=max_turns,
        wall_limit_hours=wall_limit_hours,
    )
    if deviations and not deviation:
        raise ValueError(
            "Non-protocol session budget(s): "
            + ", ".join(deviations)
            + ". Pass an explicit deviation label/mode to record a deliberate "
            "smoke run."
        )
    return deviations
