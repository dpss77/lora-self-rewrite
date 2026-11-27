"""Path validation and containment for session/held-out filesystem access.

``session_id`` values are model/CLI controlled and are interpolated into Volume
paths, including recursive deletes. They must match a strict safe grammar and the
resolved path must remain under the sessions root. Held-out dataset/output paths
get the same containment treatment under the workspace root.

This is defence-in-depth for accidental/path-traversal mistakes, not a sandbox.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from .constants import WORKSPACE_DIR

__all__ = [
    "SESSIONS_SUBDIR",
    "SESSION_ID_RE",
    "contain_path",
    "resolve_under",
    "session_dir",
    "validate_session_id",
]

SESSIONS_SUBDIR = "sessions"
SESSION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def validate_session_id(session_id: str) -> str:
    """Reject ids that could escape ``/workspace/sessions`` or target dotfiles."""
    if not isinstance(session_id, str) or not SESSION_ID_RE.fullmatch(session_id):
        raise ValueError(f"session_id must match {SESSION_ID_RE.pattern!r} (got {session_id!r}).")
    return session_id


def resolve_under(root: str | Path, path: str | Path) -> str:
    """Resolve ``path`` and require it to stay at or below ``root``.

    Both sides are ``realpath``-resolved so ``..`` components and symlinks cannot
    escape the approved root. The returned value is the resolved absolute path.
    """
    root_real = os.path.realpath(os.fspath(root))
    candidate = os.path.realpath(os.fspath(path))
    try:
        common = os.path.commonpath([root_real, candidate])
    except ValueError as error:  # different drives / invalid mix on POSIX
        raise ValueError(f"Path {path!r} is not inside approved root {root!r}.") from error
    if common != root_real:
        raise ValueError(f"Path {path!r} is not inside approved root {root!r}.")
    return candidate


contain_path = resolve_under


def session_dir(session_id: str, *, workspace_dir: str = WORKSPACE_DIR) -> str:
    """Return the contained per-session directory under ``<workspace>/sessions``."""
    validate_session_id(session_id)
    sessions_root = os.path.join(workspace_dir, SESSIONS_SUBDIR)
    return resolve_under(sessions_root, os.path.join(sessions_root, session_id))
