import os
from pathlib import Path

import pytest

from lora_self_rewrite.paths import (
    contain_path,
    resolve_under,
    session_dir,
    validate_session_id,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("value", ["run_01", "Ab-C.9", "a", "a" * 64])
def test_valid_session_ids_accepted(value):
    assert validate_session_id(value) == value


@pytest.mark.parametrize(
    "value",
    ["", "../escape", "a/b", ".hidden", "-leading", "a b", "a\nb", "a" * 65],
)
def test_unsafe_session_ids_rejected(value):
    with pytest.raises(ValueError, match="session_id"):
        validate_session_id(value)


def test_session_dir_is_contained_under_sessions_root(tmp_path):
    resolved = session_dir("run_01", workspace_dir=str(tmp_path))
    assert resolved == os.path.realpath(tmp_path / "sessions" / "run_01")


def test_session_dir_rejects_traversal(tmp_path):
    with pytest.raises(ValueError, match="session_id"):
        session_dir("../../etc", workspace_dir=str(tmp_path))


def test_resolve_under_allows_root_and_descendants(tmp_path):
    root = tmp_path / "root"
    child = root / "a" / "b.json"
    assert resolve_under(root, child) == os.path.realpath(child)
    assert resolve_under(root, root) == os.path.realpath(root)


def test_resolve_under_rejects_sibling_and_traversal(tmp_path):
    root = tmp_path / "root"
    with pytest.raises(ValueError, match="not inside"):
        resolve_under(root, tmp_path / "sibling")
    with pytest.raises(ValueError, match="not inside"):
        resolve_under(root, root / ".." / "escape")


def test_contain_path_alias_behaves_like_resolve_under(tmp_path):
    root = tmp_path / "root"
    assert contain_path(root, root / "x") == resolve_under(root, root / "x")


def test_heldout_entrypoint_contains_dataset_adapter_and_output_paths():
    source = (
        Path(__file__).resolve().parents[2] / "src" / "lora_self_rewrite" / "eval_heldout.py"
    ).read_text()
    assert "resolve_under(WORKSPACE_DIR, dataset_path)" in source
    assert "resolve_under(WORKSPACE_DIR, out_path)" in source
    assert "adapter_dir = resolve_under(" in source


def test_session_entrypoint_validates_and_contains_session_id():
    source = (
        Path(__file__).resolve().parents[2] / "src" / "lora_self_rewrite" / "session.py"
    ).read_text()
    assert "validate_session_id(session_id)" in source
    assert "session_dir_for(session_id" in source
