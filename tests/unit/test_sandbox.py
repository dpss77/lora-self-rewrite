import pytest

from lora_self_rewrite.sandbox import (
    safe_hasattr,
    safe_import,
    validate_sandbox_code,
)

pytestmark = pytest.mark.unit


def test_allows_ordinary_data_imports():
    validate_sandbox_code("import collections\nimport itertools\nprint(len([1]))")


def test_blocks_filesystem_and_eval():
    with pytest.raises(PermissionError):
        validate_sandbox_code("import os")
    with pytest.raises(PermissionError):
        validate_sandbox_code("from os import path")
    with pytest.raises(PermissionError):
        validate_sandbox_code("eval('1+1')")
    with pytest.raises(PermissionError):
        validate_sandbox_code("x = obj.__class__")


def test_blocks_relative_imports():
    with pytest.raises(PermissionError):
        validate_sandbox_code("from . import x")


def test_safe_import_blocks_forbidden_root():
    with pytest.raises(PermissionError):
        safe_import("os.path")
    with pytest.raises(PermissionError):
        safe_import("x", level=1)
    assert safe_import("collections") is not None


def test_safe_hasattr_never_reveals_dunders():
    class Obj:
        x = 1

    assert safe_hasattr(Obj(), "x") is True
    assert safe_hasattr(Obj(), "__class__") is False
    assert safe_hasattr(Obj(), "missing") is False


def test_blocks_model_and_network_loaders():
    blocked = [
        "import torch.hub",
        "from torch import hub",
        "from torch.hub import load",
        "torch.hub.load('repo', 'model')",
        "torch.load('x.pt')",
        "torch.save(obj, 'x.pt')",
        "model.load_state_dict({})",
        "save_file(tensors, 'x.safetensors')",
        "model.save_pretrained('/tmp/out')",
        "tokenizer.push_to_hub('me/model')",
        "AutoModel.from_pretrained('repo')",
        "load_dataset('repo')",
        "hf_hub_download('repo', 'file')",
        "urllib.request.urlopen('https://example.com')",
    ]
    for code in blocked:
        with pytest.raises(PermissionError):
            validate_sandbox_code(code)


def test_safe_import_blocks_torch_hub_submodule():
    with pytest.raises(PermissionError):
        safe_import("torch.hub")
    with pytest.raises(PermissionError):
        safe_import("torch", fromlist=["hub"])
    assert safe_import("torch") is not None
