"""Stateful CPU Python sandbox policy (Alg. 3 / Sec. E.9).

This is defence-in-depth, not a security boundary: a determined model can still
reach dangerous behaviour through exposed third-party libraries. The environment
is scrubbed of HF tokens and the Modal Volume is the only intended IO surface.

The AST validator and builtins surface are shared by the session loop and any
future offline replay tooling, so they live here as pure code.
"""

from __future__ import annotations

import ast
import builtins

__all__ = [
    "FORBIDDEN_IMPORTS",
    "FORBIDDEN_MODULE_PREFIXES",
    "BLOCKED_ATTRS",
    "BLOCKED_NAMES",
    "SAFE_BUILTINS",
    "safe_import",
    "safe_hasattr",
    "validate_sandbox_code",
]

FORBIDDEN_IMPORTS = {
    # process / network / filesystem escape hatches
    "subprocess",
    "socket",
    "urllib",
    "requests",
    "httpx",
    "pty",
    "os",
    "sys",
    "builtins",
    "importlib",
    "ctypes",
    "pathlib",
    "shutil",
    "glob",
    "io",
    "tempfile",
    "multiprocessing",
    "threading",
    "signal",
    "resource",
    "fcntl",
    "posix",
    "pickle",
    "marshal",
    "shelve",
    "webbrowser",
    "http",
    "ftplib",
    "smtplib",
    "ssl",
    "asyncio",
    "concurrent",
    # heavy / privileged ML runtimes (the sandbox is CPU data-shaping only)
    "transformers",
    "vllm",
    "huggingface_hub",
    "tokenizers",
}

# Model/network loaders are also reachable through otherwise-safe packages
# (e.g. ``torch.hub``), so reject the dotted module path explicitly.
FORBIDDEN_MODULE_PREFIXES = (
    "torch.hub",
    "huggingface_hub",
    "transformers",
    "vllm",
    "datasets.load_dataset",
)

BLOCKED_ATTRS = {
    "backward",
    "optim",
    "system",
    "popen",
    "Popen",
    "check_output",
    "check_call",
    "spawn",
    "fork",
    "execv",
    "execl",
    "execve",
    "environ",
    "putenv",
    "getenv",
    # model/network/file exfiltration surfaces (defence-in-depth)
    "hub",
    "load",
    "save",
    "from_pretrained",
    "save_pretrained",
    "push_to_hub",
    "upload_file",
    "upload_folder",
    "create_repo",
    "snapshot_download",
    "hf_hub_download",
    "load_dataset",
    "urlopen",
    "urlretrieve",
    "load_state_dict",
    "save_file",
    "load_file",
    "savez",
    "savez_compressed",
    "savetxt",
    "loadtxt",
    "tofile",
    "fromfile",
    "__globals__",
    "__subclasses__",
    "__builtins__",
    "__import__",
    "__loader__",
    "__spec__",
    "__code__",
    "__dict__",
    "__getattribute__",
    "__getattr__",
    "__setattr__",
    "__delattr__",
    "__reduce__",
    "__reduce_ex__",
    "__mro__",
    "__bases__",
    "__base__",
    "__class__",
    "__closure__",
    "__wrapped__",
}

BLOCKED_NAMES = {
    "eval",
    "exec",
    "compile",
    "getattr",
    "setattr",
    "delattr",
    "globals",
    "locals",
    "vars",
    "__import__",
    "__builtins__",
    "load_dataset",
    "hf_hub_download",
    "snapshot_download",
    "urlopen",
    "urlretrieve",
    "save_file",
    "load_file",
    "savez",
    "savez_compressed",
    "savetxt",
    "loadtxt",
    "tofile",
    "fromfile",
}


def _is_forbidden_module(name: str | None) -> bool:
    """Return whether an import path hits a blocked root or dotted module."""
    if not name:
        return False
    root = name.split(".")[0]
    if root in FORBIDDEN_IMPORTS:
        return True
    return any(
        name == prefix or name.startswith(f"{prefix}.") for prefix in FORBIDDEN_MODULE_PREFIXES
    )


def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    """Guarded replacement for ``__import__`` exposed to sandbox code."""
    if level != 0:
        raise PermissionError("Relative imports are not allowed in the sandbox.")
    if _is_forbidden_module(name):
        raise PermissionError(f"Blocked import '{name}' in sandbox.")
    for item in fromlist or ():
        if _is_forbidden_module(f"{name}.{item}"):
            raise PermissionError(f"Blocked import '{name}.{item}' in sandbox.")
    return builtins.__import__(name, globals, locals, fromlist, level)


def safe_hasattr(obj, attr) -> bool:
    """Boolean-only attribute check; never returns a value or dunder internals."""
    if not isinstance(attr, str) or attr.startswith("__") or attr in BLOCKED_ATTRS:
        return False
    try:
        return hasattr(obj, attr)
    except Exception:  # noqa: BLE001 - hasattr must never raise in sandbox code
        return False


SAFE_BUILTINS = {
    "__import__": safe_import,
    "__build_class__": builtins.__build_class__,
    "abs": abs,
    "all": all,
    "any": any,
    "bin": bin,
    "bool": bool,
    "bytes": bytes,
    "callable": callable,
    "chr": chr,
    "classmethod": classmethod,
    "complex": complex,
    "dict": dict,
    "dir": dir,
    "divmod": divmod,
    "enumerate": enumerate,
    "filter": filter,
    "float": float,
    "format": format,
    "frozenset": frozenset,
    "hash": hash,
    "hasattr": safe_hasattr,
    "hex": hex,
    "id": id,
    "int": int,
    "isinstance": isinstance,
    "issubclass": issubclass,
    "iter": iter,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "next": next,
    "object": object,
    "oct": oct,
    "ord": ord,
    "pow": pow,
    "print": print,
    "property": property,
    "range": range,
    "repr": repr,
    "reversed": reversed,
    "round": round,
    "set": set,
    "slice": slice,
    "sorted": sorted,
    "staticmethod": staticmethod,
    "str": str,
    "sum": sum,
    "super": super,
    "tuple": tuple,
    "type": type,
    "zip": zip,
    "True": True,
    "False": False,
    "None": None,
    "Exception": Exception,
    "ArithmeticError": ArithmeticError,
    "AssertionError": AssertionError,
    "AttributeError": AttributeError,
    "IndexError": IndexError,
    "KeyError": KeyError,
    "MemoryError": MemoryError,
    "NotImplementedError": NotImplementedError,
    "OverflowError": OverflowError,
    "RuntimeError": RuntimeError,
    "StopIteration": StopIteration,
    "TypeError": TypeError,
    "ValueError": ValueError,
    "ZeroDivisionError": ZeroDivisionError,
}


def validate_sandbox_code(code: str) -> None:
    """AST inspection blocks disallowed imports/ops without false-positiving comments."""
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _is_forbidden_module(alias.name):
                    raise PermissionError(f"Blocked import '{alias.name}' in sandbox.")
        elif isinstance(node, ast.ImportFrom):
            if node.level != 0:
                raise PermissionError("Relative imports are not allowed in sandbox.")
            module = node.module or ""
            if _is_forbidden_module(module):
                raise PermissionError(f"Blocked import '{module}' in sandbox.")
            for alias in node.names:
                if alias.name in BLOCKED_ATTRS or _is_forbidden_module(f"{module}.{alias.name}"):
                    raise PermissionError(f"Blocked import '{module}.{alias.name}' in sandbox.")
        elif isinstance(node, ast.Attribute):
            if node.attr in BLOCKED_ATTRS:
                raise PermissionError(f"Blocked attribute '.{node.attr}' in sandbox.")
        elif isinstance(node, ast.Name):
            if node.id in BLOCKED_NAMES:
                raise PermissionError(f"Blocked name '{node.id}' in sandbox.")
