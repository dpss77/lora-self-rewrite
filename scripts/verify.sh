#!/usr/bin/env bash
# CPU-only verification gate. Never launches Modal/GPU jobs.
# Uses the repo environment (uv or .venv), never PATH pytest/ruff.
set -euo pipefail
cd "$(dirname "$0")/.."

if command -v uv >/dev/null 2>&1; then
  PYTHON=(uv run --frozen --extra cpu --extra dev python)
  RUFF=(uv run --frozen --extra cpu --extra dev ruff)
elif [[ -x .venv/bin/python && -x .venv/bin/ruff ]]; then
  PYTHON=(.venv/bin/python)
  RUFF=(.venv/bin/ruff)
else
  echo "error: need uv or a repo .venv containing pytest+ruff" >&2
  exit 1
fi

# Keep bytecode/cache writes out of the working tree where practical.
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="${TMPDIR:-/tmp}/lora-self-rewrite-pycache"

echo "[1/6] pytest (unit)"
"${PYTHON[@]}" -m pytest -p no:cacheprovider -q -m unit

echo "[2/6] pytest (contract)"
"${PYTHON[@]}" -m pytest -p no:cacheprovider -q -m contract

echo "[3/6] pytest (integration)"
"${PYTHON[@]}" -m pytest -p no:cacheprovider -q -m integration

echo "[4/6] ruff check"
"${RUFF[@]}" check --no-cache src tests scripts modal_app.py

echo "[5/6] ruff format --check"
"${RUFF[@]}" format --check --no-cache src tests scripts modal_app.py

echo "[6/6] in-memory compile"
"${PYTHON[@]}" - <<'PY'
import pathlib

paths = [pathlib.Path("modal_app.py")]
for pattern in ("src/lora_self_rewrite/*.py", "scripts/*.py", "tests/**/*.py"):
    paths.extend(sorted(pathlib.Path(".").glob(pattern)))
for path in paths:
    compile(path.read_text(), str(path), "exec")
print(f"compiled {len(paths)} files")
PY

echo "OK: CPU verification passed"
