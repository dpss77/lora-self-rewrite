# Runbook

## 0. Preconditions

- Operator approval for any Modal/H200 command.
- H200 is pinned in code (`GPU_SPEC=["H200"]`); do not add an H100 fallback.
- HF/Modal secrets are in the environment; never commit them.

## 1. Build the dataset (CPU, no GPU)

```bash
uv run python scripts/build_dataset.py --out datasets/deepscaler_scrubbed.json
```

Expected: 7/7 math sources, `gpqa_ok=false` (gated), `practice=3000`,
`reserved=300`, `heldout=1000`.

## 2. Create a run record

```bash
RUN=$(uv run python scripts/new_run.py \
  --config configs/experiments/S02-DS-SEED00.toml \
  --runs-dir runs --gpu H200 --code-sha <git-sha> \
  --file-sha256 <modal_app-sha256> --model-revision <hf-revision>)
```

## 3. Execute (GPU — approval required)

```bash
modal run modal_app.py -- --session-id <id> --dataset deepscaler \
  --diag-n 256 --val-n 256 --val-samples 2 --max-turns 160
```

## 4. Resume after preemption

1. Inspect `ami`/Modal logs and the run's `60_LOGS/`.
2. Do **not** treat an existing `WORK` directory as a dirty fresh session.
3. Resume must restore the prior pass count, turn count, wall time, and transcript,
   or explicitly mark the run as a protocol deviation.

## 5. Finalize

```bash
uv run python scripts/finalize_run.py --run-dir runs/<run_id> \
  --status withdrawn|submitted --decision WITHDRAW|SUBMIT
uv run python scripts/summarize_run.py --run-dir runs/<run_id>
uv run python scripts/index_runs.py
uv run python scripts/check_artifacts.py
```

## 6. Held-out evaluation (GPU — approval required)

```bash
modal run src/lora_self_rewrite/eval_heldout.py -- \
  --dataset-path /workspace/datasets/deepscaler_scrubbed.json \
  --adapter-dir /workspace/sessions/<session_id>/WORK/submitted_lora \
  --k-shards 5 --smoke-problems 98
```

Start with `--smoke-problems`; run the full 490×8 only after the base cache and
one shard are verified.

## 7. Always

- `uv sync --frozen --extra cpu --extra dev`, then `./scripts/verify.sh` before claiming a code change is done.
- No commit/push without explicit authorization.
