# Module map

Dependency direction: pure modules -> GPU/port modules -> thin entrypoints.
Pure modules never import Modal/vLLM/torch at module scope.

## Pure (CPU-tested)

| Module | Responsibility |
| :--- | :--- |
| `constants.py` | single source of truth for protocol/runtime constants |
| `protocol_shared.py` | prompt suffix, thinking strip, grading, loop definition |
| `seeds.py` | deterministic `(pid, sample_idx)` seed |
| `stats.py` | McNemar, bootstrap CI, sign-flip, paired stats |
| `caches.py` | exact-batch signature, base cache, content-addressed LoRA ids |
| `model_layout.py` | 128-target discovery, PEFT keys, empty-LoRA init |
| `sandbox.py` | forbidden imports/attrs/names, AST validation, safe builtins |
| `datasets.py` | DeepScaleR scrubbing, fixed splits, held-out slice |
| `manifest.py` | manifest schema, run ids, atomic writes, checksums |
| `eval_common.py` | whole-problem sharding, cache tag |

## GPU / ported (not yet re-validated on hardware)

| Module | Responsibility |
| :--- | :--- |
| `modal_wiring.py` | app, image, volume, secrets, commit/reload helpers |
| `instrumentation.py` | metrics, fingerprints, bounded stdout |
| `hf_worker.py` | spawn-isolated HF hidden-state/MLP/logprob extraction |
| `sandbox_runtime.py` | `bind_sandbox_helpers` (torch/safetensors) |
| `engine.py` | `QwenSelfRewriteEngine` vLLM service |
| `session.py` | `run_self_rewrite_session` loop/budgets/resume |
| `eval_heldout.py` | H200-only held-out evaluator (unrun) |
| `../modal_app.py` | local Modal entrypoint |

## Scripts

`scripts/` contains run automation; see `docs/05-standards/STD-0001-output-contract.md`.
