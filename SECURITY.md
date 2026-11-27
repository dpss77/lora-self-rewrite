# Security policy

## Supported versions

The `0.1.x` line is the supported release series. Security fixes target `main`
and the latest `0.1.x` release; older snapshots are not maintained.

## Reporting a vulnerability

Report privately through GitHub Security Advisories:

<https://github.com/dnpardo/lora-self-rewrite/security/advisories/new>

Do not open a public issue, and do not include credentials, private model or
dataset contents, or raw logs in the report. No dedicated security email is
published; the advisory channel is the supported private route.

## Secrets

- `.env` is for local secrets only. It is gitignored and must never be
  committed. Start from [`.env.example`](.env.example).
- Hugging Face credentials (`HF_TOKEN`) come from the environment or a private
  Hugging Face secret. Do not bake them into images, configs, notebooks, or
  repository files.
- Modal credentials and `HF_TOKEN` for remote runs belong in a Modal Secret
  (for example, `HF_SECRET_NAME` points at a secret created outside Git). Do not
  put secret values in `modal_app.py`, command history, or committed TOMLs.
- Never put credentials in prompts, model input/output, logs, manifests,
  transcripts, checkpoints, or error messages. Redact tokens before sharing any
  diagnostic output.

## Sandbox

The model-written sandbox is **defence-in-depth, not a security boundary**. It
runs with CPU-only state, bounded wall time and stdout, an AST/attribute
denylist, and runtime guards, but it must not be relied on to contain hostile
code. Treat model-authored code and its outputs as untrusted, and run the real
pipeline only in the approved environment.

## Repository hygiene

Do not commit raw logs, transcripts, weights, adapters, datasets, caches,
checkpoints, or credentials. Curated records and small manifests belong in Git;
raw evidence stays in gitignored run and artifact directories. See
[`docs/05-standards/STD-0001-output-contract.md`](docs/05-standards/STD-0001-output-contract.md)
for the artifact boundary.

## Scope

This policy covers the public code and documentation in this repository. It does
not cover Hugging Face, Modal, vLLM, upstream models/datasets, or third-party
infrastructure; report those issues to the relevant provider.
