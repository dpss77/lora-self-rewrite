# Discoverability and release metadata

**Purpose:** keep GitHub/search metadata consistent without stuffing the README. These settings are
not all stored in Git; apply them to the repository after the final review.

## GitHub repository About

Suggested description:

> Research harness for self-rewriting Qwen3.5-27B with zero-initialized rank-4 LoRA workspaces,
> H200 vLLM/Modal serving, exact caches, and held-out evaluation. PoC; compute wanted.

Suggested topics:

```
llm, large-language-models, lora, peft, self-rewriting, model-editing,
interpretability, qwen, vllm, modal, h200, reproducibility, research-software
```

Do not add topics that misrepresent the status (e.g. `production-ready`, `sota-results`).

## README/search terms

The first paragraph and “Start here” table intentionally include the discoverable terms:
self-rewriting, LoRA, PEFT, Qwen3.5-27B, vLLM, Modal, H200, held-out evaluation, reproducibility.
Keep those terms natural; do not keyword-stuff.

## Releases and tags

- Use annotated semantic-version tags after review: `git tag -a v0.1.0 -m "..."`.
- Create a GitHub Release for that tag with a short summary and a link to `CHANGELOG.md`.
- Tie any release to `CITATION.cff` and, if archived, a Zenodo DOI.
- Do not tag before the final review/commit; do not push tags until the repository is approved.

## Citation and research indexes

- `CITATION.cff` is the citation source.
- `codemeta.json` is for research-software indexes.
- After a release, archive it (Zenodo/figshare) and add the DOI to `CITATION.cff` and the README.
- The paper link is in `README.md`, `CITATION.cff`, and the paper spec.

## Badges

- Keep the paper, license, and status badges.
- Replace the static test badge with the CI badge after the first GitHub Actions run.
- Add a Python/uv badge only if it stays current.
- Do not add a coverage badge until coverage is measured in CI.

## Social preview

Use a simple preview with the title, “rank-4 LoRA workspace”, “Qwen3.5-27B / H200”, and
“PoC — compute wanted”. Do not imply a positive efficacy result.

## Not discoverability

- No fake stars, topics, keywords, or result claims.
- No DOI before archival.
- No release tag before the final reviewed commit.
