# Paper reference: arXiv:2611.21885v2

**Status:** read-only reference. Update only when the paper changes or a verified extraction is corrected.
This file separates **paper facts** from **this project's v1 engineering choices**. It is the first
read for agents and reviewers. Protocol invariants and implementation decisions link here rather
than restating paper claims in code.

## Source

- Paper: *Language Models Can Rewrite Their Own Weights*
- Identifier: [`arXiv:2611.21885v2`](https://arxiv.org/abs/2611.21885v2)
- HTML: <https://arxiv.org/html/2611.21885v2>
- PDF: <https://arxiv.org/pdf/2611.21885v2>

Raw HTML/PDF are third-party artifacts. Keep local copies under ignored `reference/paper/`; do
not commit or redistribute them. The structured extracts are in
[`docs/reference/paper-extracts/`](../reference/paper-extracts/).

## Model and scale

- Target: `Qwen/Qwen3.5-27B`, thinking enabled, 64 layers, 3:1 hybrid Gated DeltaNet + full
  attention.
- The paper reports that smaller models fail to improve or damage themselves; 27B is the
  minimum viable scale for positive self-modification in its experiments.
- This project pins the model revision `81a287ca484a5e71148f5beb7f367bb5286d70f9`.

## Table 4 — session budgets

### First Protocol

- `B=20` forward passes (`generate` + `read`).
- `C=200` CPU code executions, each at most 600 s.
- `T_max=160` turns; the model is **never told** the turn limit.
- Wall limit: 3.5 h at 16k tokens, 2.5 h at 4k tokens.
- Starting diagnosis: 256 problems (64 for damaged-copy repair).

### Second Protocol (Sec. E.9)

- `B=20`, `C=200`, `T_max=160`.
- Wall limit: 3.0 h; starting diagnosis is not counted.
- Up to four saved edits may be evaluated side-by-side against the unedited base on identical
  problems and seeds.
- Reserved validation: 256 reserved practice problems × 2 samples at 4,096 tokens; the model
  replies `SUBMIT` or `WITHDRAW`. This check does not consume the 20-pass budget.

## Table 5 — datasets, sampling, grading, and tools

- Dataset: DeepScaleR, with 3,000 practice and 1,000 held-out examples, scrubbed against
  MATH-500, AIME 2024, AIME 2025, HMMT Feb 2025, AMC 2023, OlympiadBench, and GPQA-Diamond.
- Prompt: thinking enabled; ends with
  `"Please reason step by step, and put your final answer within \boxed{}."`.
- Sampling: `temperature=0.6`, `top_p=0.95`, `top_k=20`; deterministic ID-hash seeds.
- Token cap: 4,096 in the Second Protocol; 16,384 for the First Protocol / held-out evaluation.
- Grading: the last `\boxed{}` after a closed `</think>` must pass Math-Verify. Unclosed thinking
  receives zero credit.
- Loop definition: cut at the token limit **and** the last 6,000 characters compress to
  `zlib < 15%`.
- `generate`: at most 512 answers in the First Protocol (256 per paired arm) and 128 total answers
  in the Second Protocol.
- `read`: at most 256 texts and 8,192 tokens; modes `answer_mean`, `every_k`, `last_k`, `all`;
  outputs hidden states, MLP activations, logprobs.
- Sandbox: stateful CPU, gradients off, at most 600 s, at most 10,000 characters of stdout, no
  network/subprocesses/trainers.
- Forbidden edits: input embeddings, final norm, LM head, whole-tensor scalar multiplication, or
  forcing early stopping regardless of the problem.
- Held-out evaluation: 490 deduplicated problems × 8 samples at 16,384 tokens, problem-clustered
  **one-sided** sign-flip permutation test (100,000 flips, `p < 0.05`).

## Sec. D.4 and Algorithms 1–3

- D.4 contains the task prompt; the paper's boxed figure is the authoritative wording.
- Built-in helpers: `layer outputs`, `unit`, `mean_diff`, and `W - alpha * u u^T W`.
- `WORK/census.jsonl` stores diagnosis rows; `notes.md` is the model's session summary.
- Failed requests cost zero passes.
- Compaction (Alg. 1–2) keeps `(Task(c_0), notes.md, b, c, current edit, pass log, last
  observation)` and is described at 90k tokens in the paper.
- `read` span options are listed under Table 5.

## Empirical paper baselines

These are **paper-reported** values, not this project's results:

- 4,096-token starting diagnosis: **80.5% cut off**, **0 verbatim loops**, **96% of finished
  answers correct**.
- Second Protocol success: **3/8 sessions submitted**, **5/8 withdrew**; three rejected all
  candidates in-session and two withdrew after a `+0.2` gate.
- Known positive mechanisms:
  - Session 1 (First Protocol): layer 8 output projections, `alpha=0.05`, `+1.1%`, `p=0.038`.
  - Session 2 (Second Protocol): layer 63 output projections, `+4.0%`, reaching 61.3%.
  - Session 8 (Second Protocol): `dt_bias += 0.5` across all 48 Gated DeltaNet layers,
    `+6.6%` (57.2% → 63.8%, `p < 10^-5`).
- Damaged-copy repair: layer 30 `mlp.down_proj` rows 3068–3075 corrupted with
  `sigma_noise = 200 * sigma_matrix`; repaired with `15x MAD` clipping (`diag_n=64`).

## Paper vs. this v1 harness

| Area | Paper | This repo (v1) |
| :--- | :--- | :--- |
| Edit surface | Checkpoint edits, including `dt_bias += 0.5` | Zero-initialized rank-4 PEFT LoRA over 128 output projections |
| 1D DeltaNet `dt_bias`/`A_log` | Session 8 gain | Deferred to v2 |
| Damaged-copy repair | Repair benchmark | Out of v1 scope |
| Compaction | 90k tokens in the paper | 48k with `max_model_len=65,536`; 90k requires a separate 114,688+ window change |
| Loop denominator | 6,000 characters described | `chars` mode: `text[-6000:]` → UTF-8 → zlib level 6 |
| Held-out slice | 490 deduplicated problems | `heldout[500:990]` = 490 fixed projects slice |
| Validation | 256 × 2 reserved problems | Same protocol for DeepScaleR; `math500` Phase-1 used 64 × 2 because its reserved pool is 100 |
| Hardware | Paper's H200 setup | H200-only; cache tags are hardware-scoped |
| Prefix caching | Not the project's invariant | `enable_prefix_caching=False` required |

## What is not printed verbatim

- The exact boxed D.4 figure text (only a paraphrase is available in the HTML extraction).
- Exact JSON field names for four-candidate `generate` and `read` requests. This project's JSON is
  designed to match Alg. 3 semantics, not copied from a paper schema.
- Exact Python helper function names.
- The exact Table 1–3 held-out numbers at 16,384 vs 4,096 tokens; verify against the PDF before
  citing them.

## How to use this file

- `docs/03-protocol/INVARIANTS.md` is the project's enforceable rulebook.
- `docs/07-decisions/` records deliberate deviations from paper values.
- Do not change paper values here to match code. If v1 intentionally differs, record it in the
  gap table and the corresponding ADR.
