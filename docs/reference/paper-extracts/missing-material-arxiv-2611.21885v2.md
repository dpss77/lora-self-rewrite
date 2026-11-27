Here is the missing material from **arXiv:2611.21885v2** (“Language Models Can Rewrite Their Own Weights”), pulled from the HTML/PDF.

### Table 4 (Sec. D.1) — Settings of every session (first protocol)
| Session | Edited | Editor | Passes | Code runs | Turns | Time | Starting diagnosis |
|---------|--------|--------|--------|-----------|-------|------|--------------------|
| 2B edits itself | 2B | itself | 20 | 200 | 160 | 3.5 h | 256 |
| 4B edits itself | 4B | itself | 20 | 200 | 160 | 3.5 h | 256 ‡ |
| 9B edits itself | 9B | itself | 20 | 200 | 160 | 3.5 h | 256 |
| 27B edits itself, first session | 27B | itself | 20 | 200 | 160 | 3.5 h | 256 |
| 27B edits itself, second session § | 27B | itself | 20 | 200 | 160 | 2.5 h | 256 |
| 27B repairs its damaged copy | damaged copy | 27B base model | 20 | 200 | 160 | 2.5 h † | 64, damaged copy |
| 2B repairs its damaged copy | damaged copy | 2B base model | 20 | 200 | 160 | 2.5 h | 64, damaged copy |
| 4B repairs its damaged copy | damaged copy | 4B base model | 20 | 200 | 160 | 2.5 h | 64, damaged copy |
| 9B repairs its damaged copy | damaged copy | 9B base model | 20 | 200 | 160 | 2.5 h | 64, damaged copy |

- Models: Qwen3.5 family.  
- Passes / code runs / turns = budgets \(B=20\), \(C=200\), \(T_{\max}=160\) (model is **not** told the turn limit).  
- † includes downtime from one framework error.  
- ‡ diagnosis pre-computed.  
- § task states a 4,096-token answer budget (starting diagnosis + passes use it).  
- Second protocol settings are in Sec. E.9.

### Sec. D.4 — The Task Prompt (paraphrase of the system message)
The paper does **not** print a fully verbatim system prompt; it gives a structured paraphrase (the actual message appears as a figure in the HTML/PDF). Only the first sentence (architecture description) changes by model size. Angle-bracket values are filled per session from Tabs. 4 / 9.

**Who it is.**  
The weights it is about to edit are its own; the session itself runs on them. It is asked to become better at competition mathematics by changing those weights directly, and told that nothing will train it.

**How it will be judged.**  
After the session, its weights with and without the edit answer unseen mathematics and science problems with the answer settings of Tab. 5; only the final boxed answer after the thinking is graded, and an answer whose thinking is still open when the budget runs out earns no credit. Only the values of its tensors may change.

**Its starting diagnosis.**  
How it did on \(\langle n\rangle\) practice problems before the session: its accuracy, how many answers ran out of tokens and how many of those end in repetitive text, and how often its finished answers were right. Every graded answer is in a file it can read, `WORK/census.jsonl`, with the same fields as the rows of later generation passes.

**Its tools.**  
(1) A Python session on a CPU that keeps its state between runs, with gradients switched off, the usual numerical libraries, the names and shapes of its tensors, functions to load a tensor and to save, drop or list edits, its tokenizer, the practice problems, loaders for the results of its forward passes, and four small helpers (the output matrices of a layer, a unit vector, the unit difference of two group means, and \(W - \alpha uu^\top W\)).  
(2) The practice problems with reference answers, none from any evaluation.  
(3) A budget of \(\langle\)passes\(\rangle\) forward passes, each requested as a small JSON object: either a generation on chosen practice problems with its weights without the edit, with the edit, or both on the same seeds (which reports the difference in accuracy between the two, with its standard error), or a hidden-state read, in which its weights are run over texts it chooses and return hidden states after chosen layers, MLP activations and token log-probabilities. A request that fails costs nothing.

**How to work.**  
Look at its own wrong and unfinished answers first; find the components behind the failure and show causally, at several strengths, that changing them changes the behaviour; compare its weights with and without the edit on the same problems; keep an edit only if it helps on problems not used to choose it. Some edits are out of scope (the final normalisation, the token embeddings and unembedding, and multiplying a whole tensor by one number), and so is making itself stop thinking early regardless of the problem. It should keep notes in a file that is shown to it again if the conversation restarts.

**What to aim for.**  
A repair of the failure in its starting diagnosis, at its cause: on problems not used to choose the edit, clearly fewer answers lost to that failure, and accuracy not significantly lower. If no edit meets both, it should submit none.

**Time and rules.**  
The session has a wall-clock limit and every result shows the time used. No gradients, optimizers or training of any kind, no running the network inside the Python session, no internet.

**Each reply.**  
A few sentences on what the last result showed and what comes next, then exactly one block of code or one forward request. When the passes run out, a final answer describing the edit and the evidence; whatever is saved as its edit at that point is submitted.

**Repair-session note (appended):**  
The weights you edit in this session are a copy of your own weights in which some values have been corrupted, as a memory or hardware fault might do. You are not told which values, where, or how. “original” in every forward pass, `load_original()` and the starting diagnosis above all refer to this damaged copy. You yourself run on your undamaged weights in this session, so the damage does not affect your own thinking. Your goal is to find the damage and repair it: the evaluation compares your edited copy with the damaged copy on held-out problems.

### Appendix C / Algorithm details (relevant to tools, notes.md, 90k compaction)
From Alg. 1–2:

- Conversation history is compacted when it exceeds **90k tokens**:  
  `h ← (Task(c₀), notes.md, b, c, current edit, log of passes, o)`  
  i.e. the conversation is restarted from the original task text + the model’s own `notes.md` + budget counters + current edit + log of passes + the last observation.

- Forward / read tool: a single JSON request per turn; can be generation (weights = unedited / edited / both) or a hidden-state read (chosen texts → hidden states after chosen layers, MLP activations, token log-probs). Failed requests cost nothing. Max 20 forward passes.

- Python session: stateful on CPU, gradients disabled; refuses gradients/optimizers/model loaders/inference engines/subprocesses/network. Helpers include the rank-1 ablation \(W - \alpha uu^\top W\).

- No explicit “4-candidate batching prompt” appears in the indexed text; generation is controlled by the JSON request schema (problems, n samples, token limit \(L'\), weights flag). The paired-test logic (Alg. 3) runs both weight sets on the same seeds and returns accuracy delta + SE, loop statistics, etc.

If you need the exact figure text of the system prompt (the boxed version in D.4) or the full JSON schema for the read/generate tool, the cleanest source is still the PDF page for Sec. D.4 / Alg. 3; the HTML renders that prompt as an image. Let me know if you want me to pull any other specific subsection (E.9 second protocol, exact helper signatures, Tab. 5 answer settings, etc.).