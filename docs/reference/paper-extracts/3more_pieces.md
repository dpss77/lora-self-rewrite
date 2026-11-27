 The only remaining pieces worth pulling from arXiv:2611.21885v2 before Phase 4 (full SOTA replication) are these 3 narrow items:
Sec. E.9 (Second Protocol): Specifically how the 4 candidate slots work in the JSON generate call and the exact wording of the reserved-validation confirmation prompt.
Alg. 3 / Appendix C read schema: Whether read supports selecting specific token indices/positions (since returning full [seq_len, 5120] tensors across 64 texts would be huge) and exact format of MLP activations / token log-probs.
Table 5: Exact held-out evaluation settings (benchmark list + sampling/token limits).
---

Here are the three remaining pieces from **arXiv:2611.21885v2**, extracted as cleanly as the text allows.

### 1. Sec. E.9 — Second Protocol (4-candidate slots + reserved-validation confirmation)

**How the 4 candidate slots work**  
In the second protocol a single forward/generate pass can test **up to four saved edits side-by-side** against the unedited model, on the exact same problems and seeds.  
(The paper states this as a framework change: “a forward pass may test up to four saved edits side by side against the unedited model, on the same problems and seeds.”)  
The task text itself additionally tells the model “to test the parts of an edit separately and keep only the parts its tests support.”  

No full JSON schema for the multi-edit call is printed, but the mechanics are:
- The model saves candidate edits (via the usual Python `save_edit` interface).
- A generate request can reference up to four of those saved edits in one pass.
- The framework runs the unedited model + the (up to) four candidates on the same seeds and returns the usual paired-test statistics for each.

**Reserved-validation confirmation**  
Before any edit is accepted, the framework runs a one-time check that the session cannot see or influence:
- 256 of the 300 reserved practice problems,
- 2 answers each,
- at the 4 096-token budget,
- same seeds for edited vs. unedited.

After the framework returns the result of that check (accuracy delta + SE), **the model answers simply `SUBMIT` or `WITHDRAW`**.  
A withdrawn edit is discarded and never evaluated on the held-out set. The check itself does not count against the session’s 20 forward-pass budget. Sessions otherwise start from the base weights, use the same 4 096-token task text as the “second session” of the first protocol, have a 3.0 h wall-clock limit (starting diagnosis not counted), and differ only in the random seed of the model’s own turns.

### 2. Alg. 3 / Appendix C — Read schema (token selection + return format)

Exact description of a **hidden-state read**:

> A read request, a hidden-state read, runs \(\theta\) or \(\theta\oplus\Delta\) over up to 256 texts, each cut at 8 192 tokens, without sampling: texts the model writes, or answers from the starting diagnosis or from an earlier pass. It returns the hidden states after chosen layers, the MLP activations or the token log-probabilities, **averaged over the positions of the answer or at every \(k\)-th position, the last \(k\) positions or all of them**.

So yes — the read tool **does** support selecting token positions / indices:
- average over the answer span,
- every \(k\)-th token,
- the last \(k\) tokens,
- or every token.

Return types (chosen by the request):
- hidden states after selected layers (the vector a layer passes to the next, one per selected token),
- MLP activations (values inside a layer’s MLP block),
- token log-probabilities.

A failed request costs nothing. The generation side of Alg. 3 is the familiar paired test (`"weights": "both"`), limited in the second protocol to ≤128 answers total (≤512 in the first protocol).

### 3. Table 5 — Exact held-out / answer settings

**Table 5** (settings shared by all first-protocol sessions; second protocol changes are listed in E.9):

| Item | Setting |
|------|---------|
| **Practice set** | 3 000 problems from DeepScaleR after removing every problem that overlaps MATH-500, AIME 2024/2025, HMMT Feb 2025, AMC 2023, OlympiadBench or GPQA-Diamond. |
| **Answer settings** (starting diagnosis + every generation pass) | Thinking on. Prompt ends with “Please reason step by step, and put your final answer within `\boxed{}`.” Sampling: temperature 0.6, top-20, nucleus 0.95. Token budget 16 384 new tokens (4 096 in the 27B second session / second-protocol sessions), or fewer if a pass requests it (minimum 16). |
| **Grading** | Last `\boxed{}` after the thinking, checked with Math-Verify. No credit if thinking is still open at the budget. |
| **Looping answer** | Answer cut at the token limit whose last 6 000 characters compress (zlib) to <15 % of their size. |
| **Seeds** | Deterministic from problem identifier (identical with/without edit and across all passes of a session). |
| **Generation pass** | Up to 512 answers, or 256 each with/without the edit when both are run. (Second protocol: ≤128 total.) |
| **Hidden-state read** | Up to 256 texts, each ≤8 192 tokens; returns hidden states / MLP activations / token log-probs at the selected positions (see above). |
| **Code runs** | Stateful CPU Python, gradients off, ≤600 s, output truncated at 10 000 characters. |
| **Edits** | Any tensor of matching shape with finite values, **except** input embedding, final norm, and output (lm_head) layer; also refuses a pure scalar multiple of a whole tensor. |

**Held-out evaluation** (the fixed-in-advance protocol used for all significance claims):
- Problems: held-out items 500–999 after removing the 10 that repeat a practice or earlier held-out question → **490 problems**.
- 8 answers per problem at **16 384 tokens**, same seeds for base vs. edited.
- Primary test: one-sided sign-flip permutation test over the 490 per-problem accuracy differences (100 000 flips); significance only if \(p<0.05\).

These three items complete the public specification needed for a faithful re-implementation of both protocols.