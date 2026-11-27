from __future__ import annotations

# Mechanically ported from the donor `modal_app.py` (SHA-256
# fe71f74b0cebe933c155e03f59941c62ce402d92ca31ac2ce4e038c9b23dcbc2).
# Ported for structural parity; GPU behaviour must be re-validated before use.
import json
import os
import time

from .constants import MODEL_REVISION


# ---------------------------------------------------------------------------
# 2. Spawn-Isolated HF Extraction Worker (Alg. 3: Hidden States, MLP, Logprobs)
# ---------------------------------------------------------------------------
def _select_span(
    seq_tensor,
    start_idx: int,
    token_mode: str,
    k: int,
    out_dtype=None,
):
    """
    Slices tensor [1, seq_len, ...] (or [1, seq_len]) over answer span [start_idx:]
    and applies Alg. 3 token selection: answer_mean | every_k | last_k | all.

    ``out_dtype`` is applied *after* slicing so the Volume never stores fp32 features
    (the old ``.float()`` in each hook could balloon a 256x8192x5k read to tens of GiB).
    Logprobs stay fp32; hidden states / MLP activations travel as bf16.
    """
    span = seq_tensor[:, start_idx:]
    if span.shape[1] == 0:
        span = seq_tensor[:, -1:]

    if token_mode in ("answer_mean", "mean"):
        out = span.mean(dim=1).squeeze(0)
    elif token_mode == "every_k":
        step = max(1, int(k))
        out = span[:, ::step].squeeze(0)
    elif token_mode == "last_k":
        count = max(1, int(k))
        out = span[:, -count:].squeeze(0)
    elif token_mode == "all":
        out = span.squeeze(0)
    else:
        raise ValueError(f"Unknown token_mode '{token_mode}'")

    if out_dtype is not None:
        out = out.to(out_dtype)
    return out.cpu()


def _gather_target_logprobs(logits, targets, chunk_size: int = 256):
    """Per-token log P(target) without materializing one full-vocab fp32 logits tensor.

    ``log_softmax(logits.float())`` allocates ``S x vocab x 4`` bytes; this gathers only
    the target logit and computes ``logsumexp`` in sequence chunks, bounding peak memory.
    """
    import torch

    parts = []
    for start in range(0, logits.shape[1], chunk_size):
        lg = logits[:, start : start + chunk_size, :].float()
        tg = targets[:, start : start + chunk_size]
        target_lp = lg.gather(-1, tg.unsqueeze(-1)).squeeze(-1)
        parts.append(target_lp - torch.logsumexp(lg, dim=-1))
    return torch.cat(parts, dim=1)


def _hf_extract_worker(
    model_id: str,
    lora_dir: str | None,
    prompt_strs: list[str],
    answer_strs: list[str],
    layer_indices: list[int],
    extract_kinds: list[str],
    token_mode: str,
    k: int,
    max_length: int,
    out_path: str,
):
    """
    Runs HF + PEFT forward hooks inside a spawned child process while vLLM sleeps.

    Why spawn isolation is critical on an 80 GB GPU:
    - Running HF in MainProcess leaves CUDA primary context + PyTorch caching-allocator
      segments (and any lingering frame references) resident in MainProcess, causing
      vLLM `wake_up()` to hit `CUDA Error: out of memory at cumem_allocator.cpp`.
    - When this spawned child exits, the OS & NVIDIA driver deterministically reclaim
      100% of its VRAM before vLLM `wake_up()` remaps weights.

    NOTE: the worker intentionally lives in this module. `modal run modal_app.py` uploads
    only this file; moving it to `hf_worker.py` would require wiring
    `.add_local_python_source("hf_worker")` into the image or spawn would fail to import.
    """

    import torch
    from peft import PeftModel
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
        AutoTokenizer,
    )

    # Stage-0 timings: separate HF load from forward and save so the orchestrator can
    # measure the real read fixed-overhead (not just total wall time).
    t_worker0 = time.perf_counter()
    try:
        torch.cuda.reset_peak_memory_stats()
    except Exception:  # noqa: BLE001 - timing must never fail the worker
        pass

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=MODEL_REVISION)
    # Qwen3.5 nests text weights under model.language_model.layers, so the
    # image-text class is the correct loader; AutoModelForCausalLM is only a fallback.
    try:
        hf_model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            revision=MODEL_REVISION,
            dtype=torch.bfloat16,
            device_map="cuda:0",
            low_cpu_mem_usage=True,
        )
    except Exception as e:  # noqa: BLE001 - fall back across transformers versions
        print(f"[worker] ImageText loader failed ({type(e).__name__}: {e}); trying CausalLM")
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=MODEL_REVISION,
            dtype=torch.bfloat16,
            device_map="cuda:0",
            low_cpu_mem_usage=True,
        )

    if lora_dir and os.path.exists(os.path.join(lora_dir, "adapter_model.safetensors")):
        hf_model = PeftModel.from_pretrained(hf_model, lora_dir)
    hf_model.eval()
    t_model_ready = time.perf_counter()

    base_obj = hf_model.base_model.model if isinstance(hf_model, PeftModel) else hf_model
    if hasattr(base_obj, "model") and hasattr(base_obj.model, "language_model"):
        decoder_layers = base_obj.model.language_model.layers
    elif hasattr(base_obj, "language_model"):
        decoder_layers = base_obj.language_model.layers
    else:
        decoder_layers = base_obj.model.layers

    want_hs = "hidden_states" in extract_kinds
    want_mlp = "mlp_activations" in extract_kinds
    want_lp = "logprobs" in extract_kinds

    # Deduplicate defensively: duplicate indices would register duplicate hooks and
    # silently multiply every stored tensor.
    layer_indices = sorted({int(i) for i in layer_indices})
    hs_data: dict[int, list[torch.Tensor]] = {i: [] for i in layer_indices}
    mlp_data: dict[int, list[torch.Tensor]] = {i: [] for i in layer_indices}
    lp_data: list[torch.Tensor] = []
    token_counts: list[int] = []
    answer_token_counts: list[int] = []

    current_prompt_len = 0
    hooks = []

    if want_hs:
        for l_idx in layer_indices:
            if 0 <= l_idx < len(decoder_layers):

                def make_hs_hook(idx: int):
                    def fn(_mod, _inp, out):
                        h = out[0] if isinstance(out, tuple) else out
                        hs_data[idx].append(
                            _select_span(
                                h.detach(),
                                current_prompt_len,
                                token_mode,
                                k,
                                out_dtype=torch.bfloat16,
                            )
                        )

                    return fn

                hooks.append(decoder_layers[l_idx].register_forward_hook(make_hs_hook(l_idx)))

    if want_mlp:
        for l_idx in layer_indices:
            if 0 <= l_idx < len(decoder_layers):
                mlp_mod = getattr(decoder_layers[l_idx], "mlp", None)
                down_proj = getattr(mlp_mod, "down_proj", None) if mlp_mod is not None else None
                if down_proj is None:
                    continue

                def make_mlp_hook(idx: int):
                    def pre_fn(_mod, inp):
                        # down_proj input = post-gating SwiGLU activations [1, S, intermediate]
                        act = inp[0]
                        mlp_data[idx].append(
                            _select_span(
                                act.detach(),
                                current_prompt_len,
                                token_mode,
                                k,
                                out_dtype=torch.bfloat16,
                            )
                        )

                    return pre_fn

                hooks.append(down_proj.register_forward_pre_hook(make_mlp_hook(l_idx)))

    try:
        with torch.no_grad():
            for p_str, a_str in zip(prompt_strs, answer_strs, strict=True):
                # Exact-tokenization rule: BPE is not distributive over concatenation,
                # i.e. tok(prompt + answer) != tok(prompt) + tok(answer). vLLM generated
                # the continuation from a prompt-only encode, so encode the two pieces
                # separately and concatenate ids to keep hook offsets aligned.
                if p_str:
                    p_ids = tokenizer(p_str, add_special_tokens=False)["input_ids"]
                    a_ids = tokenizer(a_str, add_special_tokens=False)["input_ids"]
                    ids = (p_ids + a_ids)[:max_length]
                    current_prompt_len = min(len(p_ids), max(0, len(ids) - 1))
                else:
                    ids = tokenizer(
                        a_str,
                        add_special_tokens=True,
                        truncation=True,
                        max_length=max_length,
                    )["input_ids"]
                    current_prompt_len = 0

                if not ids:
                    # Preserve 1:1 alignment with answer_strs: run one EOS token so
                    # every hook still logs exactly one vector for this text. Using
                    # `continue` here would desynchronize all downstream indices.
                    eos_id = tokenizer.eos_token_id
                    ids = [eos_id if eos_id is not None else 0]
                    current_prompt_len = 0

                input_ids = torch.tensor([ids], dtype=torch.long, device="cuda:0")
                enc = {
                    "input_ids": input_ids,
                    "attention_mask": torch.ones_like(input_ids),
                }
                seq_len = int(input_ids.shape[1])
                token_counts.append(seq_len)
                answer_token_counts.append(seq_len - current_prompt_len)

                # Without `logits_to_keep`, a [1, S, 248320] lm_head projection is
                # materialised even when only hidden states are requested. That is
                # 4-10+ GiB at S=8k-20k on top of the ~53 GiB model. Only logprobs
                # needs the full vocab logits.
                fwd_kwargs = {"use_cache": False}
                if not want_lp:
                    fwd_kwargs["logits_to_keep"] = 1
                try:
                    out = hf_model(**enc, **fwd_kwargs)
                except TypeError as e:
                    if "logits_to_keep" not in str(e):
                        raise
                    # Model/Transformers version does not support the kwarg; fall back
                    # to the full projection (correct, just less memory-friendly).
                    fwd_kwargs.pop("logits_to_keep", None)
                    out = hf_model(**enc, **fwd_kwargs)

                if want_lp:
                    if seq_len > 1:
                        tok_lps = _gather_target_logprobs(
                            out.logits[:, :-1, :], enc["input_ids"][:, 1:]
                        )
                        lp_start = max(0, current_prompt_len - 1)
                        lp_data.append(
                            _select_span(
                                tok_lps,
                                lp_start,
                                token_mode,
                                k,
                                out_dtype=torch.float32,
                            )
                        )
                    else:
                        # Single-token fallback has no conditional logprob; append a
                        # neutral placeholder so len(lp_data) still equals n_texts.
                        placeholder = (
                            torch.zeros((), dtype=torch.float32)
                            if token_mode in ("answer_mean", "mean")
                            else torch.zeros(1, dtype=torch.float32)
                        )
                        lp_data.append(placeholder)
                del out
                if seq_len > 4096:
                    torch.cuda.empty_cache()
    finally:
        for h in hooks:
            h.remove()
    t_forward_done = time.perf_counter()

    def _pack(d: dict[int, list[torch.Tensor]]) -> dict:
        packed = {}
        for l_idx, vecs in d.items():
            if not vecs:
                continue
            packed[l_idx] = torch.stack(vecs) if token_mode in ("answer_mean", "mean") else vecs
        return packed

    hidden_dim = next((int(v[0].shape[-1]) for v in hs_data.values() if v), None)
    mlp_dim = next((int(v[0].shape[-1]) for v in mlp_data.values() if v), None)

    payload = {
        "model_id": model_id,
        "token_mode": token_mode,
        "k": k,
        "max_length": max_length,
        "hidden_dim": hidden_dim,
        "mlp_dim": mlp_dim,
        "feature_dtype": "bfloat16",
        "token_counts": token_counts,
        "answer_token_counts": answer_token_counts,
        "hidden_states": _pack(hs_data) if want_hs else {},
        "mlp_activations": _pack(mlp_data) if want_mlp else {},
        "logprobs": (
            (torch.stack(lp_data) if token_mode in ("answer_mean", "mean") else lp_data)
            if want_lp
            else None
        ),
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    # Atomic write: a hard-killed worker must never leave a half-written .pt that the
    # orchestrator would then torch.load.
    tmp_path = f"{out_path}.tmp.{os.getpid()}"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, out_path)
    t_saved = time.perf_counter()

    # Small sidecar so the parent can report load/forward/save split without loading
    # the (possibly huge) tensor payload just to read timings.
    try:
        peak_vram_gib = round(torch.cuda.max_memory_allocated() / 2**30, 3)
    except Exception:  # noqa: BLE001 - timing must not fail a valid payload
        peak_vram_gib = None
    timings = {
        "worker_total_s": round(t_saved - t_worker0, 3),
        "load_s": round(t_model_ready - t_worker0, 3),
        "forward_s": round(t_forward_done - t_model_ready, 3),
        "pack_save_s": round(t_saved - t_forward_done, 3),
        "peak_vram_gib": peak_vram_gib,
        "n_texts": len(answer_strs),
        "feature_dtype": "bfloat16",
    }
    timings_path = f"{out_path}.timings.json"
    tmp_timings = f"{timings_path}.tmp.{os.getpid()}"
    with open(tmp_timings, "w") as f:
        json.dump(timings, f)
    os.replace(tmp_timings, timings_path)


# ---------------------------------------------------------------------------
# 3. GPU Service (Fail-Fast Startup Guard + Runtime Preflight)
# ---------------------------------------------------------------------------
