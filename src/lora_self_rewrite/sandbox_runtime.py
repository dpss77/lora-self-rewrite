from __future__ import annotations

# Mechanically ported from the donor `modal_app.py` (SHA-256
# fe71f74b0cebe933c155e03f59941c62ce402d92ca31ac2ce4e038c9b23dcbc2).
# Ported for structural parity; GPU behaviour must be re-validated before use.
import json
import math
import os
import re
import shutil

from .adapter import (
    load_and_validate_adapter,
    read_adapter_config,
    validate_lora_tensors,
)


def bind_sandbox_helpers(lora_dir: str, work_dir: str, module_registry: dict[str, dict]):
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    adapter_config = read_adapter_config(lora_dir)
    lora_rank = int(adapter_config["r"])

    notes_path = os.path.join(work_dir, "notes.md")
    candidates_dir = os.path.join(work_dir, "candidates")
    os.makedirs(candidates_dir, exist_ok=True)
    if not os.path.exists(notes_path):
        with open(notes_path, "w") as f:
            f.write("# Session Notes\n")

    def layer_outputs(layer_idx: int) -> list[str]:
        return sorted(
            {
                f"layers.{info['module_path'].split('layers.', 1)[-1]}"
                for info in module_registry.values()
                if info["layer"] == layer_idx
            }
        )

    def unit(v: torch.Tensor) -> torch.Tensor:
        vf = v.float()
        return vf / vf.norm().clamp_min(1e-8)

    def mean_diff(group_a: torch.Tensor, group_b: torch.Tensor) -> torch.Tensor:
        return unit(group_a.float().mean(dim=0) - group_b.float().mean(dim=0))

    def as_float(t: torch.Tensor) -> torch.Tensor:
        """Features are stored bf16; cast before CPU linalg (SVD/PCA lack bf16 support)."""
        return t.float()

    def _candidate_dir(name: str) -> str:
        assert re.fullmatch(r"[A-Za-z0-9_\-]+", str(name)), (
            "Candidate name must be alphanumeric/_/-."
        )
        cdir = os.path.realpath(os.path.join(candidates_dir, str(name)))
        if os.path.commonpath([cdir, os.path.realpath(candidates_dir)]) != os.path.realpath(
            candidates_dir
        ):
            raise PermissionError(f"Invalid candidate path '{name}'.")
        return cdir

    def load_base_weight(name: str) -> torch.Tensor:
        if name not in module_registry:
            layer_match = re.search(r"layers\.(\d+)", str(name))
            hint = (
                f" Try layer_outputs({layer_match.group(1)})."
                if layer_match
                else " Try layer_outputs(8)."
            )
            examples = sorted(k for k in module_registry if k.startswith("layers."))[:4]
            raise KeyError(f"Unknown module '{name}'.{hint} Examples: {examples}")
        info = module_registry[name]
        with safe_open(info["shard_path"], framework="pt", device="cpu") as f:
            # Detach from the Volume-backed safetensors mmap. The sandbox namespace is
            # persistent, so a retained view would keep the shard file open and block
            # the orchestrator's Volume.reload() (v03 Turn-20 failure).
            return f.get_tensor(info["full_key"]).clone()

    def _validate_active_config() -> None:
        if read_adapter_config(lora_dir) != adapter_config:
            raise ValueError(
                "adapter_config.json is immutable; the active config no longer "
                "matches the validated protocol config."
            )

    def load_lora_tensors() -> dict[str, torch.Tensor]:
        tensors, config = load_and_validate_adapter(lora_dir, module_registry, rank=lora_rank)
        if config != adapter_config:
            raise ValueError("Active adapter_config.json changed during the session.")
        # Clone for the same reason as load_base_weight. Drop the temporary raw dict so
        # its mmap-backed tensors do not wait on an outer frame before release.
        cloned = {k: v.clone() for k, v in tensors.items()}
        del tensors
        return cloned

    def save_lora_tensors(tensors_dict: dict[str, torch.Tensor]):
        # No commit here: the orchestrator commits once per turn (single Volume write
        # instead of several), which is enough because edits are only consumed by a GPU
        # call on the *next* turn. Write atomically so a SIGALRM mid-save cannot corrupt
        # the active adapter.
        _validate_active_config()
        # Validate the incoming tensors (float inputs are allowed here) and then the
        # exact bf16 bytes that will be persisted.
        validate_lora_tensors(tensors_dict, module_registry, rank=lora_rank, require_bfloat16=False)
        converted = {k: v.to(torch.bfloat16).cpu() for k, v in tensors_dict.items()}
        validate_lora_tensors(converted, module_registry, rank=lora_rank, require_bfloat16=True)
        out = os.path.join(lora_dir, "adapter_model.safetensors")
        tmp = f"{out}.tmp.{os.getpid()}"
        save_file(converted, tmp)
        os.replace(tmp, out)

    def set_rank1_ablation(name: str, u: torch.Tensor, alpha: float, slot: int = 0):
        """
        Encodes exact rank-1 direction ablation W_eff = W_base - alpha * u (u^T W_base)
        into LoRA slot `slot` via A[slot, :] = u^T W_base, B[:, slot] = -alpha * u.
        """
        info = module_registry.get(name)
        if info is None:
            layer_match = re.search(r"layers\.(\d+)", str(name))
            hint = (
                f" Try layer_outputs({layer_match.group(1)})."
                if layer_match
                else " Try layer_outputs(8)."
            )
            examples = sorted(k for k in module_registry if k.startswith("layers."))[:4]
            raise KeyError(f"Unknown module '{name}'.{hint} Examples: {examples}")
        tensors = load_lora_tensors()
        prefix = info["peft_prefix"]
        rank = int(tensors[f"{prefix}.lora_A.weight"].shape[0])
        if not 0 <= int(slot) < rank:
            raise IndexError(f"slot must be in [0, {rank}), got {slot}.")
        assert math.isfinite(float(alpha)), "alpha must be finite."
        a_row = tensors[f"{prefix}.lora_A.weight"][slot, :]
        b_col = tensors[f"{prefix}.lora_B.weight"][:, slot]
        if float(a_row.float().norm()) > 0 or float(b_col.float().norm()) > 0:
            print(
                f"WARNING: overwriting non-zero LoRA slot {slot} on {name}. "
                f"Use another slot (0..{rank - 1}) to combine directions."
            )
        w_base = load_base_weight(name).float()
        u_vec = unit(u)
        if tuple(u_vec.shape) != (info["shape"][0],):
            raise ValueError(
                f"u must be an OUTPUT direction of {name}: expected shape "
                f"{(info['shape'][0],)}, got {tuple(u_vec.shape)}. "
                "mlp_activations are INPUT/intermediate dims and cannot be passed "
                "directly to set_rank1_ablation."
            )
        tensors[f"{prefix}.lora_A.weight"][slot, :] = (u_vec @ w_base).to(torch.bfloat16)
        tensors[f"{prefix}.lora_B.weight"][:, slot] = (-float(alpha) * u_vec).to(torch.bfloat16)
        save_lora_tensors(tensors)
        print(f"Applied rank-1 ablation to {name} (slot={slot}, alpha={alpha}).")

    def drop_edit(name: str | None = None, slot: int | None = None):
        tensors = load_lora_tensors()
        if name is None:
            for k in tensors:
                tensors[k].zero_()
            save_lora_tensors(tensors)
            print("Dropped ALL active LoRA edits (reset to zero).")
            return
        if name not in module_registry:
            raise KeyError(f"Unknown module '{name}'.")
        prefix = module_registry[name]["peft_prefix"]
        rank = int(tensors[f"{prefix}.lora_A.weight"].shape[0])
        if slot is None:
            tensors[f"{prefix}.lora_A.weight"].zero_()
            tensors[f"{prefix}.lora_B.weight"].zero_()
        else:
            if not 0 <= int(slot) < rank:
                raise IndexError(f"slot must be in [0, {rank}), got {slot}.")
            tensors[f"{prefix}.lora_A.weight"][slot, :].zero_()
            tensors[f"{prefix}.lora_B.weight"][:, slot].zero_()
        save_lora_tensors(tensors)
        print(f"Dropped edit on {name} (slot={slot}).")

    def list_edits() -> list[dict]:
        tensors = load_lora_tensors()
        active = []
        seen = set()
        for info in module_registry.values():
            mp_str = info["module_path"]
            if mp_str in seen:
                continue
            seen.add(mp_str)
            prefix = info["peft_prefix"]
            a_norm = float(tensors[f"{prefix}.lora_A.weight"].float().norm().item())
            b_norm = float(tensors[f"{prefix}.lora_B.weight"].float().norm().item())
            if a_norm > 0.0 or b_norm > 0.0:
                active.append(
                    {
                        "module": f"layers.{mp_str.split('layers.', 1)[-1]}",
                        "lora_A_norm": round(a_norm, 4),
                        "lora_B_norm": round(b_norm, 4),
                    }
                )
        return active

    def save_candidate(name: str):
        cdir = _candidate_dir(name)
        _validate_active_config()
        load_lora_tensors()  # strict whitelist/shape/finiteness check before snapshot
        os.makedirs(cdir, exist_ok=True)
        shutil.copy2(
            os.path.join(lora_dir, "adapter_config.json"),
            os.path.join(cdir, "adapter_config.json"),
        )
        shutil.copy2(
            os.path.join(lora_dir, "adapter_model.safetensors"),
            os.path.join(cdir, "adapter_model.safetensors"),
        )
        print(f"Saved active edit as candidate '{name}'.")

    def load_candidate(name: str):
        cdir = _candidate_dir(name)
        src = os.path.join(cdir, "adapter_model.safetensors")
        if not os.path.exists(src):
            raise FileNotFoundError(f"Candidate '{name}' does not exist.")
        _, candidate_config = load_and_validate_adapter(cdir, module_registry, rank=lora_rank)
        if candidate_config != adapter_config:
            raise ValueError(
                f"Candidate '{name}' adapter_config.json differs from the "
                "immutable active protocol config."
            )
        _validate_active_config()
        dst = os.path.join(lora_dir, "adapter_model.safetensors")
        tmp = f"{dst}.tmp.{os.getpid()}"
        shutil.copy2(src, tmp)
        os.replace(tmp, dst)
        print(f"Loaded candidate '{name}' into active LoRA.")

    def list_candidates() -> list[str]:
        return sorted(
            d
            for d in os.listdir(candidates_dir)
            if os.path.exists(os.path.join(candidates_dir, d, "adapter_model.safetensors"))
        )

    def drop_candidate(name: str):
        cdir = _candidate_dir(name)
        if os.path.exists(cdir):
            shutil.rmtree(cdir)
            print(f"Removed candidate '{name}'.")

    def load_census() -> list[dict]:
        with open(os.path.join(work_dir, "census.jsonl")) as f:
            return [json.loads(line) for line in f if line.strip()]

    def load_pass(pass_idx: int) -> dict:
        idx = int(pass_idx)
        if idx < 1:
            raise ValueError(f"pass_idx must be >= 1, got {idx}.")
        with open(os.path.join(work_dir, f"pass_{idx}.json")) as f:
            data = json.load(f)
        # Normalize optional keys promised to the model; legacy pass files may predate
        # finish_reason. `base` stays None when the pass used include_base: false.
        row_groups = []
        if data.get("base"):
            row_groups.append(data["base"])
        row_groups.extend(data.get("candidates", {}).values())
        for rows in row_groups:
            for row in rows:
                row.setdefault("sample_idx", 0)
                row.setdefault("tokens", 0)
                row.setdefault("finish_reason", None)
        return data

    def read_notes() -> str:
        with open(notes_path) as f:
            return f.read()

    def write_notes(content: str):
        content = str(content)
        # ~9-10k tokens. The old 200k limit could make compaction itself exceed
        # COMPACT_TOKEN_LIMIT, causing an infinite compaction loop.
        if len(content) > 36_000:
            raise ValueError("notes.md is capped at 36k characters.")
        with open(notes_path, "w") as f:
            f.write(content)
        print(f"Saved {len(content)} chars to WORK/notes.md.")

    return {
        "layer_outputs": layer_outputs,
        "unit": unit,
        "mean_diff": mean_diff,
        "as_float": as_float,
        "load_base_weight": load_base_weight,
        "load_lora_tensors": load_lora_tensors,
        "save_lora_tensors": save_lora_tensors,
        "set_rank1_ablation": set_rank1_ablation,
        "drop_edit": drop_edit,
        "list_edits": list_edits,
        "save_candidate": save_candidate,
        "load_candidate": load_candidate,
        "list_candidates": list_candidates,
        "drop_candidate": drop_candidate,
        "load_census": load_census,
        "load_pass": load_pass,
        "read_notes": read_notes,
        "write_notes": write_notes,
    }
