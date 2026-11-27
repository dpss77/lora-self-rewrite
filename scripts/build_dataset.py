#!/usr/bin/env python3
"""Build the deterministic DeepScaleR scrubbed dataset payload (CPU only)."""

from __future__ import annotations

import argparse
import json

from _common import ROOT

from lora_self_rewrite.datasets import build_deepscaler_payload


def _real_loader(repo, cfg, split, revision=None):
    from datasets import load_dataset

    kwargs = {"split": split}
    if revision:
        kwargs["revision"] = revision
    return load_dataset(repo, cfg, **kwargs) if cfg else load_dataset(repo, **kwargs)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=str(ROOT / "datasets" / "deepscaler_scrubbed.json"))
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)

    payload = build_deepscaler_payload(_real_loader)
    with open(args.out, "w") as f:
        json.dump(payload, f)
    meta = payload["meta"]
    print(
        f"wrote {args.out}: clean={meta['n_clean']} "
        f"practice={len(payload['practice'])} reserved={len(payload['reserved'])} "
        f"heldout={len(payload['heldout'])} "
        f"math={meta['n_math_ok']}/{meta['n_expected_math_sources']} gpqa={meta['gpqa_ok']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
