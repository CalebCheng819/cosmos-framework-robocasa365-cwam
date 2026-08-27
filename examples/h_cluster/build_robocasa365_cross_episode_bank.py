#!/usr/bin/env python3
"""Build strict shuffle and cross-episode routed-C banks from a normal run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from cosmos_framework.model.generator.mot.action_relevance_replay import (
    build_fixed_intervention_bank,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    trace_paths = sorted(args.trace_dir.rglob("trace_*.pt"))
    if not trace_paths:
        raise FileNotFoundError(f"No trace_*.pt files found recursively under {args.trace_dir}")
    payload = build_fixed_intervention_bank(trace_paths, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    manifest_path = args.manifest or args.output.with_suffix(".json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": payload["schema_version"],
                "pairing": payload["pairing"],
                "seed": payload["seed"],
                "source_trace_root": str(args.trace_dir.resolve()),
                "source_trace_count": len(trace_paths),
                "target_count": payload["target_count"],
                "bank_path": str(args.output.resolve()),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"saved {payload['target_count']} strict shuffle/cross-episode targets "
        f"from {len(trace_paths)} traces to {args.output}; manifest={manifest_path}"
    )


if __name__ == "__main__":
    main()
