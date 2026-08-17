#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Validate finite, all-rank B2 training progress from a Cosmos log."""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path

_FLOAT = r"[-+]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|inf(?:inity)?|nan)"
_ITER_SPEED_PATTERN = re.compile(
    rf"\[RANK\s+(?P<rank>\d+)\]\s+Iteration\s+(?P<iteration>\d+):[^\n]*?"
    rf"\|\s+Loss:\s*(?P<loss>{_FLOAT})",
    re.IGNORECASE,
)
_LEGACY_PATTERN = re.compile(
    rf"Iteration:\s*(?P<iteration>\d+),[^\n]*?total loss\s+(?P<loss>{_FLOAT})",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class TrainingRecord:
    iteration: int
    loss: float
    rank: int | None


def parse_training_records(text: str) -> list[TrainingRecord]:
    records = [
        TrainingRecord(
            iteration=int(match.group("iteration")),
            loss=float(match.group("loss")),
            rank=int(match.group("rank")),
        )
        for match in _ITER_SPEED_PATTERN.finditer(text)
    ]
    records.extend(
        TrainingRecord(
            iteration=int(match.group("iteration")),
            loss=float(match.group("loss")),
            rank=None,
        )
        for match in _LEGACY_PATTERN.finditer(text)
    )
    return records


def validate_training_records(
    records: list[TrainingRecord], *, expected_iteration: int, world_size: int
) -> dict[str, object]:
    if not records:
        raise ValueError("no supported iteration/loss records found")
    nonfinite = [record for record in records if not math.isfinite(record.loss)]
    if nonfinite:
        raise ValueError(f"non-finite loss records found: {nonfinite}")

    expected_records = [record for record in records if record.iteration == expected_iteration]
    if not expected_records:
        observed = sorted({record.iteration for record in records})
        raise ValueError(f"iteration {expected_iteration} absent; observed={observed[-10:]}")

    ranked_records = [record for record in expected_records if record.rank is not None]
    if ranked_records:
        observed_ranks = {record.rank for record in ranked_records}
        expected_ranks = set(range(world_size))
        if observed_ranks != expected_ranks:
            raise ValueError(
                f"iteration {expected_iteration} rank coverage mismatch: "
                f"expected={sorted(expected_ranks)}, observed={sorted(observed_ranks)}"
            )
        losses = [record.loss for record in ranked_records]
        log_format = "iter_speed"
    else:
        observed_ranks = set()
        losses = [record.loss for record in expected_records]
        log_format = "legacy"

    return {
        "expected_iteration": expected_iteration,
        "format": log_format,
        "mean_loss": sum(losses) / len(losses),
        "observed_ranks": sorted(observed_ranks),
        "record_count": len(records),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--expected-iteration", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        records = parse_training_records(args.log.read_text(errors="replace"))
        result = validate_training_records(
            records,
            expected_iteration=args.expected_iteration,
            world_size=args.world_size,
        )
    except (OSError, ValueError) as error:
        raise SystemExit(f"training log validation failed for {args.log}: {error}") from error
    print(
        "TRAINING_LOG_VALID",
        f"format={result['format']}",
        f"expected_iteration={result['expected_iteration']}",
        f"observed_ranks={result['observed_ranks']}",
        f"mean_loss={result['mean_loss']:.8g}",
    )


if __name__ == "__main__":
    main()
