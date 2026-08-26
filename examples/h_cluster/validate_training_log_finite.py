#!/usr/bin/env python3
"""Validate that a short Cosmos training log contains finite optimizer evidence."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path


LOSS_PATTERN = re.compile(r"Iteration\s+(?P<iteration>\d+):.*?Loss:\s*(?P<value>[^\s|]+)")
GRAD_PATTERN = re.compile(r"clip_grad_norm/[^\s:|]+:\s*(?P<value>[^\s|]+)")


def _finite_values(text: str, pattern: re.Pattern[str], label: str) -> list[float]:
    values: list[float] = []
    for match in pattern.finditer(text):
        token = match.group("value")
        try:
            value = float(token)
        except ValueError as error:
            raise ValueError(f"invalid {label} value: {token!r}") from error
        if not math.isfinite(value):
            raise ValueError(f"non-finite {label} value: {token!r}")
        values.append(value)
    return values


def validate_training_log(path: Path, *, min_iterations: int) -> tuple[int, int]:
    text = path.read_text(errors="replace")
    losses = _finite_values(text, LOSS_PATTERN, "loss")
    gradients = _finite_values(text, GRAD_PATTERN, "gradient norm")
    iterations = {int(match.group("iteration")) for match in LOSS_PATTERN.finditer(text)}
    missing = sorted(set(range(1, min_iterations + 1)) - iterations)
    if missing:
        raise ValueError(f"missing loss evidence for iterations: {missing}")
    if not gradients:
        raise ValueError("missing gradient-norm evidence")
    return len(losses), len(gradients)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--min-iterations", type=int, default=2)
    args = parser.parse_args()
    loss_count, gradient_count = validate_training_log(args.log, min_iterations=args.min_iterations)
    print(f"FINITE_TRAINING_LOG_READY losses={loss_count} gradient_norms={gradient_count}")


if __name__ == "__main__":
    main()
