# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import pytest

from cosmos_framework.callbacks.production_batch_memory_gate import (
    _evaluate_memory_rows,
    _evaluate_production_batch_rows,
)


def _row(*, rank: int, total: int = 1000, peak_reserved: int = 800, driver_used: int = 850) -> dict[str, int]:
    return {
        "rank": rank,
        "device_index": rank,
        "total_memory_bytes": total,
        "peak_allocated_bytes": 700,
        "peak_reserved_bytes": peak_reserved,
        "current_allocated_bytes": 650,
        "current_reserved_bytes": 750,
        "driver_free_bytes": total - driver_used,
        "driver_used_bytes": driver_used,
    }


def test_evaluate_memory_rows_accepts_headroom_and_sorts_ranks() -> None:
    rows, violations = _evaluate_memory_rows(
        [_row(rank=1), _row(rank=0)],
        max_peak_reserved_fraction=0.94,
        max_driver_used_fraction=0.98,
    )

    assert violations == []
    assert [row["rank"] for row in rows] == [0, 1]
    assert rows[0]["peak_reserved_fraction"] == pytest.approx(0.8)
    assert rows[0]["driver_used_fraction"] == pytest.approx(0.85)


def test_evaluate_memory_rows_fails_closed_at_each_threshold() -> None:
    rows, violations = _evaluate_memory_rows(
        [
            _row(rank=0, peak_reserved=940),
            _row(rank=1, driver_used=980),
        ],
        max_peak_reserved_fraction=0.94,
        max_driver_used_fraction=0.98,
    )

    assert len(rows) == 2
    assert violations == [
        "rank 0: peak reserved fraction 0.940000 >= 0.940000",
        "rank 1: driver used fraction 0.980000 >= 0.980000",
    ]


def test_evaluate_memory_rows_rejects_missing_rank_and_impossible_peak() -> None:
    with pytest.raises(ValueError, match="Expected one memory row"):
        _evaluate_memory_rows(
            [_row(rank=1)],
            max_peak_reserved_fraction=0.94,
            max_driver_used_fraction=0.98,
        )

    invalid = _row(rank=0)
    invalid["current_reserved_bytes"] = 900
    with pytest.raises(ValueError, match="peak_reserved_bytes"):
        _evaluate_memory_rows(
            [invalid],
            max_peak_reserved_fraction=0.94,
            max_driver_used_fraction=0.98,
        )


def test_evaluate_production_batch_rows_requires_128_by_two_on_every_rank() -> None:
    rows = []
    for rank in (0, 1):
        row: dict[str, int | float] = {"rank": rank, "observed_microbatch_count": 2}
        for microbatch in (0, 1):
            prefix = f"microbatch_{microbatch}"
            row.update(
                {
                    f"{prefix}_input_sample_count": 128,
                    f"{prefix}_unique_identity_count": 128,
                    f"{prefix}_model_batch_size": 128,
                    f"{prefix}_und_token_length": 12000,
                    f"{prefix}_gen_token_length": 84000,
                    f"{prefix}_vision_token_length": 81920,
                    f"{prefix}_action_token_length": 2048,
                }
            )
        rows.append(row)

    assert (
        _evaluate_production_batch_rows(
            rows,
            expected_samples_per_microbatch=128,
            expected_grad_accum_iter=2,
            expected_action_tokens_per_sample=16,
        )
        == []
    )

    rows[1]["microbatch_1_model_batch_size"] = 127
    rows[1]["microbatch_1_action_token_length"] = 2032
    violations = _evaluate_production_batch_rows(
        rows,
        expected_samples_per_microbatch=128,
        expected_grad_accum_iter=2,
        expected_action_tokens_per_sample=16,
    )
    assert violations == [
        "rank 1 microbatch 1: model batch size 127; expected 128",
        "rank 1 microbatch 1: action_token_length=2032; expected 2048",
    ]
