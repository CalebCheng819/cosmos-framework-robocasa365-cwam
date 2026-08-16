# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""CPU tests for packed validation dataset-name handling."""

import pytest

from cosmos_framework.callbacks.wandb_log_eval import (
    _extract_url_keys,
    _format_validation_summary,
    _normalize_dataset_name,
)


@pytest.mark.L0
@pytest.mark.CPU
def test_normalize_dataset_name_accepts_a_homogeneous_packed_batch() -> None:
    assert _normalize_dataset_name(["robocasa365", "robocasa365", "robocasa365"]) == "robocasa365"


@pytest.mark.L0
@pytest.mark.CPU
def test_normalize_dataset_name_rejects_a_mixed_packed_batch() -> None:
    with pytest.raises(ValueError, match="mixed dataset names"):
        _normalize_dataset_name(["robocasa365", "droid"])


@pytest.mark.L0
@pytest.mark.CPU
def test_normalize_dataset_name_rejects_an_empty_packed_batch() -> None:
    with pytest.raises(ValueError, match="empty list"):
        _normalize_dataset_name([])


@pytest.mark.L0
@pytest.mark.CPU
def test_extract_url_keys_preserves_every_packed_sample() -> None:
    assert _extract_url_keys(
        {
            "__url__": ["robocasa365://view/val", "robocasa365://view/val"],
            "__key__": ["000000001", "000000002"],
        }
    ) == [
        "robocasa365://view/val, 000000001",
        "robocasa365://view/val, 000000002",
    ]


@pytest.mark.L0
@pytest.mark.CPU
def test_extract_url_keys_reports_no_fake_coverage_when_metadata_is_absent() -> None:
    assert _extract_url_keys({}) == []
    assert _extract_url_keys({"__url__": ["only-url"]}) == []


@pytest.mark.L0
@pytest.mark.CPU
def test_extract_url_keys_rejects_misaligned_metadata() -> None:
    with pytest.raises(ValueError, match="cardinality mismatch"):
        _extract_url_keys({"__url__": ["u1", "u2"], "__key__": ["k1"]})


@pytest.mark.L0
@pytest.mark.CPU
def test_format_validation_summary_exposes_unstable_count() -> None:
    assert _format_validation_summary(
        unique_count=1024,
        total_count=1024,
        average_loss=1.25,
        unstable_count=0,
        iteration=500,
    ) == ("[val] number of unique url and key: 1024 / 1024; avg_final_loss: 1.25; unstable_count: 0; iteration: 500")


@pytest.mark.L0
@pytest.mark.CPU
def test_format_validation_summary_exposes_probe_metrics_in_stable_order() -> None:
    assert _format_validation_summary(
        unique_count=64,
        total_count=64,
        average_loss=0.8,
        unstable_count=0,
        iteration=500,
        probe_metrics={
            "strong_action_visual_memory_slot_abs_cosine": 0.25,
            "flow_matching_loss_action": 0.5,
        },
    ) == (
        "[val] number of unique url and key: 64 / 64; avg_final_loss: 0.8; "
        "unstable_count: 0; iteration: 500; avg_flow_matching_loss_action: 0.5; "
        "avg_strong_action_visual_memory_slot_abs_cosine: 0.25"
    )
