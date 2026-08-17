# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""CPU tests for the metadata-validated EMA BF16 exporter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.distributed.checkpoint as dcp

from cosmos_framework.checkpoint.dcp import CustomLoadPlanner
from examples.h_cluster import export_cwam_serving_ema_bf16_dcp_metadata as exporter


def _fixture(tmp_path: Path) -> tuple[Path, Path, exporter.Expectations]:
    source = tmp_path / "source-model"
    values = {
        "net.a": torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16),
        "net.b": torch.tensor([5.0, 6.0], dtype=torch.bfloat16),
        "net_ema.a": torch.tensor([[1.125, 2.25], [3.5, 4.75]], dtype=torch.float32),
        "net_ema.b": torch.tensor([5.5, 6.75], dtype=torch.float32),
    }
    dcp.save(values, checkpoint_id=source)
    config = tmp_path / "config.yaml"
    config.write_text("model: tiny\n", encoding="utf-8")
    source_bytes = sum(
        tensor.numel() * tensor.element_size()
        for key, tensor in values.items()
        if key.startswith("net_ema.")
    )
    target_bytes = sum(
        tensor.numel() * torch.bfloat16.itemsize
        for key, tensor in values.items()
        if key.startswith("net_ema.")
    )
    expectations = exporter.Expectations(
        ema_key_count=2,
        regular_key_count=2,
        shard_count=1,
        source_ema_logical_bytes=source_bytes,
        target_logical_bytes=target_bytes,
    )
    return source, config, expectations


@pytest.mark.CPU
def test_export_is_ema_only_bf16_and_planner_loadable(tmp_path: Path) -> None:
    source, config, expectations = _fixture(tmp_path)
    final = tmp_path / "serving"
    result = exporter.export_checkpoint(
        source_model=source,
        source_config=config,
        final_artifact=final,
        expectations=expectations,
        sample_keys=["net_ema.a", "net_ema.b"],
        source_run_id="tiny-run",
        source_git_commit="a" * 40,
        seal=False,
    )

    assert result["status"] == "succeeded"
    target = {
        "net.a": torch.empty((2, 2), dtype=torch.bfloat16),
        "net.b": torch.empty(2, dtype=torch.bfloat16),
    }
    dcp.load(
        target,
        checkpoint_id=final / "model",
        planner=CustomLoadPlanner(load_ema_to_reg=True),
    )
    torch.testing.assert_close(
        target["net.a"],
        torch.tensor([[1.125, 2.25], [3.5, 4.75]], dtype=torch.bfloat16),
    )
    torch.testing.assert_close(target["net.b"], torch.tensor([5.5, 6.75], dtype=torch.bfloat16))
    receipt_text = (final / "ema-bf16-metadata-receipt.json").read_text(encoding="utf-8")
    assert "sha256" not in receipt_text.lower()
    receipt = json.loads(receipt_text)
    assert receipt["validation"]["full_semantic_readback"] is True


@pytest.mark.CPU
def test_wrong_key_count_fails_without_publishing(tmp_path: Path) -> None:
    source, config, expectations = _fixture(tmp_path)
    final = tmp_path / "serving"
    bad = exporter.Expectations(
        ema_key_count=3,
        regular_key_count=expectations.regular_key_count,
        shard_count=expectations.shard_count,
        source_ema_logical_bytes=expectations.source_ema_logical_bytes,
        target_logical_bytes=expectations.target_logical_bytes,
    )
    with pytest.raises(exporter.ExportError, match="EMA key count mismatch"):
        exporter.export_checkpoint(
            source_model=source,
            source_config=config,
            final_artifact=final,
            expectations=bad,
            sample_keys=["net_ema.a"],
            source_run_id="tiny-run",
            source_git_commit="a" * 40,
            seal=False,
        )
    assert not final.exists()
