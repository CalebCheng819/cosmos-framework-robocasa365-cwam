# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""CPU tests for the streaming serving-only EMA BF16 DCP exporter."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import pickle
from pathlib import Path

import pytest
import torch
import torch.distributed.checkpoint as dcp

from cosmos_framework.checkpoint.dcp import CustomLoadPlanner
from examples.h_cluster import cwam_serving_checkpoint_manifest
from examples.h_cluster import export_cwam_serving_ema_bf16_dcp as exporter


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, exporter.ExportExpectations]:
    source = tmp_path / "source-model"
    regular = {
        "net.a": torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16),
        "net.b": torch.tensor([5.0, 6.0], dtype=torch.bfloat16),
        "net_ema.a": torch.tensor([[1.125, 2.25], [3.5, 4.75]], dtype=torch.float32),
        "net_ema.b": torch.tensor([5.5, 6.75], dtype=torch.float32),
    }
    dcp.save(regular, checkpoint_id=source)
    config = tmp_path / "config.yaml"
    config.write_text("model: tiny\n", encoding="utf-8")
    origin_manifest = cwam_serving_checkpoint_manifest.create_serving_checkpoint_manifest(source)
    manifest_path = tmp_path / "origin-manifest.json"
    manifest_path.write_text(json.dumps(origin_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    source_ema_bytes = sum(
        tensor.numel() * tensor.element_size()
        for key, tensor in regular.items()
        if key.startswith("net_ema.")
    )
    target_bytes = sum(
        tensor.numel() * torch.bfloat16.itemsize for key, tensor in regular.items() if key.startswith("net_ema.")
    )
    expectations = exporter.ExportExpectations(
        origin_manifest_sha256=_sha256(manifest_path),
        origin_tree_sha256=str(origin_manifest["tree_sha256"]),
        source_metadata_sha256=_sha256(source / ".metadata"),
        source_config_sha256=_sha256(config),
        ema_key_count=2,
        regular_key_count=2,
        shard_count=1,
        source_ema_logical_bytes=source_ema_bytes,
        target_logical_bytes=target_bytes,
    )
    return source, config, manifest_path, expectations


@pytest.mark.CPU
def test_streaming_export_keeps_only_bf16_ema_and_planner_loads_it(tmp_path: Path) -> None:
    source, config, manifest_path, expectations = _fixture(tmp_path)
    final = tmp_path / "serving-ema-bf16"

    result = exporter.export_ema_bf16_checkpoint(
        source_model=source,
        source_config=config,
        origin_manifest_path=manifest_path,
        final_artifact=final,
        expectations=expectations,
        sample_keys=["net_ema.a", "net_ema.b"],
        seal=False,
    )

    assert result["status"] == "succeeded"
    assert result["key_count"] == 2
    assert result["target_logical_bytes"] == expectations.target_logical_bytes
    assert not any(path.name.startswith(f".{final.name}.incomplete-") for path in tmp_path.iterdir())

    with (final / "model" / ".metadata").open("rb") as stream:
        metadata = pickle.load(stream)
    assert sorted(metadata.state_dict_metadata) == ["net_ema.a", "net_ema.b"]
    assert {value.properties.dtype for value in metadata.state_dict_metadata.values()} == {torch.bfloat16}

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

    manifest = json.loads((final / "serving-checkpoint-manifest.json").read_text())
    assert cwam_serving_checkpoint_manifest.verify_serving_checkpoint_manifest(
        manifest,
        final / "model",
    )["status"] == "passed"
    validation = json.loads((final / "ema-bf16-validation.json").read_text())
    assert validation["status"] == "passed"
    assert validation["full_readback"]["verified_chunk_count"] == 2
    receipt = json.loads((final / "ema-bf16-export-receipt.json").read_text())
    assert receipt["selection"]["included_prefix"] == "net_ema."
    assert receipt["selection"]["excluded_prefix"] == "net."


@pytest.mark.CPU
def test_export_refuses_wrong_source_metadata_hash_without_publishing(tmp_path: Path) -> None:
    source, config, manifest_path, expectations = _fixture(tmp_path)
    final = tmp_path / "serving-ema-bf16"
    bad = dataclasses.replace(expectations, source_metadata_sha256="0" * 64)

    with pytest.raises(exporter.EmaBf16ExportError, match=r"source \.metadata SHA-256"):
        exporter.export_ema_bf16_checkpoint(
            source_model=source,
            source_config=config,
            origin_manifest_path=manifest_path,
            final_artifact=final,
            expectations=bad,
            sample_keys=["net_ema.a"],
            seal=False,
        )

    assert not final.exists()


@pytest.mark.CPU
def test_export_refuses_same_size_source_shard_tamper(tmp_path: Path) -> None:
    source, config, manifest_path, expectations = _fixture(tmp_path)
    final = tmp_path / "serving-ema-bf16"
    shard = next(source.glob("*.distcp"))
    payload = bytearray(shard.read_bytes())
    payload[-1] ^= 1
    shard.write_bytes(payload)

    with pytest.raises(exporter.EmaBf16ExportError, match="source file SHA-256 mismatch"):
        exporter.export_ema_bf16_checkpoint(
            source_model=source,
            source_config=config,
            origin_manifest_path=manifest_path,
            final_artifact=final,
            expectations=expectations,
            sample_keys=["net_ema.a"],
            seal=False,
        )

    assert not final.exists()
