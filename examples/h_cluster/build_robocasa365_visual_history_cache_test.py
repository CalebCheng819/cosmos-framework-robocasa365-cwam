# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from cosmos_framework.data.generator.action.datasets.robocasa365_visual_history_cache import (
    RoboCasa365VisualHistoryCache,
)
from examples.h_cluster.build_robocasa365_visual_history_cache import build_cache


def _write_parent(root: Path, records: list[torch.Tensor]) -> None:
    root.mkdir()
    shard_path = root / "shards" / "parent.bin"
    shard_path.parent.mkdir()
    record_nbytes = records[0].numel() * 2
    record_stride_bytes = record_nbytes + 64
    with shard_path.open("wb") as stream:
        for record in records:
            payload = record.contiguous().view(torch.uint8).numpy().tobytes()
            stream.write(payload)
            stream.write(b"\0" * (record_stride_bytes - len(payload)))
    manifest = {
        "schema_version": 1,
        "cache_id": "parent-cache",
        "latent": {
            "shape": [48, 5, 30, 52],
            "storage_dtype": "bfloat16",
            "record_nbytes": record_nbytes,
            "record_stride_bytes": record_stride_bytes,
        },
        "mapping_gate": {"status": "passed", "unique_records": len(records)},
        "shards": [
            {
                "shard_id": 0,
                "first_ordinal": 0,
                "num_records": len(records),
                "path": "shards/parent.bin",
                "size_bytes": shard_path.stat().st_size,
            }
        ],
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    (root / "_SUCCESS.json").write_text(json.dumps({"status": "complete", "cache_id": "parent-cache"}))


def test_builder_and_reader_preserve_parent_ordinals(tmp_path: Path) -> None:
    records = [
        (torch.arange(48 * 5 * 30 * 52).reshape(48, 5, 30, 52).float() / 1024 + offset).to(
            torch.bfloat16
        )
        for offset in (0.0, 3.0, 7.0)
    ]
    parent = tmp_path / "parent"
    sidecar = tmp_path / "sidecar"
    _write_parent(parent, records)

    manifest = build_cache(parent, sidecar, chunk_records=2, workers=2)
    reader = RoboCasa365VisualHistoryCache(
        sidecar,
        expected_parent_cache_id="parent-cache",
        expected_num_records=3,
    )

    assert manifest["num_records"] == 3
    assert manifest["shape"] == [48, 15, 26]
    for ordinal, record in enumerate(records):
        expected = F.avg_pool2d(record[:, 0].float(), kernel_size=2, stride=2).to(torch.bfloat16)
        torch.testing.assert_close(reader.read_ordinal(ordinal), expected, rtol=0.0, atol=0.0)

    sidecar_shard = sidecar / "shards" / "shape-48x15x26" / "shard-000000.bin"
    shard_mtime_ns = sidecar_shard.stat().st_mtime_ns
    build_cache(parent, sidecar, chunk_records=2)
    assert sidecar_shard.stat().st_mtime_ns == shard_mtime_ns

    # Parent structure is still validated before an existing sidecar is accepted.
    parent_payload = parent / "shards" / "parent.bin"
    parent_payload.unlink()
    with pytest.raises(RuntimeError, match="Parent shard size mismatch"):
        build_cache(parent, sidecar, chunk_records=2)


def test_reader_rejects_sidecar_for_a_different_parent(tmp_path: Path) -> None:
    record = torch.zeros(48, 5, 30, 52, dtype=torch.bfloat16)
    parent = tmp_path / "parent"
    sidecar = tmp_path / "sidecar"
    _write_parent(parent, [record])
    build_cache(parent, sidecar, chunk_records=1)

    with pytest.raises(RuntimeError, match="parent cache mismatch"):
        RoboCasa365VisualHistoryCache(
            sidecar,
            expected_parent_cache_id="wrong-parent",
            expected_num_records=1,
        )
