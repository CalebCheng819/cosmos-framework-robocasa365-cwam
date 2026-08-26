# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

from cosmos_framework.data.generator.action.datasets.libero_latent_cache import LiberoLatentCache


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _write_cache(root: Path) -> list[torch.Tensor]:
    records = [
        (torch.arange(48 * 5 * 10 * 20, dtype=torch.float32).reshape(48, 5, 10, 20) + offset).to(
            torch.bfloat16
        )
        for offset in (0, 8, 16)
    ]
    record_nbytes = records[0].numel() * records[0].element_size()
    stride = ((record_nbytes + 4095) // 4096) * 4096
    shards = []
    for shard_id, shard_records in enumerate((records[:2], records[2:])):
        relative = f"shards/shard-{shard_id:06d}.bin"
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            for record in shard_records:
                payload = record.contiguous().view(torch.uint8).numpy().tobytes()
                handle.write(payload)
                handle.write(b"\0" * (stride - len(payload)))
        first = 0 if shard_id == 0 else 2
        shards.append(
            {
                "shard_id": shard_id,
                "path": relative,
                "first_index": first,
                "num_records": len(shard_records),
                "size_bytes": path.stat().st_size,
            }
        )
    _write_json(
        root / "manifest.json",
        {
            "schema_version": 1,
            "dataset": {
                "name": "libero_spatial_no_noops_lerobot",
                "suite": "libero_spatial",
                "split": "train",
                "val_ratio": 0.01,
                "seed": 0,
                "chunk_length": 16,
                "camera_mode": "concat_view",
                "image_size": 256,
                "num_samples": 3,
            },
            "latent": {
                "shape": [48, 5, 10, 20],
                "storage_dtype": "bfloat16",
                "record_nbytes": record_nbytes,
                "record_stride_bytes": stride,
                "video_num_frames": 17,
                "image_size": [192, 320, 160, 320],
            },
            "shards": shards,
        },
    )
    _write_json(
        root / "_SUCCESS.json",
        {"schema_version": 1, "status": "complete", "manifest_path": "manifest.json"},
    )
    return records


def _reader(root: Path, **overrides: Any) -> LiberoLatentCache:
    kwargs: dict[str, Any] = {
        "expected_dataset_name": "libero_spatial_no_noops_lerobot",
        "expected_suite": "libero_spatial",
        "expected_split": "train",
        "expected_val_ratio": 0.01,
        "expected_seed": 0,
        "expected_chunk_length": 16,
        "expected_camera_mode": "concat_view",
        "expected_dataset_image_size": 256,
        "expected_num_samples": 3,
        "expected_image_size": (192, 320, 160, 320),
    }
    kwargs.update(overrides)
    return LiberoLatentCache(root, **kwargs)


def test_reader_loads_aligned_records_across_shards(tmp_path: Path) -> None:
    records = _write_cache(tmp_path)
    reader = _reader(tmp_path)
    for index, expected in enumerate(records):
        torch.testing.assert_close(reader.read_sample(index), expected, rtol=0, atol=0)
    torch.testing.assert_close(reader.read_sample(-1), records[-1], rtol=0, atol=0)
    assert reader.video_num_frames == 17
    assert reader.image_size == (192, 320, 160, 320)


def test_reader_rejects_incomplete_or_truncated_cache(tmp_path: Path) -> None:
    _write_cache(tmp_path)
    (tmp_path / "_SUCCESS.json").unlink()
    with pytest.raises(FileNotFoundError, match="_SUCCESS"):
        _reader(tmp_path)

    _write_cache(tmp_path)
    shard = tmp_path / "shards/shard-000001.bin"
    shard.write_bytes(shard.read_bytes()[:-1])
    with pytest.raises(ValueError, match="missing or truncated"):
        _reader(tmp_path)


def test_reader_rejects_active_dataset_contract_drift(tmp_path: Path) -> None:
    _write_cache(tmp_path)
    with pytest.raises(ValueError, match="num_samples"):
        _reader(tmp_path, expected_num_samples=4)
    with pytest.raises(ValueError, match="resize contract"):
        _reader(tmp_path, expected_image_size=(192, 320, 192, 320))


def test_reader_rejects_out_of_range_index(tmp_path: Path) -> None:
    _write_cache(tmp_path)
    reader = _reader(tmp_path)
    with pytest.raises(IndexError, match="out of range"):
        reader.read_sample(3)
