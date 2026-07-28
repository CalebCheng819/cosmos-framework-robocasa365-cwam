# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

import cosmos_framework.data.generator.action.datasets.robocasa365_latent_cache as latent_cache_module
from cosmos_framework.data.generator.action.datasets.robocasa365_latent_cache import (
    OFFICIAL_WAN22_VAE_SHA256,
    WAN_CACHE_VAE_BEHAVIOR_CONTRACT,
    RoboCasa365LatentCache,
    active_preprocessing_code_sha256,
    ordered_identity_sha256,
    sha256_file,
    validate_published_cache_files,
)
from cosmos_framework.data.generator.action.datasets.robocasa365_segment_dataset import (
    RoboCasa365SegmentDataset,
)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True))


def _rewrite_success(cache_root: Path) -> None:
    manifest_path = cache_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    _write_json(
        cache_root / "_SUCCESS.json",
        {
            "schema_version": 1,
            "status": "complete",
            "cache_id": manifest["cache_id"],
            "manifest_path": "manifest.json",
            "manifest_sha256": sha256_file(manifest_path),
        },
    )


def _write_cache(
    cache_root: Path,
    *,
    dataset_id: str,
    dataset_manifest_sha256: str,
    mapping: np.ndarray | None = None,
    records: list[torch.Tensor] | None = None,
    ordered_visual_identity_sha256: str = "d" * 64,
    source_split_manifest_sha256: str = "e" * 64,
) -> tuple[dict[str, Any], list[torch.Tensor]]:
    cache_root.mkdir(parents=True)
    if records is None:
        records = [
            (torch.arange(48 * 5 * 10 * 20, dtype=torch.float32).reshape(48, 5, 10, 20) / 128 + offset).to(
                torch.bfloat16
            )
            for offset in (0, 32)
        ]
    if mapping is None:
        mapping = np.asarray([1, 0, 1], dtype=np.uint32)

    record_nbytes = records[0].numel() * records[0].element_size()
    record_stride_bytes = ((record_nbytes + 63) // 64) * 64
    shard_path = cache_root / "shards" / "shard-000000.bin"
    shard_path.parent.mkdir(parents=True)
    with shard_path.open("wb") as handle:
        for record in records:
            assert record.dtype == torch.bfloat16
            assert tuple(record.shape) == (48, 5, 10, 20)
            payload = record.contiguous().view(torch.uint8).numpy().tobytes()
            handle.write(payload)
            handle.write(b"\0" * (record_stride_bytes - len(payload)))

    mapping_path = cache_root / "mappings" / "train.npy"
    mapping_path.parent.mkdir(parents=True)
    np.save(mapping_path, mapping, allow_pickle=False)
    identity_path = cache_root / "identity" / "ordinal-sha256.bin"
    identity_path.parent.mkdir(parents=True)
    identity_path.write_bytes(b"".join(hashlib.sha256(str(index).encode()).digest() for index in range(len(records))))
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "cache_id": "synthetic-wan-cache",
        "dataset": {
            "dataset_id": dataset_id,
            "manifest_sha256": dataset_manifest_sha256,
        },
        "preprocessing": {
            "schema_version": 1,
            "camera_mode": "concat_view",
            "ordered_camera_fields": ["robot0_agentview_left", "robot0_eye_in_hand"],
            "fps": 20.0,
            "dataset_image_size": 256,
            "video_resize": {
                "resolution": None,
                "resolution_selection": "auto_get_vision_data_resolution",
                "keep_aspect_ratio": True,
                "pad_keys": ["video"],
                "resize_interpolation": "bicubic",
                "antialias": True,
                "padding_origin": "top_left",
                "padding_mode": "reflect_else_edge",
                "image_size": [192, 320, 160, 320],
            },
            "post_encode_crop": {
                "stage": "post_padding_crop",
                "spatial_rounding": "floor",
                "minimum_latent_size": 1,
            },
            "code_sha256": active_preprocessing_code_sha256(),
        },
        "vae": {
            "weights_sha256": OFFICIAL_WAN22_VAE_SHA256,
            "weights_size_bytes": 1,
            "behavior_relevant_config": WAN_CACHE_VAE_BEHAVIOR_CONTRACT,
        },
        "latent": {
            "shape": [48, 5, 10, 20],
            "storage_dtype": "bfloat16",
            "logical_dtype": "float32",
            "stage": "post_padding_crop",
            "record_nbytes": record_nbytes,
            "record_stride_bytes": record_stride_bytes,
            "byte_order": "little",
            "video_num_frames": 17,
            "image_size": [192, 320, 160, 320],
        },
        "splits": {
            "train": {
                "logical_samples": int(mapping.shape[0]),
                "unique_records": int(np.unique(mapping).shape[0]),
                "mapping_path": "mappings/train.npy",
                "mapping_sha256": sha256_file(mapping_path),
                "ordered_visual_identity_sha256": ordered_visual_identity_sha256,
                "source_split_manifest_sha256": source_split_manifest_sha256,
            }
        },
        "mapping_gate": {
            "status": "passed",
            "full_coverage": True,
            "algorithm": "canonical-first-occurrence-v1",
            "identity_digest_algorithm": "sha256-length-prefixed-canonical-json-v1",
            "logical_samples": int(mapping.shape[0]),
            "unique_records": len(records),
            "ordinal_identity_path": "identity/ordinal-sha256.bin",
            "ordinal_identity_record_bytes": 32,
            "ordinal_identity_size_bytes": identity_path.stat().st_size,
            "ordinal_identity_sha256": sha256_file(identity_path),
        },
        "shards": [
            {
                "shard_id": 0,
                "path": "shards/shard-000000.bin",
                "first_ordinal": 0,
                "num_records": len(records),
                "size_bytes": shard_path.stat().st_size,
                "sha256": sha256_file(shard_path),
            }
        ],
        "equivalence": {
            "status": "passed",
            "torch_equal": True,
            "num_samples": len(records),
        },
    }
    _write_json(cache_root / "manifest.json", manifest)
    _rewrite_success(cache_root)
    return manifest, records


def _reader_kwargs(
    *,
    dataset_id: str,
    dataset_manifest_sha256: str,
    expected_num_samples: int = 3,
    ordered_visual_identity_sha256: str = "d" * 64,
    source_split_manifest_sha256: str = "e" * 64,
) -> dict[str, Any]:
    return {
        "split": "train",
        "expected_dataset_id": dataset_id,
        "expected_dataset_manifest_sha256": dataset_manifest_sha256,
        "expected_num_samples": expected_num_samples,
        "expected_camera_mode": "concat_view",
        "expected_ordered_camera_fields": ("robot0_agentview_left", "robot0_eye_in_hand"),
        "expected_fps": 20.0,
        "expected_dataset_image_size": 256,
        "expected_image_size": (192, 320, 160, 320),
        "expected_vae_weights_sha256": OFFICIAL_WAN22_VAE_SHA256,
        "expected_preprocessing_code_sha256": active_preprocessing_code_sha256(),
        "expected_split_manifest_sha256": source_split_manifest_sha256,
        "expected_ordered_visual_identity_sha256": ordered_visual_identity_sha256,
    }


@pytest.fixture()
def dataset_view(tmp_path: Path) -> tuple[Path, Path, str]:
    root = tmp_path / "view"
    source_root = tmp_path / "source"
    source_root.mkdir()

    action_path = root / "data" / "actions" / "actions_rot6d_f32.npy"
    action_path.parent.mkdir(parents=True)
    np.save(action_path, np.zeros((18, 10), dtype=np.float32), allow_pickle=False)

    row = {
        "demian_episode_index": 7,
        "eligible_for_chunk": True,
        "valid_window_count": 2,
        "split_group": "physical-episode-7",
        "source_prefix": "robocasa/task/demo",
        "source_episode_index": 70,
        "source_frame_offset": 100,
        "action_offset": 0,
        "action_length": 18,
        "policy_task_text": "Put the mug in the sink.",
        "rich_caption_enabled": False,
        "caption_shard_path": "unused.parquet",
        "caption_row_offset": 0,
        "caption_row_stop": 0,
        "robot0_agentview_left_video_path": "missing-left.mp4",
        "robot0_agentview_left_from_timestamp": 1.0,
        "robot0_agentview_right_video_path": "missing-right.mp4",
        "robot0_agentview_right_from_timestamp": 1.0,
        "robot0_eye_in_hand_video_path": "missing-wrist.mp4",
        "robot0_eye_in_hand_from_timestamp": 1.0,
    }
    split_path = root / "splits" / "train.parquet"
    split_path.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([row]), split_path)

    manifest = {
        "schema_version": 1,
        "dataset_id": "synthetic-robocasa365",
        "inputs": {
            "fps": 20.0,
            "source_root": str(source_root),
            "robocasa365_revision": "synthetic-revision",
        },
        "contracts": {"window_action_steps": 16, "window_video_frames": 17},
        "dense_actions": {"rot6d10": "data/actions/actions_rot6d_f32.npy"},
        "counts": {"action_rows": 18},
    }
    manifest_path = root / "dataset_manifest.json"
    _write_json(manifest_path, manifest)
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    _write_json(
        root / "_SUCCESS.json",
        {
            "status": "complete",
            "dataset_id": manifest["dataset_id"],
            "dataset_manifest": "dataset_manifest.json",
            "dataset_manifest_sha256": manifest_sha,
        },
    )
    return root, source_root, manifest_sha


def test_fixed_record_reader_loads_mapping_into_memory_and_uses_one_pread(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_root = tmp_path / "cache"
    _, records = _write_cache(
        cache_root,
        dataset_id="dataset-v1",
        dataset_manifest_sha256="a" * 64,
    )
    reader = RoboCasa365LatentCache(
        cache_root,
        **_reader_kwargs(dataset_id="dataset-v1", dataset_manifest_sha256="a" * 64),
        verify_shard_checksums=True,
    )
    assert not isinstance(reader.mapping, np.memmap)

    real_pread = os.pread
    calls: list[tuple[int, int, int]] = []

    def recording_pread(fd: int, count: int, offset: int) -> bytes:
        calls.append((fd, count, offset))
        return real_pread(fd, count, offset)

    monkeypatch.setattr(latent_cache_module.os, "pread", recording_pread)
    first = reader.read_sample(0)
    duplicate = reader.read_sample(2)

    assert len(calls) == 2
    assert calls[0][1] == reader.layout.record_nbytes
    assert calls[0][2] == reader.layout.record_stride_bytes
    assert first.dtype == torch.bfloat16
    assert first.is_contiguous()
    torch.testing.assert_close(first, records[1], rtol=0, atol=0)
    torch.testing.assert_close(duplicate, first, rtol=0, atol=0)


def test_shared_published_files_validator_returns_inventory_and_rejects_string_shard_id(
    tmp_path: Path,
) -> None:
    cache_root = tmp_path / "cache"
    manifest, _ = _write_cache(
        cache_root,
        dataset_id="dataset-v1",
        dataset_manifest_sha256="f" * 64,
    )
    validated = validate_published_cache_files(cache_root, verify_shard_checksums=True)
    assert validated["cache_id"] == manifest["cache_id"]
    roles = {entry["role"] for entry in validated["files"]}
    assert roles == {
        "success",
        "manifest",
        "mapping:train",
        "ordinal_identity_inventory",
        "shard",
    }
    assert all(entry["checksum_verified"] for entry in validated["files"])

    manifest["shards"][0]["shard_id"] = "0"
    _write_json(cache_root / "manifest.json", manifest)
    _rewrite_success(cache_root)
    with pytest.raises(ValueError, match="shard_id must be an integer"):
        validate_published_cache_files(cache_root)


def test_reader_rejects_active_preprocessing_contract_drift(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    manifest, _ = _write_cache(
        cache_root,
        dataset_id="dataset-v1",
        dataset_manifest_sha256="1" * 64,
    )
    manifest["preprocessing"]["camera_mode"] = "wrist"
    _write_json(cache_root / "manifest.json", manifest)
    _rewrite_success(cache_root)
    with pytest.raises(RuntimeError, match="camera_mode mismatch"):
        RoboCasa365LatentCache(
            cache_root,
            **_reader_kwargs(dataset_id="dataset-v1", dataset_manifest_sha256="1" * 64),
        )


def test_reader_rejects_tampered_manifest_and_mapping(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    _write_cache(
        cache_root,
        dataset_id="dataset-v1",
        dataset_manifest_sha256="b" * 64,
    )
    manifest_path = cache_root / "manifest.json"
    manifest_path.write_text(manifest_path.read_text() + "\n")
    with pytest.raises(RuntimeError, match="manifest checksum mismatch"):
        RoboCasa365LatentCache(
            cache_root,
            **_reader_kwargs(dataset_id="dataset-v1", dataset_manifest_sha256="b" * 64),
        )

    _rewrite_success(cache_root)
    mapping_path = cache_root / "mappings" / "train.npy"
    mapping_path.write_bytes(mapping_path.read_bytes() + b"x")
    with pytest.raises(RuntimeError, match="mapping checksum mismatch"):
        RoboCasa365LatentCache(
            cache_root,
            **_reader_kwargs(dataset_id="dataset-v1", dataset_manifest_sha256="b" * 64),
        )


def test_reader_rejects_failed_gate_and_detects_full_shard_checksum_mismatch(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    manifest, _ = _write_cache(
        cache_root,
        dataset_id="dataset-v1",
        dataset_manifest_sha256="c" * 64,
    )
    manifest["equivalence"]["status"] = "failed"
    _write_json(cache_root / "manifest.json", manifest)
    _rewrite_success(cache_root)
    with pytest.raises(RuntimeError, match="bitwise equivalence gate"):
        RoboCasa365LatentCache(
            cache_root,
            **_reader_kwargs(dataset_id="dataset-v1", dataset_manifest_sha256="c" * 64),
        )

    manifest["equivalence"]["status"] = "passed"
    _write_json(cache_root / "manifest.json", manifest)
    _rewrite_success(cache_root)
    shard_path = cache_root / "shards" / "shard-000000.bin"
    with shard_path.open("r+b") as handle:
        handle.seek(0)
        handle.write(b"\xff")
    with pytest.raises(RuntimeError, match="shard 0 checksum mismatch"):
        RoboCasa365LatentCache(
            cache_root,
            **_reader_kwargs(dataset_id="dataset-v1", dataset_manifest_sha256="c" * 64),
            verify_shard_checksums=True,
        )


def test_reader_rejects_nonfinite_record_payload(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    manifest, _ = _write_cache(
        cache_root,
        dataset_id="dataset-v1",
        dataset_manifest_sha256="2" * 64,
    )
    shard_path = cache_root / "shards" / "shard-000000.bin"
    nan_bytes = torch.tensor([float("nan")], dtype=torch.bfloat16).view(torch.uint8).numpy().tobytes()
    with shard_path.open("r+b") as handle:
        handle.write(nan_bytes)
    manifest["shards"][0]["sha256"] = sha256_file(shard_path)
    _write_json(cache_root / "manifest.json", manifest)
    _rewrite_success(cache_root)
    reader = RoboCasa365LatentCache(
        cache_root,
        **_reader_kwargs(dataset_id="dataset-v1", dataset_manifest_sha256="2" * 64),
    )
    with pytest.raises(RuntimeError, match="non-finite"):
        reader.read_sample(1)


def test_dataset_required_mode_returns_cache_and_never_decodes_video(
    dataset_view: tuple[Path, Path, str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, source_root, dataset_manifest_sha = dataset_view
    cache_root = tmp_path / "cache"
    identities = [
        {
            "source_revision": "synthetic-revision",
            "source_prefix": "robocasa/task/demo",
            "source_episode_index": 70,
            "source_frame_index": 100 + index,
            "camera_mode": "concat_view",
            "camera_fields": ["robot0_agentview_left", "robot0_eye_in_hand"],
            "camera_video_paths": ["missing-left.mp4", "missing-wrist.mp4"],
            "camera_effective_start_timestamps": [1.0 + index / 20.0, 1.0 + index / 20.0],
            "fps": 20.0,
            "num_frames": 17,
        }
        for index in range(2)
    ]
    split_manifest_sha = sha256_file(root / "splits" / "train.parquet")
    _, records = _write_cache(
        cache_root,
        dataset_id="synthetic-robocasa365",
        dataset_manifest_sha256=dataset_manifest_sha,
        mapping=np.asarray([1, 0], dtype=np.uint32),
        ordered_visual_identity_sha256=ordered_identity_sha256(identities),
        source_split_manifest_sha256=split_manifest_sha,
    )
    dataset = RoboCasa365SegmentDataset(
        root=str(root),
        source_root=str(source_root),
        action_normalization=None,
        latent_cache_root=str(cache_root),
        latent_cache_mode="required",
    )

    def forbidden_video_load(*_args: object, **_kwargs: object) -> torch.Tensor:
        raise AssertionError("required latent-cache mode must not decode video")

    monkeypatch.setattr(dataset, "_load_video", forbidden_video_load)
    sample = dataset[0]
    assert "video" not in sample
    assert sample["video_num_frames"].item() == 17
    assert sample["image_size"].tolist() == [192.0, 320.0, 160.0, 320.0]
    torch.testing.assert_close(sample["video_latent"], records[1], rtol=0, atol=0)

    assert dataset._latent_cache is not None
    monkeypatch.setattr(
        dataset._latent_cache,
        "read_sample",
        lambda _idx: (_ for _ in ()).throw(OSError("bad cache")),
    )
    with pytest.raises(RuntimeError, match="bad cache"):
        dataset[1]


def test_builder_visual_only_dataset_does_not_load_dense_actions(
    dataset_view: tuple[Path, Path, str],
) -> None:
    root, source_root, _ = dataset_view
    dataset = RoboCasa365SegmentDataset(
        root=str(root),
        source_root=str(source_root),
        action_normalization=None,
        load_actions=False,
    )

    assert dataset._actions is None
    assert dataset.get_visual_window_identity(0)["num_frames"] == 17
    with pytest.raises(RuntimeError, match="load_actions=False"):
        dataset[0]
    assert dataset.get_visual_window_identity(1)["source_frame_index"] == 101


def test_visual_identity_canonicalizes_overlapping_segment_timestamp_ulps(
    dataset_view: tuple[Path, Path, str],
) -> None:
    root, source_root, _ = dataset_view
    dataset = RoboCasa365SegmentDataset(
        root=str(root),
        source_root=str(source_root),
        action_normalization=None,
        load_actions=False,
    )
    first = dict(dataset._segments[0])
    first.update(
        {
            "source_frame_offset": 0,
            "valid_window_count": 89,
            "robot0_agentview_left_from_timestamp": 11.7,
            "robot0_eye_in_hand_from_timestamp": 11.7,
        }
    )
    overlapping = dict(first)
    overlapping.update(
        {
            "demian_episode_index": 8,
            "source_frame_offset": 80,
            "valid_window_count": 8,
            "robot0_agentview_left_from_timestamp": 15.7,
            "robot0_eye_in_hand_from_timestamp": 15.7,
        }
    )
    dataset._segments = [first, overlapping]
    dataset._valid_cum = np.asarray([89, 97], dtype=np.int64)

    identity_from_first = dataset.get_visual_window_identity(87)
    identity_from_overlap = dataset.get_visual_window_identity(96)

    assert 11.7 + 87 / 20.0 != 15.7 + 7 / 20.0
    assert identity_from_first == identity_from_overlap
    assert identity_from_first["camera_effective_start_timestamps"] == [16.05, 16.05]


def test_batched_overlapping_window_decode_is_pixel_exact(
    dataset_view: tuple[Path, Path, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, source_root, _ = dataset_view
    dataset = RoboCasa365SegmentDataset(
        root=str(root),
        source_root=str(source_root),
        image_size=8,
        action_normalization=None,
        latent_cache_mode="disabled",
    )
    calls: list[tuple[str, list[float]]] = []

    def fake_decode(path: Path, timestamps: list[float]) -> torch.Tensor:
        calls.append((path.name, timestamps))
        camera_offset = 0.1 if "left" in path.name else 0.2
        values = torch.tensor([camera_offset + timestamp / 100.0 for timestamp in timestamps])
        return values[:, None, None, None].expand(-1, 3, 4, 4).contiguous()

    monkeypatch.setattr(dataset, "_decode_video", fake_decode)
    expected = torch.stack(
        [dataset._load_video(*dataset.resolve_window(index)) for index in (0, 1)],
        dim=0,
    )
    calls.clear()
    actual = dataset.load_video_windows([0, 1])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert len(calls) == 2
    assert [len(timestamps) for _, timestamps in calls] == [18, 18]


def test_dataset_requires_explicit_cache_contract(
    dataset_view: tuple[Path, Path, str],
    tmp_path: Path,
) -> None:
    root, source_root, _ = dataset_view
    with pytest.raises(ValueError, match="latent_cache_root is required"):
        RoboCasa365SegmentDataset(
            root=str(root),
            source_root=str(source_root),
            action_normalization=None,
            latent_cache_mode="required",
        )
    with pytest.raises(ValueError, match="must be omitted"):
        RoboCasa365SegmentDataset(
            root=str(root),
            source_root=str(source_root),
            action_normalization=None,
            latent_cache_root=str(tmp_path / "cache"),
            latent_cache_mode="disabled",
        )
