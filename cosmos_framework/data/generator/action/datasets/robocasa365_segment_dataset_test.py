# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from cosmos_framework.data.generator.action.action_processing import ActionProcessor
from cosmos_framework.data.generator.action.datasets.robocasa365_control_cache import sha256_file
from cosmos_framework.data.generator.action.datasets.robocasa365_segment_dataset import (
    RoboCasa365SegmentDataset,
)
from cosmos_framework.data.generator.action.transforms import ActionTransformPipeline


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True))


def _segment(
    *,
    episode: int,
    group: str,
    source_frame_offset: int,
    action_offset: int,
    action_length: int,
    caption_start: int,
    caption_stop: int,
    rich: bool,
    policy: str,
) -> dict:
    return {
        "demian_episode_index": episode,
        "eligible_for_chunk": True,
        "valid_window_count": action_length - 16,
        "split_group": group,
        "source_prefix": "fixture-source",
        # Local source IDs are only unique within source_prefix. Control state
        # arrays use the dataset-global episode index instead.
        "source_episode_index": 100 if group == "g1" else 101,
        "source_dataset_episode_index": 0 if group == "g1" else 1,
        "source_frame_offset": source_frame_offset,
        "action_offset": action_offset,
        "action_length": action_length,
        "policy_task_text": policy,
        "rich_caption_enabled": rich,
        "caption_shard_path": "metadata/captions/chunk-000/file-000000.parquet",
        "caption_row_offset": caption_start,
        "caption_row_stop": caption_stop,
        "robot0_agentview_left_video_path": "videos/left.mp4",
        "robot0_agentview_left_from_timestamp": 10.0 + episode,
        "robot0_agentview_right_video_path": "videos/right.mp4",
        "robot0_agentview_right_from_timestamp": 20.0 + episode,
        "robot0_eye_in_hand_video_path": "videos/wrist.mp4",
        "robot0_eye_in_hand_from_timestamp": 30.0 + episode,
    }


@pytest.fixture()
def robocasa_view(tmp_path: Path) -> tuple[Path, Path, np.ndarray]:
    root = tmp_path / "view"
    source_root = tmp_path / "source"
    (source_root / "videos").mkdir(parents=True)

    actions = np.arange(56 * 10, dtype=np.float32).reshape(56, 10) / 100.0
    action_path = root / "data/actions/actions_rot6d_f32.npy"
    action_path.parent.mkdir(parents=True)
    np.save(action_path, actions, allow_pickle=False)

    rows = [
        # Deliberately out of order: initialization must group physical episode g1.
        _segment(
            episode=221,
            group="g2",
            source_frame_offset=0,
            action_offset=39,
            action_length=17,
            caption_start=8,
            caption_stop=9,
            rich=False,
            policy="Move from cold to hot.",
        ),
        _segment(
            episode=12,
            group="g1",
            source_frame_offset=100,
            action_offset=22,
            action_length=17,
            caption_start=5,
            caption_stop=8,
            rich=True,
            policy="Place the mug.",
        ),
        _segment(
            episode=11,
            group="g1",
            source_frame_offset=0,
            action_offset=2,
            action_length=20,
            caption_start=0,
            caption_stop=5,
            rich=True,
            policy="Increase the temperature.",
        ),
    ]
    split_path = root / "splits/train.parquet"
    split_path.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows), split_path)

    caption_rows = [
        {"demian_episode_index": 11, "frame_index": frame, "current_action": f"aligned-{frame}"}
        for frame in (0, 4, 8, 12, 16)
    ]
    caption_rows.extend(
        {"demian_episode_index": 12, "frame_index": frame, "current_action": f"mug-{frame}"} for frame in (0, 4, 8)
    )
    caption_rows.append(
        {
            "demian_episode_index": 221,
            "frame_index": 8,
            "current_action": "SENTINEL WRONG hot-to-cold annotation",
        }
    )
    caption_path = root / "metadata/captions/chunk-000/file-000000.parquet"
    caption_path.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(caption_rows), caption_path)

    stats = {
        "global_raw": {
            "mean": [0.0] * 10,
            "std": [1.0] * 10,
            "min": [-2.0] * 10,
            "max": [2.0] * 10,
            # Deliberately non-identity: offset=-1, scale=2. This makes
            # normalization-chain regressions observable in the test below.
            "q01": [-3.0] * 10,
            "q99": [1.0] * 10,
        }
    }
    _write_json(root / "stats/robocasa365_native_frame_wise_relative_rot6d.json", stats)

    manifest = {
        "schema_version": 1,
        "dataset_id": "synthetic-robocasa365",
        "inputs": {"fps": 20.0, "source_root": str(source_root)},
        "contracts": {"window_action_steps": 16, "window_video_frames": 17},
        "dense_actions": {"rot6d10": "data/actions/actions_rot6d_f32.npy"},
        "normalization_stats": "stats/robocasa365_native_frame_wise_relative_rot6d.json",
    }
    manifest_path = root / "dataset_manifest.json"
    _write_json(manifest_path, manifest)
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    _write_json(
        root / "_SUCCESS.json",
        {
            "status": "complete",
            "dataset_manifest": "dataset_manifest.json",
            "dataset_manifest_sha256": manifest_sha,
        },
    )
    return root, source_root, actions


def _write_control_cache(root: Path, *, dataset_manifest_sha256: str) -> Path:
    cache = root / "control-cache"
    arrays_root = cache / "arrays"
    arrays_root.mkdir(parents=True)
    episode_offsets = np.asarray([[0, 120], [120, 17]], dtype=np.int64)
    states = np.arange(137 * 16, dtype=np.float32).reshape(137, 16) / 100.0
    actions = np.arange(137 * 10, dtype=np.float32).reshape(137, 10) / 1000.0
    action_valid = np.ones(137, dtype=np.bool_)
    values = {
        "states": ("arrays/states_f32.npy", states),
        "actions_rot6d10": ("arrays/actions_rot6d10_f32.npy", actions),
        "action_valid": ("arrays/action_valid_bool.npy", action_valid),
        "episode_offsets": ("arrays/episode_offsets_i64.npy", episode_offsets),
    }
    descriptors = {}
    for name, (relative_path, value) in values.items():
        path = cache / relative_path
        np.save(path, value, allow_pickle=False)
        descriptors[name] = {
            "path": relative_path,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    manifest_path = cache / "manifest.json"
    _write_json(
        manifest_path,
        {
            "schema_version": 1,
            "dataset_manifest_sha256": dataset_manifest_sha256,
            "arrays": descriptors,
        },
    )
    _write_json(
        cache / "_SUCCESS.json",
        {"status": "complete", "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest()},
    )
    return cache


def test_segment_index_video_caption_and_action_contract(
    robocasa_view: tuple[Path, Path, np.ndarray],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, source_root, actions = robocasa_view
    dataset = RoboCasa365SegmentDataset(
        root=str(root),
        source_root=str(source_root),
        image_size=256,
        caption_mode="oracle_policy_rich",
        resample_on_decode_error=False,
    )
    calls: list[tuple[Path, list[float]]] = []

    def fake_decode(path: Path, timestamps: list[float]) -> torch.Tensor:
        calls.append((path, timestamps))
        value = 0.25 if path.name == "left.mp4" else 0.75
        return torch.full((17, 3, 8, 8), value, dtype=torch.float32)

    monkeypatch.setattr(dataset, "_decode_video", fake_decode)

    assert len(dataset) == 6
    # The two g1 segments are one physical-episode block; g2 is separate.
    assert dataset.get_shuffle_blocks() == [(0, 5), (5, 1)]
    assert dataset.domain_id == 22
    assert dataset.action_dim == 10

    sample = dataset[0]
    assert sample["__url__"].startswith("robocasa365://")
    assert sample["__key__"] == "000000000"
    torch.testing.assert_close(sample["action"], torch.from_numpy(actions[2:18]))
    assert sample["video"].shape == (3, 17, 256, 512)
    assert sample["video"].dtype == torch.uint8
    assert torch.all(sample["video"][..., :256] == 63)
    assert torch.all(sample["video"][..., 256:] == 191)
    assert sample["ai_caption"] == "Increase the temperature.\nCurrent action: aligned-8"
    assert calls[0][0] == source_root / "videos/left.mp4"
    assert calls[1][0] == source_root / "videos/wrist.mp4"
    assert calls[0][1] == pytest.approx([21.0 + frame / 20.0 for frame in range(17)])
    assert calls[1][1] == pytest.approx([41.0 + frame / 20.0 for frame in range(17)])

    # Conflict sample: canonical source policy remains and the rich sentinel is gated.
    conflict = dataset[5]
    assert conflict["ai_caption"] == "Move from cold to hot."
    assert "SENTINEL" not in conflict["ai_caption"]

    normalizer = dataset.get_action_normalizer()
    assert normalizer is not None
    pipeline = ActionTransformPipeline(
        tokenizer_config=None,
        max_action_dim=64,
        append_viewpoint_info=False,
        append_duration_fps_timestamps=False,
        append_resolution_info=False,
        append_idle_frames=False,
        format_prompt_as_json=False,
    )
    transformed = pipeline(sample, None, action_normalizer=normalizer)
    assert transformed["video"].shape == (3, 17, 192, 320)
    assert transformed["image_size"].tolist() == [192.0, 320.0, 160.0, 320.0]
    assert transformed["action_raw"].shape == (16, 10)
    torch.testing.assert_close(transformed["action_raw"], torch.from_numpy(actions[2:18]))
    assert transformed["action"].shape == (16, 64)
    torch.testing.assert_close(
        transformed["action"][:, :10],
        (transformed["action_raw"] + 1.0) / 2.0,
    )
    assert not torch.equal(transformed["action"][:, :10], transformed["action_raw"])
    assert torch.count_nonzero(transformed["action"][:, 10:]) == 0
    assert transformed["raw_action_dim"].item() == 10
    recovered = ActionProcessor.postprocess_action(
        transformed["action"],
        transformed["action_processing_record"],
    )
    torch.testing.assert_close(recovered, transformed["action_raw"])

    plan = transformed["sequence_plan"]
    assert plan.condition_frame_indexes_vision == [0]
    assert plan.condition_frame_indexes_action == []
    assert plan.action_start_frame_offset == 1


def test_policy_caption_is_causal_by_default(
    robocasa_view: tuple[Path, Path, np.ndarray],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, source_root, _ = robocasa_view
    dataset = RoboCasa365SegmentDataset(
        root=str(root),
        source_root=str(source_root),
        image_size=8,
        resample_on_decode_error=False,
    )

    monkeypatch.setattr(
        dataset,
        "_decode_video",
        lambda _path, _timestamps: torch.zeros((17, 3, 8, 8), dtype=torch.uint8),
    )

    sample = dataset[0]
    assert sample["ai_caption"] == "Increase the temperature."
    assert "Current action:" not in sample["ai_caption"]


def test_triple_view_and_masked_control_history_match_strong_contract(
    robocasa_view: tuple[Path, Path, np.ndarray],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, source_root, _ = robocasa_view
    control_cache = _write_control_cache(
        root,
        dataset_manifest_sha256=hashlib.sha256((root / "dataset_manifest.json").read_bytes()).hexdigest(),
    )
    dataset = RoboCasa365SegmentDataset(
        root=str(root),
        source_root=str(source_root),
        image_size=256,
        camera_mode="triple_view",
        control_cache_root=str(control_cache),
        proprio_history_length=4,
        proprio_history_stride=12,
        action_history_length=16,
        transform_resolution="480",
        resample_on_decode_error=False,
    )

    def fake_decode(path: Path, _timestamps: list[float]) -> torch.Tensor:
        value = {"left.mp4": 0.1, "wrist.mp4": 0.2, "right.mp4": 0.3}[path.name]
        return torch.full((17, 3, 8, 8), value, dtype=torch.float32)

    monkeypatch.setattr(dataset, "_decode_video", fake_decode)
    sample = dataset[4]
    assert sample["video"].shape == (3, 17, 256, 768)
    assert torch.all(sample["video"][..., :256] == 25)
    assert torch.all(sample["video"][..., 256:512] == 51)
    assert torch.all(sample["video"][..., 512:] == 76)
    assert sample["control_proprio_history"].shape == (4, 16)
    assert sample["control_proprio_history_mask"].tolist() == [True, True, True, True]
    assert sample["control_action_history"].shape == (16, 10)
    assert sample["control_action_history_mask"].all()

    pipeline = ActionTransformPipeline(
        tokenizer_config=None,
        max_action_dim=64,
        append_viewpoint_info=False,
        append_duration_fps_timestamps=False,
        append_resolution_info=False,
        append_idle_frames=False,
        format_prompt_as_json=False,
    )
    transformed = pipeline(sample, "480", action_normalizer=dataset.get_action_normalizer())
    assert transformed["video"].shape[-2:] == (480, 832)
    assert transformed["image_size"][:2].tolist() == [480.0, 832.0]
    torch.testing.assert_close(
        transformed["control_action_history"],
        (transformed["control_action_history_raw"] + 1.0) / 2.0,
    )


def test_policy_json_prompt_omits_future_action_metadata(
    robocasa_view: tuple[Path, Path, np.ndarray],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, source_root, _ = robocasa_view
    dataset = RoboCasa365SegmentDataset(
        root=str(root),
        source_root=str(source_root),
        image_size=8,
        resample_on_decode_error=False,
    )
    monkeypatch.setattr(
        dataset,
        "_decode_video",
        lambda _path, _timestamps: torch.zeros((17, 3, 8, 8), dtype=torch.uint8),
    )

    sample = dataset[0]
    assert "idle_frames" in sample  # target-derived diagnostic remains available outside the prompt
    pipeline = ActionTransformPipeline(
        tokenizer_config=None,
        max_action_dim=64,
        append_idle_frames=False,
        format_prompt_as_json=True,
    )
    transformed = pipeline(sample, None, action_normalizer=dataset.get_action_normalizer())
    prompt = transformed["ai_caption"]
    assert isinstance(prompt, dict)
    assert "idle_frame" not in prompt["actions"][0]
    assert "Current action:" not in json.dumps(prompt)


def test_manifest_checksum_is_enforced(robocasa_view: tuple[Path, Path, np.ndarray]) -> None:
    root, source_root, _ = robocasa_view
    manifest_path = root / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["dataset_id"] = "tampered"
    _write_json(manifest_path, manifest)

    with pytest.raises(RuntimeError, match="checksum mismatch"):
        RoboCasa365SegmentDataset(root=str(root), source_root=str(source_root))
