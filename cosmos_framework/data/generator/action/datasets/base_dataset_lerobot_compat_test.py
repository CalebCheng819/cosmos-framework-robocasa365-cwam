from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from cosmos_framework.data.generator.action.datasets.base_dataset import (
    ActionBaseDataset,
    _data_parquet_paths,
    _episode_records,
    _task_records,
)
from cosmos_framework.data.generator.action.action_spec import ActionSpec, Gripper, Pos, Rot, build_action_spec


class _ConcreteDataset(ActionBaseDataset):
    @property
    def action_dim(self) -> int:
        return 7

    def _action_spec(self) -> ActionSpec:
        return build_action_spec(Pos(), Rot("axisangle"), Gripper())

    @classmethod
    def _stats_path(cls) -> Path:
        return Path("unused.json")

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        raise NotImplementedError


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_reads_lerobot_v21_jsonl_metadata_and_episode_parquet(tmp_path: Path) -> None:
    root = tmp_path / "libero"
    data_path = root / "data/chunk-000/episode_000000.parquet"
    data_path.parent.mkdir(parents=True)
    pq.write_table(pa.table({"index": [0]}), data_path)
    _write_jsonl(root / "meta/episodes.jsonl", [{"episode_index": 0, "length": 1}])
    _write_jsonl(root / "meta/tasks.jsonl", [{"task_index": 3, "task": "pick the bowl"}])

    assert _data_parquet_paths(root) == [data_path]
    assert _episode_records(root) == [{"episode_index": 0, "length": 1}]
    assert _task_records(root) == [{"task_index": 3, "task": "pick the bowl"}]


def test_video_path_supports_lerobot_v21_episode_index_template(tmp_path: Path) -> None:
    dataset = object.__new__(_ConcreteDataset)
    dataset._root = tmp_path
    dataset._info = {
        "chunks_size": 1000,
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    }

    path = ActionBaseDataset._video_path(
        dataset,
        {"episode_index": 1002},
        "observation.images.image",
    )

    assert path == tmp_path / "videos/chunk-001/observation.images.image/episode_001002.mp4"
