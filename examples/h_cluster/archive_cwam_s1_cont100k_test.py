import json
from pathlib import Path

import pytest

from examples.h_cluster import archive_cwam_s1_cont100k as archiver
from examples.h_cluster.archive_cwam_s1_cont100k import (
    checkpoint_name,
    checkpoint_ready,
    compact_verified_window,
    intermediate_steps_for_boundary,
    tree_stats,
)


def test_checkpoint_ready_requires_all_dcp_components(tmp_path: Path) -> None:
    checkpoint = tmp_path / checkpoint_name(20_000)
    for component in ("model", "optim", "scheduler"):
        directory = checkpoint / component
        directory.mkdir(parents=True)
        (directory / ".metadata").write_bytes(b"metadata")
    assert not checkpoint_ready(checkpoint)
    trainer = checkpoint / "trainer"
    trainer.mkdir()
    (trainer / ".metadata").write_bytes(b"metadata")
    assert checkpoint_ready(checkpoint)
    assert tree_stats(checkpoint) == (4, 32)


def test_checkpoint_name_uses_nine_digit_cosmos_format() -> None:
    assert checkpoint_name(100_000) == "iter_000100000"


def write_verified_receipt(receipts_dir: Path, remote_root: str, step: int) -> None:
    name = checkpoint_name(step)
    receipt = {
        "status": "verified",
        "checkpoint": name,
        "destination": f"{remote_root}/checkpoints/{name}",
    }
    receipts_dir.mkdir(parents=True, exist_ok=True)
    (receipts_dir / f"{name}.json").write_text(json.dumps(receipt), encoding="utf-8")


def test_intermediate_steps_cover_only_the_completed_window() -> None:
    assert intermediate_steps_for_boundary(9_000, 1_000, 10_000) == []
    assert intermediate_steps_for_boundary(10_000, 1_000, 10_000) == list(range(1_000, 10_000, 1_000))
    assert intermediate_steps_for_boundary(20_000, 1_000, 10_000) == list(range(11_000, 20_000, 1_000))
    with pytest.raises(ValueError, match="multiple"):
        intermediate_steps_for_boundary(10_000, 3_000, 10_000)


def test_compaction_deletes_only_verified_intermediate_checkpoints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoints_dir = tmp_path / "checkpoints"
    receipts_dir = tmp_path / "receipts"
    remote_root = "remote:bucket/chengjuntao/run"
    for step in range(1_000, 11_000, 1_000):
        (checkpoints_dir / checkpoint_name(step)).mkdir(parents=True)
        write_verified_receipt(receipts_dir, remote_root, step)

    rclone_calls: list[list[str]] = []
    monkeypatch.setattr(
        archiver,
        "run_rclone",
        lambda arguments, retries, retry_seconds: rclone_calls.append(arguments),
    )

    receipt_path = compact_verified_window(
        10_000,
        checkpoints_dir,
        remote_root,
        receipts_dir,
        1_000,
        10_000,
        "experiment",
        "job",
        "commit",
        1,
        0,
    )

    expected_remote_deletes = [
        ["delete", f"{remote_root}/checkpoints/{checkpoint_name(step)}", "--rmdirs"]
        for step in range(1_000, 10_000, 1_000)
    ]
    assert rclone_calls[:-1] == expected_remote_deletes
    assert rclone_calls[-1][0] == "copyto"
    assert receipt_path == receipts_dir / "compaction_at_iter_000010000.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["retained_checkpoint"] == checkpoint_name(10_000)
    assert receipt["deleted_checkpoints"] == [
        checkpoint_name(step) for step in range(1_000, 10_000, 1_000)
    ]
    assert (checkpoints_dir / checkpoint_name(10_000)).is_dir()
    assert not any((checkpoints_dir / checkpoint_name(step)).exists() for step in range(1_000, 10_000, 1_000))


def test_compaction_requires_all_receipts_before_any_delete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoints_dir = tmp_path / "checkpoints"
    receipts_dir = tmp_path / "receipts"
    remote_root = "remote:bucket/chengjuntao/run"
    for step in range(1_000, 11_000, 1_000):
        if step != 5_000:
            write_verified_receipt(receipts_dir, remote_root, step)

    rclone_calls: list[list[str]] = []
    monkeypatch.setattr(
        archiver,
        "run_rclone",
        lambda arguments, retries, retry_seconds: rclone_calls.append(arguments),
    )

    with pytest.raises(RuntimeError, match="verified receipt missing"):
        compact_verified_window(
            10_000,
            checkpoints_dir,
            remote_root,
            receipts_dir,
            1_000,
            10_000,
            "experiment",
            "job",
            "commit",
            1,
            0,
        )
    assert rclone_calls == []
