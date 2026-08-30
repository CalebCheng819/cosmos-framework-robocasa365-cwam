import json
from argparse import Namespace
from pathlib import Path

import pytest

from examples.h_cluster import archive_cwam_s1_cont100k as archiver
from examples.h_cluster.archive_cwam_s1_cont100k import (
    checkpoint_name,
    checkpoint_ready,
    checkpoint_is_stable,
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


def write_complete_checkpoint(checkpoint: Path, step: int) -> None:
    for component in ("model", "optim", "scheduler", "trainer"):
        directory = checkpoint / component
        directory.mkdir(parents=True)
        (directory / ".metadata").write_bytes(b"metadata")
        (directory / "__0_0.distcp").write_bytes(b"shard")
    (checkpoint / archiver.COMPLETE_MARKER).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "complete",
                "checkpoint": checkpoint.name,
                "iteration": step,
                "components": ["model", "optim", "scheduler", "trainer"],
            }
        ),
        encoding="utf-8",
    )


def test_checkpoint_ready_requires_valid_completion_marker(tmp_path: Path) -> None:
    step = 13_001
    checkpoint = tmp_path / checkpoint_name(step)
    write_complete_checkpoint(checkpoint, step)

    assert checkpoint_ready(checkpoint, require_completion_marker=True)
    marker = checkpoint / archiver.COMPLETE_MARKER
    payload = json.loads(marker.read_text(encoding="utf-8"))
    payload["iteration"] = step - 1
    marker.write_text(json.dumps(payload), encoding="utf-8")
    assert not checkpoint_ready(checkpoint, require_completion_marker=True)


def test_checkpoint_stability_resets_when_tree_changes(tmp_path: Path) -> None:
    checkpoint = tmp_path / checkpoint_name(13_001)
    write_complete_checkpoint(checkpoint, 13_001)
    observations: dict[Path, tuple[tuple[int, int], float]] = {}

    assert not checkpoint_is_stable(checkpoint, observations, 30, now=100.0)
    assert not checkpoint_is_stable(checkpoint, observations, 30, now=120.0)
    (checkpoint / "model" / "__0_0.distcp").write_bytes(b"larger shard")
    assert not checkpoint_is_stable(checkpoint, observations, 30, now=125.0)
    assert checkpoint_is_stable(checkpoint, observations, 30, now=155.0)


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
    assert intermediate_steps_for_boundary(20_000, 1_000, 10_000, 14_000) == list(
        range(14_000, 20_000, 1_000)
    )


def test_archiver_scans_past_a_missing_first_step(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoints_dir = tmp_path / "checkpoints"
    receipts_dir = tmp_path / "receipts"
    write_complete_checkpoint(checkpoints_dir / checkpoint_name(2_000), 2_000)
    archived: list[int] = []

    def fake_archive(checkpoint: Path, *_args: object, **_kwargs: object) -> Path:
        step = int(checkpoint.name.removeprefix("iter_"))
        archived.append(step)
        receipt = receipts_dir / f"{checkpoint.name}.json"
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_text("{}", encoding="utf-8")
        return receipt

    monkeypatch.setattr(
        archiver,
        "parse_args",
        lambda: Namespace(
            checkpoints_dir=checkpoints_dir,
            remote_root="remote:run",
            receipts_dir=receipts_dir,
            experiment_id="experiment",
            job_id="job",
            source_commit="commit",
            start_step=1_000,
            end_step=2_000,
            interval=1_000,
            compact_interval=0,
            poll_seconds=1,
            stability_seconds=0,
            drain_seconds=0,
            heartbeat_path=tmp_path / "heartbeat.json",
            allow_legacy_metadata_only=False,
            keep_local=2,
            retries=1,
            retry_seconds=0,
        ),
    )
    monkeypatch.setattr(archiver, "archive_checkpoint", fake_archive)
    monkeypatch.setattr(archiver, "prune_verified", lambda *_args: None)
    monkeypatch.setattr(archiver.signal, "signal", lambda *_args: None)
    monkeypatch.setattr(archiver.time, "sleep", lambda _seconds: None)
    archiver.STOP_REQUESTED = True
    archiver.STOP_REQUESTED_AT = archiver.time.monotonic()
    try:
        assert archiver.main() == 0
    finally:
        archiver.STOP_REQUESTED = False
        archiver.STOP_REQUESTED_AT = None

    assert archived == [2_000]


def test_archive_rejects_remote_size_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / checkpoint_name(13_001)
    write_complete_checkpoint(checkpoint, 13_001)
    monkeypatch.setattr(archiver, "run_rclone", lambda *_args: None)
    monkeypatch.setattr(
        archiver,
        "run_rclone_json",
        lambda *_args: {"count": 1, "bytes": 1},
    )

    with pytest.raises(RuntimeError, match="statistics differ"):
        archiver.archive_checkpoint(
            checkpoint,
            "remote:run",
            tmp_path / "receipts",
            "experiment",
            "job",
            "commit",
            1,
            0,
        )


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
