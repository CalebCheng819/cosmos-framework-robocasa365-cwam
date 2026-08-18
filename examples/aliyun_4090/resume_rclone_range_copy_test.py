from __future__ import annotations

import argparse
import json
from pathlib import Path

import resume_rclone_range_copy as module


def _args(tmp_path: Path, expected: int = 26, chunk: int = 8) -> argparse.Namespace:
    return argparse.Namespace(
        rclone=tmp_path / "rclone",
        config=tmp_path / "rclone.conf",
        source="remote:bucket/object",
        destination=tmp_path / "output.bin",
        expected_bytes=expected,
        state_path=tmp_path / "state.json",
        temp_dir=tmp_path,
        chunk_bytes=chunk,
        retry_delay_seconds=0,
        max_attempts=10,
    )


def test_failed_chunk_is_not_appended(monkeypatch, tmp_path: Path) -> None:
    payload = b"abcdefghijklmnopqrstuvwxyz"
    calls = 0

    def fake_fetch(*, offset: int, count: int, temporary: Path, **_: object) -> tuple[int, str]:
        nonlocal calls
        calls += 1
        if calls == 2:
            temporary.write_bytes(payload[offset : offset + 3])
            return 1, "unexpected EOF"
        temporary.write_bytes(payload[offset : offset + count])
        return 0, ""

    monkeypatch.setattr(module, "_fetch_chunk", fake_fetch)
    args = _args(tmp_path)
    result = module.copy(args)

    assert args.destination.read_bytes() == payload
    assert result["status"] == "complete"
    assert result["attempts"] == 5
    assert json.loads(args.state_path.read_text())["completed_bytes"] == len(payload)


def test_legacy_partial_is_aligned_before_resume(monkeypatch, tmp_path: Path) -> None:
    payload = b"abcdefghijklmnopqrstuvwxyz"
    args = _args(tmp_path)
    legacy = args.destination.with_name(f"{args.destination.name}.legacy.r3.partial")
    legacy.write_bytes(payload[:13])
    offsets: list[int] = []

    def fake_fetch(*, offset: int, count: int, temporary: Path, **_: object) -> tuple[int, str]:
        offsets.append(offset)
        temporary.write_bytes(payload[offset : offset + count])
        return 0, ""

    monkeypatch.setattr(module, "_fetch_chunk", fake_fetch)
    module.copy(args)

    assert offsets[0] == 8
    assert args.destination.read_bytes() == payload
    assert not legacy.exists()


def test_existing_complete_destination_is_idempotent(tmp_path: Path) -> None:
    args = _args(tmp_path)
    args.destination.write_bytes(b"x" * args.expected_bytes)

    result = module.copy(args)

    assert result["status"] == "complete"
    assert result["attempts"] == 0
