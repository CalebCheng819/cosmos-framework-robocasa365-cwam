# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from examples.h_cluster import cwam_secure_tee
from examples.h_cluster.cwam_secure_tee import main, mirror_binary_stream, run_secure_tee


def _new_root(tmp_path: Path) -> Path:
    root = tmp_path / "pinned"
    root.mkdir()
    return root


def _run_main(root: Path, target: Path, data: bytes) -> tuple[int, bytes, str]:
    output = io.BytesIO()
    errors = io.StringIO()
    return_code = main(
        ["--root", str(root), "--path", str(target)],
        input_stream=io.BytesIO(data),
        output_stream=output,
        error_stream=errors,
    )
    return return_code, output.getvalue(), errors.getvalue()


class _PartialWriter(io.BytesIO):
    def __init__(self, maximum_write: int) -> None:
        super().__init__()
        self.maximum_write = maximum_write
        self.flush_count = 0

    def write(self, data: bytes | bytearray | memoryview) -> int:
        return super().write(bytes(data[: self.maximum_write]))

    def flush(self) -> None:
        self.flush_count += 1
        super().flush()


class _BrokenPipeWriter(io.BytesIO):
    def write(self, data: bytes | bytearray | memoryview) -> int:
        raise BrokenPipeError("downstream stdout closed")


class _FlushFailureWriter(io.BytesIO):
    def flush(self) -> None:
        raise OSError("stdout flush failed")


class _ReadFailure(io.BytesIO):
    def read(self, size: int = -1) -> bytes:
        raise OSError("stdin read failed")


def test_main_mirrors_binary_stdin_exactly(tmp_path: Path) -> None:
    root = _new_root(tmp_path)
    target = root / "train.log"
    data = bytes(range(256)) + b"\x00\xffCWAM\n"

    return_code, output, errors = _run_main(root, target, data)

    assert return_code == 0
    assert output == data
    assert target.read_bytes() == data
    assert errors == ""


def test_standalone_cli_mirrors_binary_without_pythonpath(tmp_path: Path) -> None:
    root = _new_root(tmp_path)
    target = root / "standalone.log"
    data = b"\x00standalone\xff\n"
    script = Path(__file__).with_name("cwam_secure_tee.py")
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)

    result = subprocess.run(
        [sys.executable, str(script), "--root", str(root), "--path", str(target)],
        input=data,
        capture_output=True,
        check=False,
        env=environment,
    )

    assert result.returncode == 0
    assert result.stdout == data
    assert result.stderr == b""
    assert target.read_bytes() == data


def test_mirror_binary_stream_handles_partial_fd_and_stdout_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "partial.log"
    file_fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    output = _PartialWriter(maximum_write=5)
    data = bytes(range(64))
    real_write = os.write

    def partial_fd_write(fd: int, chunk: bytes | bytearray | memoryview) -> int:
        return real_write(fd, bytes(chunk[:3]))

    monkeypatch.setattr(cwam_secure_tee.os, "write", partial_fd_write)
    try:
        stats = mirror_binary_stream(io.BytesIO(data), output, file_fd, chunk_bytes=11)
    finally:
        os.close(file_fd)

    assert target.read_bytes() == data
    assert output.getvalue() == data
    assert output.flush_count == 1
    assert stats == {"bytes_mirrored": len(data), "chunks_mirrored": 6}


def test_run_secure_tee_uses_supplied_open_fd_and_returns_counts(tmp_path: Path) -> None:
    target = tmp_path / "owned-fd.log"
    file_fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    calls: list[tuple[str, str]] = []

    def create_file(
        root: str,
        path: str,
        *,
        file_mode: int,
    ) -> tuple[int, dict[str, object]]:
        calls.append((root, path))
        assert file_mode == 0o444
        return file_fd, {"ok": True, "operation": "create_file_exclusive"}

    output = io.BytesIO()
    receipt = run_secure_tee(
        root="/pinned/root",
        path="/pinned/root/log",
        input_stream=io.BytesIO(b"abc"),
        output_stream=output,
        file_mode=0o444,
        create_file=create_file,
    )

    assert calls == [("/pinned/root", "/pinned/root/log")]
    assert output.getvalue() == b"abc"
    assert target.read_bytes() == b"abc"
    assert receipt["operation"] == "secure_tee"
    assert receipt["bytes_mirrored"] == 3
    with pytest.raises(OSError):
        os.fstat(file_fd)


def test_cli_can_create_read_only_log_while_retaining_open_write_fd(tmp_path: Path) -> None:
    root = _new_root(tmp_path)
    target = root / "immutable.log"
    output = io.BytesIO()
    errors = io.StringIO()

    return_code = main(
        ["--root", str(root), "--path", str(target), "--file-mode", "0444"],
        input_stream=io.BytesIO(b"complete\n"),
        output_stream=output,
        error_stream=errors,
    )

    assert return_code == 0
    assert output.getvalue() == b"complete\n"
    assert target.read_bytes() == b"complete\n"
    assert target.stat().st_mode & 0o777 == 0o444
    assert errors.getvalue() == ""


def test_existing_file_is_never_reopened_or_overwritten(tmp_path: Path) -> None:
    root = _new_root(tmp_path)
    target = root / "train.log"
    target.write_bytes(b"existing")

    return_code, output, errors = _run_main(root, target, b"replacement")

    assert return_code != 0
    assert output == b""
    assert target.read_bytes() == b"existing"
    assert json.loads(errors)["error_type"] == "SecurePathAlreadyExistsError"


@pytest.mark.parametrize("dangling", [False, True])
def test_leaf_symlink_is_rejected(tmp_path: Path, dangling: bool) -> None:
    root = _new_root(tmp_path)
    destination = root / "destination.log"
    if not dangling:
        destination.write_bytes(b"destination")
    target = root / "train.log"
    target.symlink_to(destination)

    return_code, output, errors = _run_main(root, target, b"must not write")

    assert return_code != 0
    assert output == b""
    assert target.is_symlink()
    assert json.loads(errors)["error_type"] == "SecurePathAlreadyExistsError"
    if not dangling:
        assert destination.read_bytes() == b"destination"


@pytest.mark.parametrize("dangling", [False, True])
def test_ancestor_symlink_is_rejected(tmp_path: Path, dangling: bool) -> None:
    root = _new_root(tmp_path)
    actual = root / "actual"
    if not dangling:
        actual.mkdir()
    alias = root / "logs"
    alias.symlink_to(actual, target_is_directory=True)
    target = alias / "train.log"

    return_code, output, errors = _run_main(root, target, b"must not write")

    assert return_code != 0
    assert output == b""
    assert json.loads(errors)["error_type"] in {"SecurePathNotFoundError", "SymlinkPathError"}
    if not dangling:
        assert not (actual / "train.log").exists()


def test_target_escape_is_rejected_before_reading_stdin(tmp_path: Path) -> None:
    root = _new_root(tmp_path)
    outside = tmp_path / "outside.log"
    input_stream = _ReadFailure()
    output = io.BytesIO()
    errors = io.StringIO()

    return_code = main(
        ["--root", str(root), "--path", str(outside)],
        input_stream=input_stream,
        output_stream=output,
        error_stream=errors,
    )

    assert return_code != 0
    assert output.getvalue() == b""
    assert not outside.exists()
    assert json.loads(errors.getvalue())["error_type"] == "PathEscapeError"


def test_broken_pipe_is_nonzero_and_secure_fd_is_closed(tmp_path: Path) -> None:
    root = _new_root(tmp_path)
    target = root / "train.log"
    errors = io.StringIO()

    return_code = main(
        ["--root", str(root), "--path", str(target)],
        input_stream=io.BytesIO(b"persisted prefix"),
        output_stream=_BrokenPipeWriter(),
        error_stream=errors,
    )

    assert return_code != 0
    assert target.read_bytes() == b"persisted prefix"
    assert json.loads(errors.getvalue())["error_type"] == "BrokenPipeError"


def test_file_write_failure_is_nonzero_and_stdout_receives_no_chunk(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _new_root(tmp_path)
    target = root / "train.log"
    output = io.BytesIO()
    errors = io.StringIO()

    def failed_write(fd: int, data: bytes | bytearray | memoryview) -> int:
        raise OSError("injected file write failure")

    monkeypatch.setattr(cwam_secure_tee.os, "write", failed_write)
    return_code = main(
        ["--root", str(root), "--path", str(target)],
        input_stream=io.BytesIO(b"data"),
        output_stream=output,
        error_stream=errors,
    )

    assert return_code != 0
    assert target.read_bytes() == b""
    assert output.getvalue() == b""
    assert "injected file write failure" in json.loads(errors.getvalue())["error"]


def test_stdout_flush_failure_is_nonzero(tmp_path: Path) -> None:
    root = _new_root(tmp_path)
    target = root / "train.log"
    output = _FlushFailureWriter()
    errors = io.StringIO()

    return_code = main(
        ["--root", str(root), "--path", str(target)],
        input_stream=io.BytesIO(b"data"),
        output_stream=output,
        error_stream=errors,
    )

    assert return_code != 0
    assert target.read_bytes() == b"data"
    assert output.getvalue() == b"data"
    assert "stdout flush failed" in json.loads(errors.getvalue())["error"]


def test_fsync_failure_is_nonzero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _new_root(tmp_path)
    target = root / "train.log"
    output = io.BytesIO()
    errors = io.StringIO()

    def failed_fsync(fd: int) -> None:
        raise OSError("injected fsync failure")

    monkeypatch.setattr(cwam_secure_tee.os, "fsync", failed_fsync)
    return_code = main(
        ["--root", str(root), "--path", str(target)],
        input_stream=io.BytesIO(b"data"),
        output_stream=output,
        error_stream=errors,
    )

    assert return_code != 0
    assert target.read_bytes() == b"data"
    assert output.getvalue() == b"data"
    assert "injected fsync failure" in json.loads(errors.getvalue())["error"]


def test_stdin_read_failure_is_nonzero(tmp_path: Path) -> None:
    root = _new_root(tmp_path)
    target = root / "train.log"
    errors = io.StringIO()

    return_code = main(
        ["--root", str(root), "--path", str(target)],
        input_stream=_ReadFailure(),
        output_stream=io.BytesIO(),
        error_stream=errors,
    )

    assert return_code != 0
    assert target.read_bytes() == b""
    assert "stdin read failed" in json.loads(errors.getvalue())["error"]


def test_nonbinary_stdin_is_rejected(tmp_path: Path) -> None:
    root = _new_root(tmp_path)
    target = root / "train.log"
    errors = io.StringIO()

    return_code = main(
        ["--root", str(root), "--path", str(target)],
        input_stream=io.StringIO("text"),  # type: ignore[arg-type]
        output_stream=io.BytesIO(),
        error_stream=errors,
    )

    assert return_code != 0
    assert target.read_bytes() == b""
    assert json.loads(errors.getvalue())["error_type"] == "TypeError"


def test_invalid_chunk_size_fails_before_reading() -> None:
    with pytest.raises(ValueError):
        mirror_binary_stream(_ReadFailure(), io.BytesIO(), 1, chunk_bytes=0)
