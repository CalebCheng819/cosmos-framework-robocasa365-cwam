#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Mirror binary stdin to stdout and one securely created CWAM log file."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable
from typing import BinaryIO, TextIO

if __package__:
    from examples.h_cluster.cwam_secure_paths import (
        PathReceipt,
        SecurePathError,
        create_file_exclusive,
        normalized_json,
    )
else:
    from cwam_secure_paths import (  # type: ignore[import-not-found]
        PathReceipt,
        SecurePathError,
        create_file_exclusive,
        normalized_json,
    )

_CHUNK_BYTES = 1024 * 1024


def _write_fd_all(fd: int, data: bytes | bytearray | memoryview) -> None:
    view = memoryview(data)
    offset = 0
    while offset < len(view):
        try:
            written = os.write(fd, view[offset:])
        except InterruptedError:
            continue
        if written <= 0:
            raise OSError("secure log write made no forward progress")
        offset += written


def _write_stream_all(stream: BinaryIO, data: bytes | bytearray | memoryview) -> None:
    view = memoryview(data)
    offset = 0
    while offset < len(view):
        try:
            written = stream.write(view[offset:])
        except InterruptedError:
            continue
        if written is None or written <= 0:
            raise OSError("stdout write made no forward progress")
        offset += written


def mirror_binary_stream(
    input_stream: BinaryIO,
    output_stream: BinaryIO,
    file_fd: int,
    *,
    chunk_bytes: int = _CHUNK_BYTES,
) -> dict[str, int]:
    """Mirror a binary stream using only the already-open secure file fd."""

    if isinstance(chunk_bytes, bool) or not isinstance(chunk_bytes, int) or chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be a positive integer")

    chunks = 0
    total_bytes = 0
    while True:
        try:
            chunk = input_stream.read(chunk_bytes)
        except InterruptedError:
            continue
        if chunk == b"":
            break
        if not isinstance(chunk, bytes):
            raise TypeError("stdin must be a binary stream whose read() returns bytes")
        _write_fd_all(file_fd, chunk)
        _write_stream_all(output_stream, chunk)
        chunks += 1
        total_bytes += len(chunk)

    output_stream.flush()
    os.fsync(file_fd)
    return {"bytes_mirrored": total_bytes, "chunks_mirrored": chunks}


def run_secure_tee(
    *,
    root: str,
    path: str,
    input_stream: BinaryIO,
    output_stream: BinaryIO,
    file_mode: int = 0o640,
    create_file: Callable[..., tuple[int, PathReceipt]] = create_file_exclusive,
) -> PathReceipt:
    """Create the log exclusively, mirror bytes, fsync, and close its sole fd."""

    file_fd, receipt = create_file(root, path, file_mode=file_mode)
    operation_error: Exception | None = None
    stats: dict[str, int] | None = None
    try:
        stats = mirror_binary_stream(input_stream, output_stream, file_fd)
    except Exception as error:
        operation_error = error

    try:
        os.close(file_fd)
    except OSError as error:
        if operation_error is None:
            operation_error = error

    if operation_error is not None:
        raise operation_error
    if stats is None:
        raise AssertionError("secure tee completed without stream statistics")
    return {**receipt, **stats, "operation": "secure_tee"}


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--path", required=True)
    parser.add_argument("--file-mode", type=lambda value: int(value, 8), default=0o640)
    return parser.parse_args(argv)


def _report_error(error: Exception, stderr: TextIO) -> None:
    try:
        stderr.write(
            normalized_json(
                {
                    "error": str(error),
                    "error_type": type(error).__name__,
                    "ok": False,
                    "operation": "secure_tee",
                    "schema_version": 1,
                }
            )
        )
        stderr.flush()
    except Exception:
        # The original operation has already failed.  A broken diagnostic
        # stream must not turn that failure into a successful exit.
        pass


def main(
    argv: list[str] | None = None,
    *,
    input_stream: BinaryIO | None = None,
    output_stream: BinaryIO | None = None,
    error_stream: TextIO | None = None,
) -> int:
    """Run secure tee; mirrored bytes are the only successful stdout output."""

    args = _parse_args(argv)
    binary_input = sys.stdin.buffer if input_stream is None else input_stream
    binary_output = sys.stdout.buffer if output_stream is None else output_stream
    text_error = sys.stderr if error_stream is None else error_stream
    try:
        run_secure_tee(
            root=args.root,
            path=args.path,
            input_stream=binary_input,
            output_stream=binary_output,
            file_mode=args.file_mode,
        )
    except (BrokenPipeError, OSError, SecurePathError, TypeError, ValueError) as error:
        _report_error(error, text_error)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
