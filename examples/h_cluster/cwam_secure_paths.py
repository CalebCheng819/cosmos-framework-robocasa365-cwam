#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Fail-closed path handling for formal CWAM runs.

The helpers in this module deliberately do not accept a path merely because
``Path.resolve()`` places it under a trusted root.  Every existing component is
inspected with lstat semantics and then opened relative to an already verified
directory descriptor with ``O_NOFOLLOW``.  The lstat and descriptor identities
must agree, which turns component replacement into a hard failure.

An absence check is not a reservation.  Callers that need a reservation should
use :func:`create_file_exclusive`; callers that need complete file contents to
appear atomically should use :func:`write_file_exclusive`.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import secrets
import stat
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

PathKind = Literal["directory", "file"]
PathReceipt = dict[str, Any]

_SCHEMA_VERSION = 1
_DIRECTORY_OPEN_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
_FILE_OPEN_FLAGS = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW


class SecurePathError(RuntimeError):
    """Base class for a rejected secure-path operation."""


class InvalidLiteralPathError(SecurePathError):
    """Raised when a supplied path is not an exact absolute literal."""


class PathEscapeError(SecurePathError):
    """Raised when a target is not strictly beneath its pinned root."""


class SymlinkPathError(SecurePathError):
    """Raised when any existing path component is a symlink."""


class PathTypeError(SecurePathError):
    """Raised when a component has the wrong filesystem type."""


class SecurePathNotFoundError(SecurePathError):
    """Raised when a required component does not exist."""


class SecurePathAlreadyExistsError(SecurePathError):
    """Raised when an exclusive target already exists, including as a symlink."""


class SecurePathRaceError(SecurePathError):
    """Raised when lstat and descriptor identities reveal path replacement."""


@dataclass(frozen=True)
class _LiteralPath:
    value: str
    parts: tuple[str, ...]


@dataclass
class _RootHandle:
    literal: _LiteralPath
    fd: int
    stat_result: os.stat_result


def _identity(stat_result: os.stat_result, *, kind: PathKind) -> dict[str, int | str]:
    payload: dict[str, int | str] = {
        "device": int(stat_result.st_dev),
        "inode": int(stat_result.st_ino),
        "kind": kind,
        "mode": f"{stat.S_IMODE(stat_result.st_mode):04o}",
    }
    if kind == "file":
        payload["size"] = int(stat_result.st_size)
    return payload


def _same_inode(left: os.stat_result, right: os.stat_result) -> bool:
    return left.st_dev == right.st_dev and left.st_ino == right.st_ino


def _parse_literal(path: str | os.PathLike[str], *, label: str) -> _LiteralPath:
    raw = os.fspath(path)
    if not isinstance(raw, str):
        raise InvalidLiteralPathError(f"{label} must be a text path")
    if not raw:
        raise InvalidLiteralPathError(f"{label} must not be empty")
    if "\x00" in raw:
        raise InvalidLiteralPathError(f"{label} contains a NUL byte")
    if not os.path.isabs(raw) or not raw.startswith("/"):
        raise InvalidLiteralPathError(f"{label} must be an absolute literal path: {raw!r}")
    if raw == "/":
        return _LiteralPath(value=raw, parts=())

    split_parts = raw.split("/")
    if split_parts[0] != "":
        raise InvalidLiteralPathError(f"{label} must start at the POSIX filesystem root: {raw!r}")
    parts = split_parts[1:]
    if any(component == ".." for component in parts):
        raise InvalidLiteralPathError(f"{label} must not contain '..': {raw!r}")
    if any(component in {"", "."} for component in parts):
        raise InvalidLiteralPathError(
            f"{label} must be lexically normalized (no '.', repeated '/', or trailing '/'): {raw!r}"
        )
    return _LiteralPath(value=raw, parts=tuple(parts))


def _validate_mode(mode: int, *, label: str) -> int:
    if isinstance(mode, bool) or not isinstance(mode, int) or not 0 <= mode <= 0o777:
        raise ValueError(f"{label} must be an integer between 0o000 and 0o777")
    return mode


def _relative_parts(root: _LiteralPath, target: _LiteralPath) -> tuple[str, ...]:
    root_length = len(root.parts)
    if len(target.parts) <= root_length or target.parts[:root_length] != root.parts:
        raise PathEscapeError(f"target must be strictly beneath root: root={root.value!r}, target={target.value!r}")
    return target.parts[root_length:]


def _kind_from_mode(mode: int) -> PathKind:
    if stat.S_ISDIR(mode):
        return "directory"
    if stat.S_ISREG(mode):
        return "file"
    raise PathTypeError("only regular files and directories are accepted")


def _lstat_at(parent_fd: int, name: str, *, display_path: str) -> os.stat_result:
    try:
        return os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError as error:
        raise SecurePathNotFoundError(f"path component does not exist: {display_path}") from error


def _open_existing_directory_component(parent_fd: int, name: str, *, display_path: str) -> tuple[int, os.stat_result]:
    before = _lstat_at(parent_fd, name, display_path=display_path)
    if stat.S_ISLNK(before.st_mode):
        raise SymlinkPathError(f"symlink path component is forbidden: {display_path}")
    if not stat.S_ISDIR(before.st_mode):
        raise PathTypeError(f"directory path component has the wrong type: {display_path}")

    try:
        child_fd = os.open(name, _DIRECTORY_OPEN_FLAGS, dir_fd=parent_fd)
    except FileNotFoundError as error:
        raise SecurePathRaceError(f"directory component disappeared during validation: {display_path}") from error
    except OSError as error:
        if error.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise SecurePathRaceError(f"directory component changed during validation: {display_path}") from error
        raise

    after = os.fstat(child_fd)
    if not _same_inode(before, after):
        os.close(child_fd)
        raise SecurePathRaceError(f"directory component identity changed during validation: {display_path}")
    return child_fd, after


def _open_existing_leaf(
    parent_fd: int,
    name: str,
    *,
    display_path: str,
    expected_kind: PathKind,
) -> tuple[int, os.stat_result]:
    before = _lstat_at(parent_fd, name, display_path=display_path)
    if stat.S_ISLNK(before.st_mode):
        raise SymlinkPathError(f"symlink target is forbidden: {display_path}")
    actual_kind = _kind_from_mode(before.st_mode)
    if actual_kind != expected_kind:
        raise PathTypeError(
            f"path has the wrong type: expected={expected_kind}, actual={actual_kind}, path={display_path}"
        )

    flags = _DIRECTORY_OPEN_FLAGS if expected_kind == "directory" else _FILE_OPEN_FLAGS
    try:
        leaf_fd = os.open(name, flags, dir_fd=parent_fd)
    except FileNotFoundError as error:
        raise SecurePathRaceError(f"path disappeared during validation: {display_path}") from error
    except OSError as error:
        if error.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise SecurePathRaceError(f"path changed during validation: {display_path}") from error
        raise
    after = os.fstat(leaf_fd)
    if not _same_inode(before, after):
        os.close(leaf_fd)
        raise SecurePathRaceError(f"path identity changed during validation: {display_path}")
    return leaf_fd, after


def _verify_opened_literal(
    literal: _LiteralPath,
    opened_stat: os.stat_result,
    *,
    expected_kind: PathKind,
) -> None:
    try:
        path_stat = os.lstat(literal.value)
    except FileNotFoundError as error:
        raise SecurePathRaceError(f"path disappeared after descriptor traversal: {literal.value}") from error
    if stat.S_ISLNK(path_stat.st_mode):
        raise SymlinkPathError(f"symlink target is forbidden: {literal.value}")
    actual_kind = _kind_from_mode(path_stat.st_mode)
    if actual_kind != expected_kind:
        raise PathTypeError(
            f"path has the wrong type: expected={expected_kind}, actual={actual_kind}, path={literal.value}"
        )
    if not _same_inode(path_stat, opened_stat):
        raise SecurePathRaceError(f"literal path no longer names the opened inode: {literal.value}")
    try:
        resolved = str(Path(literal.value).resolve(strict=True))
    except FileNotFoundError as error:
        raise SecurePathRaceError(f"path disappeared during strict resolution: {literal.value}") from error
    if resolved != literal.value:
        raise InvalidLiteralPathError(
            f"strictly resolved path must equal the supplied literal: literal={literal.value!r}, resolved={resolved!r}"
        )


def _open_absolute_directory(literal: _LiteralPath) -> tuple[int, os.stat_result]:
    fd = os.open("/", _DIRECTORY_OPEN_FLAGS)
    current_display = ""
    try:
        current_stat = os.fstat(fd)
        for component in literal.parts:
            current_display = f"{current_display}/{component}"
            child_fd, current_stat = _open_existing_directory_component(
                fd,
                component,
                display_path=current_display,
            )
            os.close(fd)
            fd = child_fd
        _verify_opened_literal(literal, current_stat, expected_kind="directory")
        return fd, current_stat
    except Exception:
        os.close(fd)
        raise


@contextmanager
def _open_root(root: str | os.PathLike[str]):
    literal = _parse_literal(root, label="root")
    if not literal.parts:
        raise InvalidLiteralPathError("the filesystem root '/' is too broad to use as a pinned root")
    fd, stat_result = _open_absolute_directory(literal)
    handle = _RootHandle(literal=literal, fd=fd, stat_result=stat_result)
    try:
        yield handle
    finally:
        os.close(fd)


def _walk_existing(
    root: _RootHandle,
    target: _LiteralPath,
    relative_parts: tuple[str, ...],
    *,
    expected_kind: PathKind,
) -> tuple[int, os.stat_result]:
    current_fd = os.dup(root.fd)
    current_parts = list(root.literal.parts)
    try:
        for component in relative_parts[:-1]:
            current_parts.append(component)
            display_path = "/" + "/".join(current_parts)
            child_fd, _ = _open_existing_directory_component(current_fd, component, display_path=display_path)
            os.close(current_fd)
            current_fd = child_fd
        leaf = relative_parts[-1]
        current_parts.append(leaf)
        leaf_fd, leaf_stat = _open_existing_leaf(
            current_fd,
            leaf,
            display_path="/" + "/".join(current_parts),
            expected_kind=expected_kind,
        )
        os.close(current_fd)
        current_fd = leaf_fd
        _verify_opened_literal(target, leaf_stat, expected_kind=expected_kind)
        return current_fd, leaf_stat
    except Exception:
        os.close(current_fd)
        raise


def _mkdir_component(
    parent_fd: int,
    name: str,
    *,
    display_path: str,
    mode: int,
    exclusive: bool,
) -> tuple[int, os.stat_result, bool]:
    created = False
    try:
        os.mkdir(name, mode=mode, dir_fd=parent_fd)
        created = True
    except FileExistsError as error:
        if exclusive:
            raise SecurePathAlreadyExistsError(
                f"exclusive directory target already exists, including as a symlink: {display_path}"
            ) from error

    try:
        child_fd, child_stat = _open_existing_directory_component(parent_fd, name, display_path=display_path)
    except SecurePathNotFoundError as error:
        if created:
            raise SecurePathRaceError(f"new directory disappeared before it could be opened: {display_path}") from error
        raise
    return child_fd, child_stat, created


def _walk_directories(
    root: _RootHandle,
    relative_parts: tuple[str, ...],
    *,
    create_parents: bool,
    directory_mode: int,
    exclusive_leaf: bool,
) -> tuple[int, os.stat_result, list[str]]:
    current_fd = os.dup(root.fd)
    current_stat = os.fstat(current_fd)
    current_parts = list(root.literal.parts)
    created: list[str] = []
    try:
        for index, component in enumerate(relative_parts):
            is_leaf = index == len(relative_parts) - 1
            current_parts.append(component)
            display_path = "/" + "/".join(current_parts)
            if is_leaf and exclusive_leaf:
                child_fd, current_stat, was_created = _mkdir_component(
                    current_fd,
                    component,
                    display_path=display_path,
                    mode=directory_mode,
                    exclusive=True,
                )
            else:
                try:
                    child_fd, current_stat = _open_existing_directory_component(
                        current_fd,
                        component,
                        display_path=display_path,
                    )
                    was_created = False
                except SecurePathNotFoundError:
                    if not is_leaf and not create_parents:
                        raise
                    child_fd, current_stat, was_created = _mkdir_component(
                        current_fd,
                        component,
                        display_path=display_path,
                        mode=directory_mode,
                        exclusive=False,
                    )
            if was_created:
                created.append("/".join(relative_parts[: index + 1]))
            os.close(current_fd)
            current_fd = child_fd
        return current_fd, current_stat, created
    except Exception:
        os.close(current_fd)
        raise


def _open_parent(
    root: _RootHandle,
    target: _LiteralPath,
    relative_parts: tuple[str, ...],
    *,
    create_parents: bool,
    directory_mode: int,
) -> tuple[int, os.stat_result, str, _LiteralPath, list[str]]:
    parent_parts = relative_parts[:-1]
    leaf = relative_parts[-1]
    if parent_parts:
        parent_fd, parent_stat, created = _walk_directories(
            root,
            parent_parts,
            create_parents=create_parents,
            directory_mode=directory_mode,
            exclusive_leaf=False,
        )
        parent_literal = _LiteralPath(
            value="/" + "/".join((*root.literal.parts, *parent_parts)),
            parts=(*root.literal.parts, *parent_parts),
        )
    else:
        parent_fd = os.dup(root.fd)
        parent_stat = os.fstat(parent_fd)
        created = []
        parent_literal = root.literal
    try:
        _verify_opened_literal(parent_literal, parent_stat, expected_kind="directory")
    except Exception:
        os.close(parent_fd)
        raise
    return parent_fd, parent_stat, leaf, parent_literal, created


def _target_lstat_if_present(parent_fd: int, leaf: str) -> os.stat_result | None:
    try:
        return os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None


def _base_receipt(
    *,
    operation: str,
    root: _RootHandle,
    target: _LiteralPath | None,
) -> PathReceipt:
    receipt: PathReceipt = {
        "ok": True,
        "operation": operation,
        "root": root.literal.value,
        "root_identity": _identity(root.stat_result, kind="directory"),
        "schema_version": _SCHEMA_VERSION,
        "symlink_policy": "reject_every_existing_component",
    }
    if target is not None:
        receipt["relative_path"] = "/".join(_relative_parts(root.literal, target))
        receipt["target"] = target.value
        receipt["strictly_within_root"] = True
    return receipt


def validate_pinned_root(root: str | os.PathLike[str]) -> PathReceipt:
    """Validate an existing absolute, canonical, symlink-free pinned root."""

    with _open_root(root) as handle:
        receipt = _base_receipt(operation="validate_pinned_root", root=handle, target=None)
        receipt["resolved_strict"] = True
        receipt["target_kind"] = "directory"
        return receipt


def validate_existing_path(
    root: str | os.PathLike[str],
    target: str | os.PathLike[str],
    *,
    expected_kind: PathKind,
) -> PathReceipt:
    """Validate an existing regular file or directory strictly below ``root``."""

    target_literal = _parse_literal(target, label="target")
    with _open_root(root) as handle:
        relative_parts = _relative_parts(handle.literal, target_literal)
        target_fd, target_stat = _walk_existing(
            handle,
            target_literal,
            relative_parts,
            expected_kind=expected_kind,
        )
        try:
            receipt = _base_receipt(operation="validate_existing_path", root=handle, target=target_literal)
            receipt.update(
                {
                    "resolved_strict": True,
                    "target_identity": _identity(target_stat, kind=expected_kind),
                    "target_kind": expected_kind,
                    "target_state": "existing",
                }
            )
            return receipt
        finally:
            os.close(target_fd)


def ensure_directory(
    root: str | os.PathLike[str],
    target: str | os.PathLike[str],
    *,
    create_parents: bool = True,
    directory_mode: int = 0o750,
) -> PathReceipt:
    """Create missing directories securely and validate the resulting leaf."""

    directory_mode = _validate_mode(directory_mode, label="directory_mode")
    target_literal = _parse_literal(target, label="target")
    with _open_root(root) as handle:
        relative_parts = _relative_parts(handle.literal, target_literal)
        target_fd, target_stat, created = _walk_directories(
            handle,
            relative_parts,
            create_parents=create_parents,
            directory_mode=directory_mode,
            exclusive_leaf=False,
        )
        try:
            _verify_opened_literal(target_literal, target_stat, expected_kind="directory")
            receipt = _base_receipt(operation="ensure_directory", root=handle, target=target_literal)
            receipt.update(
                {
                    "created": bool(created),
                    "created_directories": created,
                    "resolved_strict": True,
                    "target_identity": _identity(target_stat, kind="directory"),
                    "target_kind": "directory",
                    "target_state": "existing",
                }
            )
            return receipt
        finally:
            os.close(target_fd)


def create_directory_exclusive(
    root: str | os.PathLike[str],
    target: str | os.PathLike[str],
    *,
    create_parents: bool = True,
    directory_mode: int = 0o750,
) -> PathReceipt:
    """Atomically create a new leaf directory; fail if any leaf object exists."""

    directory_mode = _validate_mode(directory_mode, label="directory_mode")
    target_literal = _parse_literal(target, label="target")
    with _open_root(root) as handle:
        relative_parts = _relative_parts(handle.literal, target_literal)
        target_fd, target_stat, created = _walk_directories(
            handle,
            relative_parts,
            create_parents=create_parents,
            directory_mode=directory_mode,
            exclusive_leaf=True,
        )
        try:
            _verify_opened_literal(target_literal, target_stat, expected_kind="directory")
            receipt = _base_receipt(operation="create_directory_exclusive", root=handle, target=target_literal)
            receipt.update(
                {
                    "atomic_leaf_creation": True,
                    "created": True,
                    "created_directories": created,
                    "exclusive": True,
                    "resolved_strict": True,
                    "target_identity": _identity(target_stat, kind="directory"),
                    "target_kind": "directory",
                    "target_state": "existing",
                }
            )
            return receipt
        finally:
            os.close(target_fd)


def assert_file_target_absent(
    root: str | os.PathLike[str],
    target: str | os.PathLike[str],
    *,
    create_parents: bool = False,
    directory_mode: int = 0o750,
) -> PathReceipt:
    """Prove that a file target is absent at check time.

    This operation deliberately records ``reservation=False``.  It must not be
    used as a substitute for exclusive creation when another writer can race.
    """

    directory_mode = _validate_mode(directory_mode, label="directory_mode")
    target_literal = _parse_literal(target, label="target")
    with _open_root(root) as handle:
        relative_parts = _relative_parts(handle.literal, target_literal)
        parent_fd, parent_stat, leaf, parent_literal, created = _open_parent(
            handle,
            target_literal,
            relative_parts,
            create_parents=create_parents,
            directory_mode=directory_mode,
        )
        try:
            existing = _target_lstat_if_present(parent_fd, leaf)
            if existing is not None:
                object_type = "symlink" if stat.S_ISLNK(existing.st_mode) else "filesystem object"
                raise SecurePathAlreadyExistsError(
                    f"file target must not exist (including dangling symlinks); found {object_type}: "
                    f"{target_literal.value}"
                )
            _verify_opened_literal(parent_literal, parent_stat, expected_kind="directory")
            receipt = _base_receipt(operation="assert_file_target_absent", root=handle, target=target_literal)
            receipt.update(
                {
                    "created_directories": created,
                    "parent": parent_literal.value,
                    "parent_identity": _identity(parent_stat, kind="directory"),
                    "parent_resolved_strict": True,
                    "reservation": False,
                    "target_kind": "file",
                    "target_state": "absent",
                }
            )
            return receipt
        finally:
            os.close(parent_fd)


def _unlink_if_same_inode(parent_fd: int, leaf: str, expected: os.stat_result) -> None:
    observed = _target_lstat_if_present(parent_fd, leaf)
    if observed is not None and _same_inode(observed, expected):
        os.unlink(leaf, dir_fd=parent_fd)


def create_file_exclusive(
    root: str | os.PathLike[str],
    target: str | os.PathLike[str],
    *,
    create_parents: bool = False,
    directory_mode: int = 0o750,
    file_mode: int = 0o640,
) -> tuple[int, PathReceipt]:
    """Create a regular file with ``O_EXCL|O_NOFOLLOW`` and return its open fd.

    The caller owns the returned descriptor and must close it.  Creation is
    exclusive, but an empty file is visible before the caller writes content.
    Use :func:`write_file_exclusive` for atomic publication of complete bytes.
    """

    directory_mode = _validate_mode(directory_mode, label="directory_mode")
    file_mode = _validate_mode(file_mode, label="file_mode")
    target_literal = _parse_literal(target, label="target")
    with _open_root(root) as handle:
        relative_parts = _relative_parts(handle.literal, target_literal)
        parent_fd, parent_stat, leaf, parent_literal, created = _open_parent(
            handle,
            target_literal,
            relative_parts,
            create_parents=create_parents,
            directory_mode=directory_mode,
        )
        file_fd: int | None = None
        file_stat: os.stat_result | None = None
        keep_file = False
        try:
            _verify_opened_literal(parent_literal, parent_stat, expected_kind="directory")
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW
            try:
                file_fd = os.open(leaf, flags, file_mode, dir_fd=parent_fd)
            except FileExistsError as error:
                raise SecurePathAlreadyExistsError(
                    f"exclusive file target already exists, including as a symlink: {target_literal.value}"
                ) from error
            file_stat = os.fstat(file_fd)
            if not stat.S_ISREG(file_stat.st_mode):
                raise PathTypeError(f"exclusive file target is not regular: {target_literal.value}")
            _verify_opened_literal(target_literal, file_stat, expected_kind="file")
            receipt = _base_receipt(operation="create_file_exclusive", root=handle, target=target_literal)
            receipt.update(
                {
                    "atomic_leaf_creation": True,
                    "content_publication": "caller_managed",
                    "created": True,
                    "created_directories": created,
                    "exclusive": True,
                    "resolved_strict": True,
                    "target_identity": _identity(file_stat, kind="file"),
                    "target_kind": "file",
                    "target_state": "existing",
                }
            )
            keep_file = True
            return file_fd, receipt
        finally:
            if not keep_file:
                if file_fd is not None:
                    os.close(file_fd)
                if file_stat is not None:
                    _unlink_if_same_inode(parent_fd, leaf, file_stat)
            os.close(parent_fd)


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    offset = 0
    while offset < len(view):
        try:
            written = os.write(fd, view[offset:])
        except InterruptedError:
            continue
        if written <= 0:
            raise OSError("short write while creating secure file")
        offset += written


def write_file_exclusive(
    root: str | os.PathLike[str],
    target: str | os.PathLike[str],
    data: bytes,
    *,
    create_parents: bool = False,
    directory_mode: int = 0o750,
    file_mode: int = 0o640,
) -> PathReceipt:
    """Atomically publish complete bytes without replacing any target object.

    Bytes are written and fsynced to a private inode in the target directory.
    A same-directory hard link then publishes that inode under ``target``.  The
    link operation is atomic and fails if a competing file or symlink won the
    name.  The temporary link is removed after publication.
    """

    if not isinstance(data, bytes):
        raise TypeError("data must be bytes")
    directory_mode = _validate_mode(directory_mode, label="directory_mode")
    file_mode = _validate_mode(file_mode, label="file_mode")
    target_literal = _parse_literal(target, label="target")
    with _open_root(root) as handle:
        relative_parts = _relative_parts(handle.literal, target_literal)
        parent_fd, parent_stat, leaf, parent_literal, created = _open_parent(
            handle,
            target_literal,
            relative_parts,
            create_parents=create_parents,
            directory_mode=directory_mode,
        )
        temp_fd: int | None = None
        temp_name: str | None = None
        temp_stat: os.stat_result | None = None
        published = False
        completed = False
        try:
            _verify_opened_literal(parent_literal, parent_stat, expected_kind="directory")
            if _target_lstat_if_present(parent_fd, leaf) is not None:
                raise SecurePathAlreadyExistsError(
                    f"exclusive file target already exists, including as a symlink: {target_literal.value}"
                )

            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW
            for _ in range(16):
                candidate = f".cwam-secure-{os.getpid()}-{secrets.token_hex(12)}.tmp"
                try:
                    temp_fd = os.open(candidate, flags, file_mode, dir_fd=parent_fd)
                except FileExistsError:
                    continue
                temp_name = candidate
                break
            if temp_fd is None or temp_name is None:
                raise SecurePathError("could not allocate a private temporary file after 16 attempts")

            _write_all(temp_fd, data)
            os.fsync(temp_fd)
            temp_stat = os.fstat(temp_fd)
            if not stat.S_ISREG(temp_stat.st_mode):
                raise PathTypeError("private publication inode is not a regular file")

            _verify_opened_literal(parent_literal, parent_stat, expected_kind="directory")
            if _target_lstat_if_present(parent_fd, leaf) is not None:
                raise SecurePathAlreadyExistsError(
                    f"a competing object appeared at exclusive file target: {target_literal.value}"
                )
            try:
                os.link(
                    temp_name,
                    leaf,
                    src_dir_fd=parent_fd,
                    dst_dir_fd=parent_fd,
                    follow_symlinks=False,
                )
            except FileExistsError as error:
                raise SecurePathAlreadyExistsError(
                    f"a competing object won exclusive file target: {target_literal.value}"
                ) from error
            published = True
            os.unlink(temp_name, dir_fd=parent_fd)
            temp_name = None
            os.fsync(parent_fd)

            published_stat = _target_lstat_if_present(parent_fd, leaf)
            if published_stat is None or not _same_inode(published_stat, temp_stat):
                raise SecurePathRaceError(f"published file identity changed: {target_literal.value}")
            _verify_opened_literal(target_literal, temp_stat, expected_kind="file")
            receipt = _base_receipt(operation="write_file_exclusive", root=handle, target=target_literal)
            receipt.update(
                {
                    "atomic_content_publication": True,
                    "created": True,
                    "created_directories": created,
                    "exclusive": True,
                    "resolved_strict": True,
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "target_identity": _identity(temp_stat, kind="file"),
                    "target_kind": "file",
                    "target_state": "existing",
                }
            )
            completed = True
            return receipt
        finally:
            if not completed and published and temp_stat is not None:
                _unlink_if_same_inode(parent_fd, leaf, temp_stat)
            if temp_fd is not None:
                os.close(temp_fd)
            if temp_name is not None:
                try:
                    os.unlink(temp_name, dir_fd=parent_fd)
                except FileNotFoundError:
                    pass
            os.close(parent_fd)


def write_json_file_exclusive(
    root: str | os.PathLike[str],
    target: str | os.PathLike[str],
    payload: Any,
    *,
    create_parents: bool = False,
    directory_mode: int = 0o750,
    file_mode: int = 0o640,
) -> PathReceipt:
    """Normalize JSON and publish it with :func:`write_file_exclusive`."""

    data = normalized_json(payload).encode("utf-8")
    receipt = write_file_exclusive(
        root,
        target,
        data,
        create_parents=create_parents,
        directory_mode=directory_mode,
        file_mode=file_mode,
    )
    receipt["content_type"] = "application/json"
    return receipt


def normalized_json(payload: Any) -> str:
    """Return deterministic UTF-8 JSON with exactly one trailing newline."""

    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"


def _octal_mode(value: str) -> int:
    try:
        parsed = int(value, 8)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid octal mode: {value!r}") from error
    try:
        return _validate_mode(parsed, label="mode")
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _add_root_and_path(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", required=True)
    parser.add_argument("--path", required=True)


def _add_creation_options(parser: argparse.ArgumentParser, *, include_file_mode: bool = False) -> None:
    parser.add_argument("--create-parents", action="store_true")
    parser.add_argument("--directory-mode", type=_octal_mode, default=0o750)
    if include_file_mode:
        parser.add_argument("--file-mode", type=_octal_mode, default=0o640)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_root_parser = subparsers.add_parser("validate-root")
    validate_root_parser.add_argument("--root", required=True)

    validate_existing_parser = subparsers.add_parser("validate-existing")
    _add_root_and_path(validate_existing_parser)
    validate_existing_parser.add_argument("--kind", choices=("directory", "file"), required=True)

    ensure_dir_parser = subparsers.add_parser("ensure-dir")
    _add_root_and_path(ensure_dir_parser)
    _add_creation_options(ensure_dir_parser)

    create_dir_parser = subparsers.add_parser("create-dir-exclusive")
    _add_root_and_path(create_dir_parser)
    _add_creation_options(create_dir_parser)

    assert_file_parser = subparsers.add_parser("assert-new-file")
    _add_root_and_path(assert_file_parser)
    _add_creation_options(assert_file_parser)

    write_file_parser = subparsers.add_parser("write-file-exclusive")
    _add_root_and_path(write_file_parser)
    _add_creation_options(write_file_parser, include_file_mode=True)
    write_file_parser.add_argument("--text", required=True)

    write_json_parser = subparsers.add_parser("write-json-exclusive")
    _add_root_and_path(write_json_parser)
    _add_creation_options(write_json_parser, include_file_mode=True)
    write_json_parser.add_argument("--json", required=True)

    return parser.parse_args(argv)


def _dispatch(args: argparse.Namespace) -> PathReceipt:
    if args.command == "validate-root":
        return validate_pinned_root(args.root)
    if args.command == "validate-existing":
        return validate_existing_path(args.root, args.path, expected_kind=args.kind)
    if args.command == "ensure-dir":
        return ensure_directory(
            args.root,
            args.path,
            create_parents=args.create_parents,
            directory_mode=args.directory_mode,
        )
    if args.command == "create-dir-exclusive":
        return create_directory_exclusive(
            args.root,
            args.path,
            create_parents=args.create_parents,
            directory_mode=args.directory_mode,
        )
    if args.command == "assert-new-file":
        return assert_file_target_absent(
            args.root,
            args.path,
            create_parents=args.create_parents,
            directory_mode=args.directory_mode,
        )
    if args.command == "write-file-exclusive":
        return write_file_exclusive(
            args.root,
            args.path,
            args.text.encode("utf-8"),
            create_parents=args.create_parents,
            directory_mode=args.directory_mode,
            file_mode=args.file_mode,
        )
    if args.command == "write-json-exclusive":
        try:
            payload = json.loads(args.json)
        except json.JSONDecodeError as error:
            raise SecurePathError(f"--json is not valid JSON: {error}") from error
        return write_json_file_exclusive(
            args.root,
            args.path,
            payload,
            create_parents=args.create_parents,
            directory_mode=args.directory_mode,
            file_mode=args.file_mode,
        )
    raise AssertionError(f"unhandled command: {args.command}")


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and emit one normalized JSON receipt or error object."""

    args = _parse_args(argv)
    try:
        receipt = _dispatch(args)
    except (SecurePathError, OSError, TypeError, ValueError) as error:
        sys.stderr.write(
            normalized_json(
                {
                    "error": str(error),
                    "error_type": type(error).__name__,
                    "ok": False,
                    "schema_version": _SCHEMA_VERSION,
                }
            )
        )
        return 2
    sys.stdout.write(normalized_json(receipt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
