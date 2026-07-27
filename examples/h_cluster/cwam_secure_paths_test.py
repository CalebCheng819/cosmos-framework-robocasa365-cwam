# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from examples.h_cluster import cwam_secure_paths
from examples.h_cluster.cwam_secure_paths import (
    InvalidLiteralPathError,
    PathEscapeError,
    PathTypeError,
    SecurePathAlreadyExistsError,
    SecurePathError,
    SecurePathNotFoundError,
    SecurePathRaceError,
    SymlinkPathError,
    assert_file_target_absent,
    create_directory_exclusive,
    create_file_exclusive,
    ensure_directory,
    main,
    normalized_json,
    validate_existing_path,
    validate_pinned_root,
    write_file_exclusive,
    write_json_file_exclusive,
)


def _new_root(tmp_path: Path) -> Path:
    root = tmp_path / "pinned"
    root.mkdir()
    return root


def _temporary_entries(directory: Path) -> list[Path]:
    return sorted(directory.glob(".cwam-secure-*.tmp"))


def test_validate_pinned_root_returns_normalized_identity(tmp_path: Path) -> None:
    root = _new_root(tmp_path)

    receipt = validate_pinned_root(str(root))

    assert receipt["ok"] is True
    assert receipt["operation"] == "validate_pinned_root"
    assert receipt["root"] == str(root)
    assert receipt["root_identity"]["kind"] == "directory"
    assert receipt["resolved_strict"] is True


@pytest.mark.parametrize(
    "bad_path",
    [
        "relative/path",
        "/tmp/../tmp/path",
        "/tmp/./path",
        "/tmp//path",
        "/tmp/path/",
    ],
)
def test_rejects_noncanonical_literal_paths(bad_path: str) -> None:
    with pytest.raises(InvalidLiteralPathError):
        validate_pinned_root(bad_path)


def test_rejects_filesystem_root_as_pinned_root() -> None:
    with pytest.raises(InvalidLiteralPathError, match="too broad"):
        validate_pinned_root("/")


def test_rejects_symlink_in_ancestor_of_pinned_root(tmp_path: Path) -> None:
    actual_parent = tmp_path / "actual"
    root = actual_parent / "root"
    root.mkdir(parents=True)
    alias = tmp_path / "alias"
    alias.symlink_to(actual_parent, target_is_directory=True)

    with pytest.raises(SymlinkPathError, match="symlink path component"):
        validate_pinned_root(str(alias / "root"))


def test_rejects_pinned_root_that_is_a_leaf_symlink(tmp_path: Path) -> None:
    actual = tmp_path / "actual"
    actual.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(actual, target_is_directory=True)

    with pytest.raises(SymlinkPathError):
        validate_pinned_root(str(alias))


def test_rejects_dangling_symlink_in_target_ancestor(tmp_path: Path) -> None:
    root = _new_root(tmp_path)
    dangling_parent = root / "evidence"
    dangling_parent.symlink_to(root / "missing-parent", target_is_directory=True)

    with pytest.raises(SymlinkPathError):
        ensure_directory(str(root), str(dangling_parent / "run"))


def test_rejects_existing_symlink_in_target_ancestor(tmp_path: Path) -> None:
    root = _new_root(tmp_path)
    actual = root / "actual"
    actual.mkdir()
    alias = root / "logs"
    alias.symlink_to(actual, target_is_directory=True)

    with pytest.raises(SymlinkPathError):
        ensure_directory(str(root), str(alias / "run"))


@pytest.mark.parametrize("dangling", [False, True])
def test_rejects_leaf_symlink_for_directory(tmp_path: Path, dangling: bool) -> None:
    root = _new_root(tmp_path)
    actual = root / "actual"
    if not dangling:
        actual.mkdir()
    leaf = root / "run"
    leaf.symlink_to(actual, target_is_directory=True)

    with pytest.raises(SymlinkPathError):
        ensure_directory(str(root), str(leaf))


@pytest.mark.parametrize("dangling", [False, True])
def test_absent_file_check_rejects_leaf_symlink(tmp_path: Path, dangling: bool) -> None:
    root = _new_root(tmp_path)
    actual = root / "actual.log"
    if not dangling:
        actual.write_text("old")
    leaf = root / "train.log"
    leaf.symlink_to(actual)

    with pytest.raises(SecurePathAlreadyExistsError, match="symlink"):
        assert_file_target_absent(str(root), str(leaf))


def test_rejects_target_outside_root_and_prefix_collision(tmp_path: Path) -> None:
    root = _new_root(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    prefix_collision = tmp_path / "pinned-other"
    prefix_collision.mkdir()

    with pytest.raises(PathEscapeError):
        ensure_directory(str(root), str(outside / "run"))
    with pytest.raises(PathEscapeError):
        ensure_directory(str(root), str(prefix_collision / "run"))
    with pytest.raises(PathEscapeError):
        ensure_directory(str(root), str(root))


def test_ensure_directory_creates_parents_and_returns_receipt(tmp_path: Path) -> None:
    root = _new_root(tmp_path)
    target = root / "formal_evidence" / "experiment" / "run"

    receipt = ensure_directory(str(root), str(target), create_parents=True, directory_mode=0o710)

    assert target.is_dir()
    assert receipt["created"] is True
    assert receipt["created_directories"] == [
        "formal_evidence",
        "formal_evidence/experiment",
        "formal_evidence/experiment/run",
    ]
    assert receipt["relative_path"] == "formal_evidence/experiment/run"
    assert receipt["resolved_strict"] is True
    assert receipt["target_identity"]["kind"] == "directory"

    second = ensure_directory(str(root), str(target), create_parents=False)
    assert second["created"] is False
    assert second["created_directories"] == []


def test_ensure_directory_requires_missing_parent_opt_in(tmp_path: Path) -> None:
    root = _new_root(tmp_path)
    target = root / "missing" / "leaf"

    with pytest.raises(SecurePathNotFoundError):
        ensure_directory(str(root), str(target), create_parents=False)


def test_ensure_directory_rejects_regular_file_component(tmp_path: Path) -> None:
    root = _new_root(tmp_path)
    regular = root / "logs"
    regular.write_text("not a directory")

    with pytest.raises(PathTypeError):
        ensure_directory(str(root), str(regular / "run"), create_parents=True)


def test_create_directory_exclusive_is_atomic_at_leaf(tmp_path: Path) -> None:
    root = _new_root(tmp_path)
    target = root / "runs" / "experiment"

    receipt = create_directory_exclusive(str(root), str(target), create_parents=True)

    assert receipt["atomic_leaf_creation"] is True
    assert receipt["exclusive"] is True
    assert target.is_dir()
    with pytest.raises(SecurePathAlreadyExistsError):
        create_directory_exclusive(str(root), str(target), create_parents=True)


def test_create_directory_exclusive_rejects_dangling_leaf_symlink(tmp_path: Path) -> None:
    root = _new_root(tmp_path)
    target = root / "reservation"
    target.symlink_to(root / "does-not-exist", target_is_directory=True)

    with pytest.raises(SecurePathAlreadyExistsError):
        create_directory_exclusive(str(root), str(target))


def test_validate_existing_file_and_directory(tmp_path: Path) -> None:
    root = _new_root(tmp_path)
    directory = root / "evidence"
    directory.mkdir()
    regular_file = directory / "receipt.json"
    regular_file.write_text("{}\n")

    dir_receipt = validate_existing_path(str(root), str(directory), expected_kind="directory")
    file_receipt = validate_existing_path(str(root), str(regular_file), expected_kind="file")

    assert dir_receipt["target_kind"] == "directory"
    assert file_receipt["target_kind"] == "file"
    assert file_receipt["target_identity"]["size"] == 3
    with pytest.raises(PathTypeError):
        validate_existing_path(str(root), str(regular_file), expected_kind="directory")


def test_absent_file_check_can_create_only_verified_parents(tmp_path: Path) -> None:
    root = _new_root(tmp_path)
    target = root / "logs" / "formal" / "train.log"

    receipt = assert_file_target_absent(str(root), str(target), create_parents=True)

    assert receipt["target_state"] == "absent"
    assert receipt["reservation"] is False
    assert receipt["parent_resolved_strict"] is True
    assert receipt["created_directories"] == ["logs", "logs/formal"]
    assert not target.exists()


@pytest.mark.parametrize("object_kind", ["file", "directory"])
def test_absent_file_check_rejects_any_existing_leaf(tmp_path: Path, object_kind: str) -> None:
    root = _new_root(tmp_path)
    target = root / "train.log"
    if object_kind == "file":
        target.write_text("old")
    else:
        target.mkdir()

    with pytest.raises(SecurePathAlreadyExistsError):
        assert_file_target_absent(str(root), str(target))


def test_create_file_exclusive_returns_owned_open_descriptor(tmp_path: Path) -> None:
    root = _new_root(tmp_path)
    target = root / "logs" / "train.log"

    fd, receipt = create_file_exclusive(
        str(root),
        str(target),
        create_parents=True,
        directory_mode=0o700,
        file_mode=0o600,
    )
    try:
        os.write(fd, b"first line\n")
        os.fsync(fd)
    finally:
        os.close(fd)

    assert target.read_bytes() == b"first line\n"
    assert receipt["exclusive"] is True
    assert receipt["content_publication"] == "caller_managed"
    assert receipt["target_identity"]["mode"] == "0600"
    with pytest.raises(SecurePathAlreadyExistsError):
        create_file_exclusive(str(root), str(target))


def test_create_file_exclusive_rejects_dangling_symlink(tmp_path: Path) -> None:
    root = _new_root(tmp_path)
    target = root / "train.log"
    target.symlink_to(root / "missing.log")

    with pytest.raises(SecurePathAlreadyExistsError):
        create_file_exclusive(str(root), str(target))


def test_write_file_exclusive_atomically_publishes_complete_bytes(tmp_path: Path) -> None:
    root = _new_root(tmp_path)
    target = root / "evidence" / "receipt.json"
    data = b'{"status":"accepted"}\n'

    receipt = write_file_exclusive(
        str(root),
        str(target),
        data,
        create_parents=True,
        directory_mode=0o700,
        file_mode=0o600,
    )

    assert target.read_bytes() == data
    assert receipt["atomic_content_publication"] is True
    assert receipt["exclusive"] is True
    assert receipt["sha256"] == "2df5de9000729506e6c423e4849c62052b96e5a70d851802bf1fd318ede19f1a"
    assert receipt["target_identity"]["size"] == len(data)
    assert _temporary_entries(target.parent) == []


def test_write_json_file_exclusive_normalizes_payload(tmp_path: Path) -> None:
    root = _new_root(tmp_path)
    target = root / "receipt.json"
    payload = {"z": 1, "a": {"β": True}}

    receipt = write_json_file_exclusive(str(root), str(target), payload)

    assert target.read_text() == '{"a":{"β":true},"z":1}\n'
    assert receipt["content_type"] == "application/json"
    assert normalized_json(payload) == target.read_text()


def test_write_file_exclusive_never_overwrites_existing_file(tmp_path: Path) -> None:
    root = _new_root(tmp_path)
    target = root / "receipt.json"
    target.write_bytes(b"original")

    with pytest.raises(SecurePathAlreadyExistsError):
        write_file_exclusive(str(root), str(target), b"replacement")

    assert target.read_bytes() == b"original"
    assert _temporary_entries(root) == []


def test_write_file_exclusive_never_overwrites_dangling_symlink(tmp_path: Path) -> None:
    root = _new_root(tmp_path)
    target = root / "receipt.json"
    target.symlink_to(root / "missing")

    with pytest.raises(SecurePathAlreadyExistsError):
        write_file_exclusive(str(root), str(target), b"replacement")

    assert target.is_symlink()
    assert _temporary_entries(root) == []


def test_atomic_publish_loser_preserves_competing_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _new_root(tmp_path)
    target = root / "receipt.json"
    real_link = os.link

    def competing_link(
        source: str,
        destination: str,
        *,
        src_dir_fd: int,
        dst_dir_fd: int,
        follow_symlinks: bool,
    ) -> None:
        competitor_fd = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
            dir_fd=dst_dir_fd,
        )
        try:
            os.write(competitor_fd, b"competitor")
        finally:
            os.close(competitor_fd)
        real_link(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
            follow_symlinks=follow_symlinks,
        )

    monkeypatch.setattr(cwam_secure_paths.os, "link", competing_link)

    with pytest.raises(SecurePathAlreadyExistsError, match="competing object"):
        write_file_exclusive(str(root), str(target), b"ours")

    assert target.read_bytes() == b"competitor"
    assert _temporary_entries(root) == []


def test_atomic_publish_detects_parent_replacement_and_cleans_detached_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _new_root(tmp_path)
    parent = root / "evidence"
    parent.mkdir()
    moved_parent = root / "evidence-moved"
    target = parent / "receipt.json"
    real_link = os.link
    replaced = False

    def replacing_link(
        source: str,
        destination: str,
        *,
        src_dir_fd: int,
        dst_dir_fd: int,
        follow_symlinks: bool,
    ) -> None:
        nonlocal replaced
        if not replaced:
            parent.rename(moved_parent)
            parent.mkdir()
            replaced = True
        real_link(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
            follow_symlinks=follow_symlinks,
        )

    monkeypatch.setattr(cwam_secure_paths.os, "link", replacing_link)

    with pytest.raises(SecurePathRaceError):
        write_file_exclusive(str(root), str(target), b"never publish at stale path")

    assert not target.exists()
    assert not (moved_parent / "receipt.json").exists()
    assert _temporary_entries(moved_parent) == []


def test_cli_emits_one_normalized_json_receipt(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _new_root(tmp_path)
    target = root / "formal_evidence" / "run"

    return_code = main(
        [
            "ensure-dir",
            "--root",
            str(root),
            "--path",
            str(target),
            "--create-parents",
            "--directory-mode",
            "0700",
        ]
    )
    captured = capsys.readouterr()

    assert return_code == 0
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload["target"] == str(target)
    assert payload["target_identity"]["mode"] == "0700"
    assert captured.out == normalized_json(payload)


def test_cli_error_is_normalized_json_and_fail_closed(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _new_root(tmp_path)
    outside = tmp_path / "outside" / "run"

    return_code = main(["ensure-dir", "--root", str(root), "--path", str(outside), "--create-parents"])
    captured = capsys.readouterr()

    assert return_code == 2
    assert captured.out == ""
    payload = json.loads(captured.err)
    assert payload["ok"] is False
    assert payload["error_type"] == "PathEscapeError"
    assert captured.err == normalized_json(payload)
    assert not outside.exists()


def test_public_functions_reject_invalid_modes_before_creation(tmp_path: Path) -> None:
    root = _new_root(tmp_path)
    target = root / "run"

    with pytest.raises(ValueError):
        ensure_directory(str(root), str(target), directory_mode=0o1000)
    with pytest.raises(ValueError):
        create_file_exclusive(str(root), str(root / "log"), file_mode=-1)
    assert not target.exists()


def test_normalized_json_rejects_nonserializable_payload() -> None:
    with pytest.raises(TypeError):
        normalized_json({"bad": SecurePathError})
