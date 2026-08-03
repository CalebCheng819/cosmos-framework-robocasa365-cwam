#!/usr/bin/env python3
"""Publish and verify the complete DSW Cosmos environment as one tar archive."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import stat
import subprocess
import tarfile
import uuid
from pathlib import Path, PurePosixPath
from typing import Any

PROJECT_ROOT = Path("/mnt/workspace/cwam_cosmos3_robocasa365_eval")
FORMAL_SOURCE = Path(
    "/tmp/cwam_cosmos3_robocasa365_eval_nvme/inputs/envs/"
    "cosmos-framework-09f23119-cu128-train"
)
FORMAL_ARCHIVE = (
    PROJECT_ROOT
    / "inputs/envs/archives/"
    "cosmos-framework-09f23119-cu128-train-r22-complete.tar"
)
FORMAL_RECEIPT = (
    PROJECT_ROOT
    / "control/receipts/"
    "cosmos-framework-09f23119-cu128-train-r22-complete-archive-publication.json"
)
FORMAL_MATERIALIZATION_RECEIPT = (
    PROJECT_ROOT
    / "control/receipts/cosmos-nvme-native-env-r22-resume-a2-20260802.json"
)
READY_MARKER = ".cwam_env_ready.json"
_SHA256_LENGTH = 64


class EnvironmentArchiveError(RuntimeError):
    """Raised when the environment archive contract is violated."""


def _sha256(path: Path) -> tuple[int, str]:
    before = path.lstat()
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise EnvironmentArchiveError(f"not a regular non-symlink file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    after = path.lstat()
    def identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
        return (
            value.st_dev,
            value.st_ino,
            value.st_mode,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )
    if identity(before) != identity(after):
        raise EnvironmentArchiveError(f"file changed while hashing: {path}")
    return before.st_size, digest.hexdigest()


def _canonical_directory(path: Path, label: str) -> Path:
    supplied = Path(os.path.abspath(path.expanduser()))
    try:
        canonical = supplied.resolve(strict=True)
        metadata = supplied.lstat()
    except FileNotFoundError:
        raise EnvironmentArchiveError(f"{label} is missing: {supplied}") from None
    if supplied != canonical or stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise EnvironmentArchiveError(f"{label} must be a canonical non-symlink directory: {supplied}")
    return canonical


def _canonical_file(path: Path, label: str, *, mode: int | None = None) -> Path:
    supplied = Path(os.path.abspath(path.expanduser()))
    try:
        canonical = supplied.resolve(strict=True)
        metadata = supplied.lstat()
    except FileNotFoundError:
        raise EnvironmentArchiveError(f"{label} is missing: {supplied}") from None
    if supplied != canonical or stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise EnvironmentArchiveError(f"{label} must be a canonical non-symlink file: {supplied}")
    if mode is not None and stat.S_IMODE(metadata.st_mode) != mode:
        raise EnvironmentArchiveError(
            f"{label} mode mismatch: expected={oct(mode)} actual={oct(stat.S_IMODE(metadata.st_mode))}"
        )
    return canonical


def _inventory(root: Path) -> dict[str, Any]:
    root = _canonical_directory(root, "environment root")
    digest = hashlib.sha256()
    regular_files = 0
    directories = 0
    symlinks = 0
    regular_file_bytes = 0
    paths = [root, *root.rglob("*")]
    paths.sort(key=lambda value: "" if value == root else value.relative_to(root).as_posix())
    for path in paths:
        relative = "." if path == root else path.relative_to(root).as_posix()
        metadata = path.lstat()
        mode = stat.S_IMODE(metadata.st_mode)
        if stat.S_ISDIR(metadata.st_mode):
            kind = "directory"
            directories += 1
            size = 0
            target = ""
        elif stat.S_ISREG(metadata.st_mode):
            kind = "regular_file"
            regular_files += 1
            regular_file_bytes += metadata.st_size
            size = metadata.st_size
            target = ""
        elif stat.S_ISLNK(metadata.st_mode):
            kind = "symlink"
            symlinks += 1
            size = 0
            target = os.readlink(path)
        else:
            raise EnvironmentArchiveError(f"special environment entry is forbidden: {path}")
        digest.update(f"{kind}\t{mode:04o}\t{size}\t{target}\t{relative}\n".encode())
    return {
        "regular_files": regular_files,
        "directories_including_root": directories,
        "symlinks": symlinks,
        "regular_file_bytes": regular_file_bytes,
        "metadata_inventory_sha256": digest.hexdigest(),
    }


def _tar_command(root: Path, output: str) -> list[str]:
    return [
        "tar",
        "--sort=name",
        "--format=posix",
        "--numeric-owner",
        "--owner=0",
        "--group=0",
        "--mtime=@0",
        "--pax-option=exthdr.name=%d/PaxHeaders/%f,delete=atime,delete=ctime",
        "-cf",
        output,
        "-C",
        str(root),
        ".",
    ]


def _tar_stream_sha256(root: Path) -> str:
    process = subprocess.Popen(_tar_command(root, "-"), stdout=subprocess.PIPE)
    if process.stdout is None:
        raise EnvironmentArchiveError("tar stdout was not created")
    digest = hashlib.sha256()
    while chunk := process.stdout.read(16 * 1024 * 1024):
        digest.update(chunk)
    process.stdout.close()
    status = process.wait()
    if status != 0:
        raise EnvironmentArchiveError(f"deterministic tar stream failed: status={status}")
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_read_only_file(temporary: Path, final: Path) -> None:
    if os.path.lexists(final):
        raise EnvironmentArchiveError(f"refusing to overwrite publication: {final}")
    os.link(temporary, final, follow_symlinks=False)
    temporary.unlink()
    _fsync_directory(final.parent)


def _write_receipt(path: Path, value: dict[str, Any]) -> tuple[int, str]:
    if os.path.lexists(path):
        raise EnvironmentArchiveError(f"refusing to overwrite receipt: {path}")
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    temporary = path.with_name(f".{path.name}.incomplete-{uuid.uuid4().hex}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fchmod(stream.fileno(), 0o444)
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)
    _publish_read_only_file(temporary, path)
    return _sha256(path)


def publish(
    *,
    source: Path,
    archive: Path,
    receipt: Path,
    materialization_receipt: Path,
    publisher_source: Path,
) -> dict[str, Any]:
    source = _canonical_directory(source, "source environment")
    archive = Path(os.path.abspath(archive.expanduser()))
    receipt = Path(os.path.abspath(receipt.expanduser()))
    archive_parent = _canonical_directory(archive.parent, "archive parent")
    receipt_parent = _canonical_directory(receipt.parent, "receipt parent")
    materialization_receipt = _canonical_file(
        materialization_receipt,
        "source materialization receipt",
        mode=0o444,
    )
    publisher_source = _canonical_file(publisher_source, "publisher source")
    if archive.parent != archive_parent or receipt.parent != receipt_parent:
        raise EnvironmentArchiveError("publication parents must be canonical")
    if os.path.lexists(archive) or os.path.lexists(receipt):
        raise EnvironmentArchiveError("archive or receipt already exists")

    marker = _canonical_file(source / READY_MARKER, "source ready marker", mode=0o444)
    inventory_before = _inventory(source)
    marker_size, marker_sha = _sha256(marker)
    materialization_size, materialization_sha = _sha256(materialization_receipt)
    publisher_size, publisher_sha = _sha256(publisher_source)
    source_tar_before = _tar_stream_sha256(source)

    temporary = archive.with_name(f".{archive.name}.incomplete-{uuid.uuid4().hex}")
    if os.path.lexists(temporary):
        raise EnvironmentArchiveError(f"archive temporary already exists: {temporary}")
    try:
        subprocess.run(_tar_command(source, str(temporary)), check=True)
        descriptor = os.open(temporary, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.chmod(temporary, 0o444)
        archive_size, archive_sha = _sha256(temporary)
        source_tar_after = _tar_stream_sha256(source)
        inventory_after = _inventory(source)
        marker_size_after, marker_sha_after = _sha256(marker)
        if not (
            source_tar_before == archive_sha == source_tar_after
            and inventory_before == inventory_after
            and (marker_size, marker_sha) == (marker_size_after, marker_sha_after)
        ):
            raise EnvironmentArchiveError("source environment changed while the archive was built")
        _publish_read_only_file(temporary, archive)
    finally:
        if os.path.lexists(temporary):
            temporary.unlink()

    published_size, published_sha = _sha256(archive)
    if (published_size, published_sha) != (archive_size, archive_sha):
        raise EnvironmentArchiveError("published archive changed after commit")
    value = {
        "schema_version": 1,
        "kind": "cwam_cosmos_complete_environment_archive_publication",
        "status": "published",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "source_environment": {
            "path": str(source),
            "inventory": inventory_before,
            "deterministic_tar_sha256_before": source_tar_before,
            "deterministic_tar_sha256_after": source_tar_after,
            "stable_during_publication": True,
        },
        "source_materialization_receipt": {
            "path": str(materialization_receipt),
            "size_bytes": materialization_size,
            "sha256": materialization_sha,
        },
        "ready_marker": {
            "relative_path": READY_MARKER,
            "mode": "0444",
            "size_bytes": marker_size,
            "sha256": marker_sha,
        },
        "archive": {
            "path": str(archive),
            "format": "uncompressed-posix-tar-normalized-mtime",
            "root_member": ".",
            "mode": "0444",
            "size_bytes": archive_size,
            "sha256": archive_sha,
        },
        "publication": {
            "archive_exclusive_hard_link": True,
            "receipt_exclusive_hard_link": True,
            "receipt_is_commit_point": True,
            "overwrite": False,
        },
        "publisher": {
            "path": str(publisher_source),
            "size_bytes": publisher_size,
            "sha256": publisher_sha,
        },
    }
    receipt_size, receipt_sha = _write_receipt(receipt, value)
    result = {
        "status": "published",
        "archive": str(archive),
        "archive_size_bytes": archive_size,
        "archive_sha256": archive_sha,
        "receipt": str(receipt),
        "receipt_size_bytes": receipt_size,
        "receipt_sha256": receipt_sha,
        "ready_marker_sha256": marker_sha,
        "inventory": inventory_before,
    }
    print(json.dumps(result, sort_keys=True))
    return result


def _safe_tar_path(value: str, label: str, *, allow_root: bool = False) -> PurePosixPath:
    path = PurePosixPath(value)
    if value == "." and allow_root:
        return path
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise EnvironmentArchiveError(f"unsafe {label}: {value!r}")
    return path


def _validate_archive_members(archive: Path) -> int:
    names: set[str] = set()
    count = 0
    with tarfile.open(archive, "r:") as bundle:
        for member in bundle:
            _safe_tar_path(member.name, "archive member", allow_root=True)
            if member.name in names:
                raise EnvironmentArchiveError(f"duplicate archive member: {member.name!r}")
            names.add(member.name)
            if not (member.isdir() or member.isfile() or member.issym() or member.islnk()):
                raise EnvironmentArchiveError(f"special archive member is forbidden: {member.name!r}")
            if member.issym() or member.islnk():
                _safe_tar_path(member.linkname, "archive link target")
            count += 1
    if "." not in names and "./" not in names:
        raise EnvironmentArchiveError("archive root member is missing")
    return count


def _load_receipt(path: Path, expected_sha256: str) -> tuple[dict[str, Any], str]:
    if len(expected_sha256) != _SHA256_LENGTH:
        raise EnvironmentArchiveError("expected receipt SHA-256 is malformed")
    path = _canonical_file(path, "archive publication receipt", mode=0o444)
    _, actual_sha = _sha256(path)
    if actual_sha != expected_sha256:
        raise EnvironmentArchiveError("archive publication receipt SHA-256 mismatch")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise EnvironmentArchiveError(f"invalid archive publication receipt: {error}") from error
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != 1
        or value.get("kind") != "cwam_cosmos_complete_environment_archive_publication"
        or value.get("status") != "published"
    ):
        raise EnvironmentArchiveError("archive publication receipt header is invalid")
    return value, actual_sha


def verify_and_extract(
    *,
    archive: Path,
    receipt: Path,
    expected_archive_sha256: str,
    expected_receipt_sha256: str,
    expected_ready_marker_sha256: str,
    stage: Path,
    verifier_source: Path,
) -> dict[str, Any]:
    for label, value in (
        ("archive", expected_archive_sha256),
        ("ready marker", expected_ready_marker_sha256),
    ):
        if len(value) != _SHA256_LENGTH:
            raise EnvironmentArchiveError(f"expected {label} SHA-256 is malformed")
    archive = _canonical_file(archive, "environment archive", mode=0o444)
    verifier_source = _canonical_file(verifier_source, "archive verifier source", mode=0o444)
    value, receipt_sha = _load_receipt(receipt, expected_receipt_sha256)
    archive_size, archive_sha = _sha256(archive)
    verifier_size, verifier_sha = _sha256(verifier_source)
    archive_value = value.get("archive")
    marker_value = value.get("ready_marker")
    publisher_value = value.get("publisher")
    source_value = value.get("source_environment")
    publication_value = value.get("publication")
    if not all(
        isinstance(item, dict)
        for item in (
            archive_value,
            marker_value,
            publisher_value,
            source_value,
            publication_value,
        )
    ):
        raise EnvironmentArchiveError("archive publication receipt closure is malformed")
    inventory_value = source_value.get("inventory")
    if not isinstance(inventory_value, dict):
        raise EnvironmentArchiveError("archive publication receipt inventory is malformed")
    if (
        archive_value.get("path") != str(archive)
        or archive_value.get("format") != "uncompressed-posix-tar-normalized-mtime"
        or archive_value.get("root_member") != "."
        or archive_value.get("mode") != "0444"
        or archive_value.get("size_bytes") != archive_size
        or archive_value.get("sha256") != archive_sha
        or archive_sha != expected_archive_sha256
        or source_value.get("deterministic_tar_sha256_before") != archive_sha
        or source_value.get("deterministic_tar_sha256_after") != archive_sha
        or source_value.get("stable_during_publication") is not True
        or marker_value.get("relative_path") != READY_MARKER
        or marker_value.get("mode") != "0444"
        or marker_value.get("sha256") != expected_ready_marker_sha256
        or publisher_value.get("sha256") != verifier_sha
        or publisher_value.get("size_bytes") != verifier_size
        or publication_value
        != {
            "archive_exclusive_hard_link": True,
            "receipt_exclusive_hard_link": True,
            "receipt_is_commit_point": True,
            "overwrite": False,
        }
    ):
        raise EnvironmentArchiveError("archive publication receipt binding mismatch")
    member_count = _validate_archive_members(archive)
    stage = Path(os.path.abspath(stage.expanduser()))
    if not stage.exists() or stage.is_symlink() or not stage.is_dir() or any(stage.iterdir()):
        raise EnvironmentArchiveError("extraction stage must be an existing empty real directory")
    subprocess.run(
        ["tar", "-xf", str(archive), "-C", str(stage), "--no-same-owner"],
        check=True,
    )
    marker = _canonical_file(stage / READY_MARKER, "extracted ready marker", mode=0o444)
    marker_size, marker_sha = _sha256(marker)
    if marker_size != marker_value.get("size_bytes") or marker_sha != expected_ready_marker_sha256:
        raise EnvironmentArchiveError("extracted ready marker differs from the publication receipt")
    inventory = _inventory(stage)
    if inventory != inventory_value:
        raise EnvironmentArchiveError("extracted environment inventory differs from the publication receipt")
    extracted_tar_sha = _tar_stream_sha256(stage)
    if extracted_tar_sha != archive_sha:
        raise EnvironmentArchiveError("extracted environment deterministic tar SHA-256 mismatch")
    result = {
        "status": "passed",
        "archive_sha256": archive_sha,
        "receipt_sha256": receipt_sha,
        "ready_marker_sha256": marker_sha,
        "member_count": member_count,
        "inventory": inventory,
        "stage": str(stage),
        "extracted_tar_sha256": extracted_tar_sha,
    }
    print(json.dumps(result, sort_keys=True))
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("publish-formal")
    verify = subparsers.add_parser("verify-extract")
    verify.add_argument("--archive", type=Path, required=True)
    verify.add_argument("--receipt", type=Path, required=True)
    verify.add_argument("--expected-archive-sha256", required=True)
    verify.add_argument("--expected-receipt-sha256", required=True)
    verify.add_argument("--expected-ready-marker-sha256", required=True)
    verify.add_argument("--stage", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "publish-formal":
            publish(
                source=FORMAL_SOURCE,
                archive=FORMAL_ARCHIVE,
                receipt=FORMAL_RECEIPT,
                materialization_receipt=FORMAL_MATERIALIZATION_RECEIPT,
                publisher_source=Path(__file__),
            )
        else:
            verify_and_extract(
                archive=args.archive,
                receipt=args.receipt,
                expected_archive_sha256=args.expected_archive_sha256,
                expected_receipt_sha256=args.expected_receipt_sha256,
                expected_ready_marker_sha256=args.expected_ready_marker_sha256,
                stage=args.stage,
                verifier_source=Path(__file__),
            )
    except (EnvironmentArchiveError, OSError, subprocess.CalledProcessError, tarfile.TarError) as error:
        print(f"ERROR: {error}", file=os.sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
