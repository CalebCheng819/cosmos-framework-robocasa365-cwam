#!/usr/bin/env python3
"""Safely expand the pinned RoboCasa repository and official asset ZIPs."""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import stat
import sys
import tarfile
import uuid
import zipfile
from pathlib import Path, PurePosixPath
from typing import BinaryIO

PINNED_PACKAGE_REGULAR_FILES = 1_253
PINNED_PACKAGE_DIRECTORIES = 202
PINNED_PACKAGE_REGULAR_FILE_BYTES = 11_099_314_178
PINNED_FINAL_PAYLOAD_FILES = 124_524
PINNED_FINAL_PAYLOAD_DIRECTORIES = 12_311
PINNED_FINAL_PAYLOAD_BYTES = 24_112_441_876

ZIP_TARGETS = {
    "aigen_objs.zip": PurePosixPath("robocasa/models/assets/objects"),
    "fixtures.zip": PurePosixPath("robocasa/models/assets"),
    "generative_textures.zip": PurePosixPath("robocasa/models/assets"),
    "lightwheel.zip": PurePosixPath("robocasa/models/assets/objects"),
    "objaverse.zip": PurePosixPath("robocasa/models/assets/objects"),
    "textures.zip": PurePosixPath("robocasa/models/assets"),
}
ZIP_ROOTS = {name: name.removesuffix(".zip") for name in ZIP_TARGETS}
PINNED_ASSET_README = (
    b"Assets from https://huggingface.co/datasets/nvidia/"
    b"PhysicalAI-Robotics-Manipulation-Objects-Kitchen-MJCF\n\n"
    b"Please visit https://huggingface.co/datasets/nvidia/"
    b"PhysicalAI-Robotics-Manipulation-Objects-Kitchen-MJCF for more information"
)
ZIP_ROOT_METADATA = {
    "fixtures.zip": {PurePosixPath("README.md"): PINNED_ASSET_README},
    "lightwheel.zip": {PurePosixPath("README.md"): PINNED_ASSET_README},
}
PROVENANCE_FILES = (
    PurePosixPath(".cwam_provenance/asset-tree-manifest.jsonl"),
    PurePosixPath(".cwam_provenance/asset-tree-receipt.json"),
    PurePosixPath(".cwam_provenance/headless-physics-smoke-receipt.json"),
)
EXPANDED_PREFIXES = tuple(
    target / ZIP_ROOTS[name] for name, target in ZIP_TARGETS.items()
)
COPY_CHUNK_BYTES = 8 * 1024 * 1024


class AssetPackageError(RuntimeError):
    """Raised when the package violates the fixed source-package contract."""


def _safe_relative_path(raw: str, label: str) -> PurePosixPath:
    if not raw or raw.startswith("/") or "\\" in raw:
        raise AssetPackageError(f"unsafe {label} path: {raw!r}")
    if any(ord(character) < 32 or ord(character) == 127 for character in raw):
        raise AssetPackageError(f"control character in {label} path: {raw!r}")
    trimmed = raw[:-1] if raw.endswith("/") else raw
    raw_parts = trimmed.split("/")
    if not raw_parts or any(part in {"", ".", ".."} for part in raw_parts):
        raise AssetPackageError(f"non-canonical {label} path: {raw!r}")
    return PurePosixPath(*raw_parts)


def _require_empty_stage(path: Path) -> Path:
    lexical = path.absolute()
    if path.is_symlink() or not path.is_dir():
        raise AssetPackageError(f"stage must be an existing non-symlink directory: {path}")
    resolved = path.resolve(strict=True)
    if lexical != resolved:
        raise AssetPackageError(f"stage must be canonical and contain no symlink components: {path}")
    if any(path.iterdir()):
        raise AssetPackageError(f"stage must be empty: {path}")
    return resolved


def _ensure_directory(root: Path, relative: PurePosixPath) -> Path:
    current = root
    for part in relative.parts:
        current = current / part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            current.mkdir(mode=0o755)
            continue
        if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
            raise AssetPackageError(f"directory path collides with a non-directory: {current}")
    return current


def _copy_exact(source: BinaryIO, destination: Path, expected_bytes: int, executable: bool) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(destination, flags, 0o600)
    copied = 0
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            while True:
                chunk = source.read(COPY_CHUNK_BYTES)
                if not chunk:
                    break
                stream.write(chunk)
                copied += len(chunk)
            stream.flush()
        if copied != expected_bytes:
            raise AssetPackageError(
                f"file length mismatch for {destination}: expected={expected_bytes} actual={copied}"
            )
        os.chmod(destination, 0o555 if executable else 0o444, follow_symlinks=False)
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    finally:
        os.close(descriptor)


def _tar_destination(stage: Path, archive_path: PurePosixPath) -> Path:
    if archive_path == PurePosixPath("repo"):
        return stage
    if archive_path.parts[0] == "repo":
        return stage.joinpath(*archive_path.parts[1:])
    if (
        len(archive_path.parts) == 2
        and archive_path.parts[0] == "zips"
        and archive_path.parts[1] in ZIP_TARGETS
    ):
        return stage / ".source-zips" / archive_path.parts[1]
    raise AssetPackageError(f"tar member is outside repo/ and fixed zips/: {archive_path}")


def _reject_preexpanded_tar_member(archive_path: PurePosixPath) -> None:
    if archive_path.parts[0] != "repo":
        return
    relative = PurePosixPath(*archive_path.parts[1:])
    for prefix in EXPANDED_PREFIXES:
        if relative == prefix or prefix in relative.parents:
            raise AssetPackageError(f"tar skeleton contains expanded asset path: {archive_path}")


def _extract_tar(
    archive: Path,
    stage: Path,
    expected_archive_bytes: int,
    expected_files: int,
    expected_directories: int,
    expected_regular_file_bytes: int,
) -> dict[str, int]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(archive, flags)
    seen: set[PurePosixPath] = set()
    files = directories = regular_file_bytes = 0
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise AssetPackageError(f"package is not a regular file: {archive}")
        if metadata.st_size != expected_archive_bytes:
            raise AssetPackageError(
                f"package byte size mismatch: expected={expected_archive_bytes} actual={metadata.st_size}"
            )
        with os.fdopen(os.dup(descriptor), "rb") as stream:
            with tarfile.open(fileobj=stream, mode="r|") as package:
                for member in package:
                    member_path = _safe_relative_path(member.name, "tar member")
                    if member_path in seen:
                        raise AssetPackageError(f"duplicate tar member: {member_path}")
                    seen.add(member_path)
                    _reject_preexpanded_tar_member(member_path)
                    destination = _tar_destination(stage, member_path)
                    if member.isdir():
                        if member_path.parts[0] == "zips":
                            raise AssetPackageError(f"unexpected ZIP directory member: {member_path}")
                        _ensure_directory(stage, PurePosixPath(destination.relative_to(stage).as_posix()))
                        directories += 1
                        continue
                    if not member.isreg():
                        raise AssetPackageError(
                            f"unsupported tar member type for {member_path}: {member.type!r}"
                        )
                    _ensure_directory(
                        stage,
                        PurePosixPath(destination.parent.relative_to(stage).as_posix()),
                    )
                    source = package.extractfile(member)
                    if source is None:
                        raise AssetPackageError(f"cannot read tar member: {member_path}")
                    with source:
                        _copy_exact(
                            source,
                            destination,
                            member.size,
                            executable=bool(member.mode & 0o111),
                        )
                    files += 1
                    regular_file_bytes += member.size
    finally:
        os.close(descriptor)

    observed = {
        "regular_files": files,
        "directories": directories,
        "regular_file_bytes": regular_file_bytes,
    }
    expected = {
        "regular_files": expected_files,
        "directories": expected_directories,
        "regular_file_bytes": expected_regular_file_bytes,
    }
    if observed != expected:
        raise AssetPackageError(f"source package inventory mismatch: expected={expected} actual={observed}")
    for relative in PROVENANCE_FILES:
        path = stage / relative
        if not path.is_file() or path.is_symlink():
            raise AssetPackageError(f"required publisher provenance file is missing: {relative}")
    zip_root = stage / ".source-zips"
    observed_zips = {path.name for path in zip_root.iterdir()} if zip_root.is_dir() else set()
    if observed_zips != set(ZIP_TARGETS):
        raise AssetPackageError(
            f"source package ZIP set mismatch: expected={sorted(ZIP_TARGETS)} actual={sorted(observed_zips)}"
        )
    return observed


def _zip_entry_kind(info: zipfile.ZipInfo) -> str:
    unix_mode = (info.external_attr >> 16) & 0xFFFF
    file_type = stat.S_IFMT(unix_mode)
    if info.is_dir():
        if file_type not in {0, stat.S_IFDIR}:
            raise AssetPackageError(f"ZIP directory has incompatible mode: {info.filename}")
        return "directory"
    if file_type not in {0, stat.S_IFREG}:
        raise AssetPackageError(f"ZIP contains link or special entry: {info.filename}")
    return "file"


def _require_pinned_zip_metadata(
    package: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    archive_name: str,
    member: PurePosixPath,
) -> bytes:
    expected = ZIP_ROOT_METADATA.get(archive_name, {}).get(member)
    if expected is None:
        raise AssetPackageError(f"ZIP member has unexpected root: {archive_name}:{member}")
    if _zip_entry_kind(info) != "file" or info.file_size != len(expected):
        raise AssetPackageError(f"pinned ZIP metadata mismatch: {archive_name}:{member}")
    with package.open(info, "r") as source:
        observed = source.read(len(expected) + 1)
    if observed != expected:
        raise AssetPackageError(f"pinned ZIP metadata mismatch: {archive_name}:{member}")
    return expected


def _require_identical_existing_file(path: Path, expected: bytes) -> None:
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise AssetPackageError(f"pinned ZIP metadata collides with a non-file: {path}")
    if metadata.st_size != len(expected) or path.read_bytes() != expected:
        raise AssetPackageError(f"pinned ZIP metadata collides with different content: {path}")


def _extract_zip(stage: Path, archive_name: str, occupied: set[PurePosixPath]) -> None:
    archive = stage / ".source-zips" / archive_name
    target_base = ZIP_TARGETS[archive_name]
    expected_root = ZIP_ROOTS[archive_name]
    seen: set[PurePosixPath] = set()
    with zipfile.ZipFile(archive) as package:
        for info in package.infolist():
            if info.flag_bits & 0x1:
                raise AssetPackageError(f"encrypted ZIP entry is forbidden: {archive_name}:{info.filename}")
            member = _safe_relative_path(info.filename, f"{archive_name} member")
            if member in seen:
                raise AssetPackageError(f"duplicate ZIP member: {archive_name}:{member}")
            seen.add(member)
            final_relative = target_base / member
            destination = stage.joinpath(*final_relative.parts)
            if member.parts[0] != expected_root:
                expected_metadata = _require_pinned_zip_metadata(
                    package,
                    info,
                    archive_name,
                    member,
                )
                if final_relative in occupied:
                    _require_identical_existing_file(destination, expected_metadata)
                    continue
                _ensure_directory(stage, PurePosixPath(*final_relative.parts[:-1]))
                _copy_exact(
                    io.BytesIO(expected_metadata),
                    destination,
                    len(expected_metadata),
                    executable=False,
                )
                occupied.add(final_relative)
                continue
            if final_relative in occupied:
                raise AssetPackageError(f"ZIP member collides with repository skeleton: {final_relative}")
            occupied.add(final_relative)
            kind = _zip_entry_kind(info)
            if kind == "directory":
                _ensure_directory(stage, final_relative)
                continue
            _ensure_directory(stage, PurePosixPath(*final_relative.parts[:-1]))
            with package.open(info, "r") as source:
                _copy_exact(
                    source,
                    destination,
                    info.file_size,
                    executable=bool(((info.external_attr >> 16) & 0o777) & 0o111),
                )


def _skeleton_paths(stage: Path) -> set[PurePosixPath]:
    occupied: set[PurePosixPath] = set()
    for root, directories, files in os.walk(stage, topdown=True, followlinks=False):
        root_path = Path(root)
        relative_root = root_path.relative_to(stage)
        directories[:] = sorted(name for name in directories if name != ".source-zips")
        for name in directories:
            path = root_path / name
            if path.is_symlink():
                raise AssetPackageError(f"repository skeleton contains a symlink: {path}")
            occupied.add(PurePosixPath((relative_root / name).as_posix()))
        for name in sorted(files):
            path = root_path / name
            if path.is_symlink() or not path.is_file():
                raise AssetPackageError(f"repository skeleton contains an unsupported entry: {path}")
            occupied.add(PurePosixPath((relative_root / name).as_posix()))
    return occupied


def _scan_final_payload(stage: Path) -> dict[str, int]:
    files = directories = symlinks = total_bytes = 0

    def scan(directory: Path, top_level: bool = False) -> None:
        nonlocal files, directories, symlinks, total_bytes
        with os.scandir(directory) as entries:
            for entry in sorted(entries, key=lambda item: item.name):
                if top_level and entry.name in {".git", ".cwam_provenance"}:
                    continue
                metadata = entry.stat(follow_symlinks=False)
                if stat.S_ISLNK(metadata.st_mode):
                    symlinks += 1
                elif stat.S_ISDIR(metadata.st_mode):
                    directories += 1
                    scan(Path(entry.path))
                elif stat.S_ISREG(metadata.st_mode):
                    files += 1
                    total_bytes += metadata.st_size
                else:
                    raise AssetPackageError(f"final payload contains a special file: {entry.path}")

    scan(stage, top_level=True)
    return {
        "files": files,
        "directories": directories,
        "symlinks": symlinks,
        "bytes": total_bytes,
    }


def _write_receipt(path: Path, value: dict[str, object]) -> None:
    if path.exists() or path.is_symlink():
        raise AssetPackageError(f"refusing to replace extraction receipt: {path}")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
    temporary = path.parent / f".{path.name}.incomplete-{uuid.uuid4().hex}"
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o444, follow_symlinks=False)
        os.replace(temporary, path)
    finally:
        os.close(descriptor)
        temporary.unlink(missing_ok=True)


def verify_extract(args: argparse.Namespace) -> None:
    archive = args.archive.absolute()
    if archive.is_symlink() or not archive.is_file() or archive.resolve(strict=True) != archive:
        raise AssetPackageError(f"archive must be a canonical regular non-symlink file: {archive}")
    stage = _require_empty_stage(args.stage)
    package_inventory = _extract_tar(
        archive,
        stage,
        args.expected_archive_bytes,
        args.expected_package_files,
        args.expected_package_directories,
        args.expected_package_bytes,
    )
    occupied = _skeleton_paths(stage)
    for archive_name in sorted(ZIP_TARGETS):
        _extract_zip(stage, archive_name, occupied)
    shutil.rmtree(stage / ".source-zips")
    final_inventory = _scan_final_payload(stage)
    expected_final = {
        "files": args.expected_final_files,
        "directories": args.expected_final_directories,
        "symlinks": 0,
        "bytes": args.expected_final_bytes,
    }
    if final_inventory != expected_final:
        raise AssetPackageError(
            f"expanded RoboCasa payload inventory mismatch: expected={expected_final} actual={final_inventory}"
        )
    receipt = {
        "schema_version": 1,
        "kind": "cwam_robocasa_asset_source_package_extraction",
        "status": "passed",
        "archive_path": archive.as_posix(),
        "archive_bytes": args.expected_archive_bytes,
        "package_inventory": package_inventory,
        "final_payload_inventory": final_inventory,
    }
    _write_receipt(args.receipt_output.absolute(), receipt)
    print(
        "CWAM_ROBOCASA_ASSET_SOURCE_PACKAGE_READY "
        f"archive_bytes={args.expected_archive_bytes} "
        f"package_files={package_inventory['regular_files']} "
        f"final_files={final_inventory['files']} final_bytes={final_inventory['bytes']}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    extract = subparsers.add_parser("verify-extract")
    extract.add_argument("--archive", type=Path, required=True)
    extract.add_argument("--expected-archive-bytes", type=int, required=True)
    extract.add_argument("--stage", type=Path, required=True)
    extract.add_argument("--receipt-output", type=Path, required=True)
    extract.add_argument(
        "--expected-package-files", type=int, default=PINNED_PACKAGE_REGULAR_FILES
    )
    extract.add_argument(
        "--expected-package-directories", type=int, default=PINNED_PACKAGE_DIRECTORIES
    )
    extract.add_argument(
        "--expected-package-bytes", type=int, default=PINNED_PACKAGE_REGULAR_FILE_BYTES
    )
    extract.add_argument("--expected-final-files", type=int, default=PINNED_FINAL_PAYLOAD_FILES)
    extract.add_argument(
        "--expected-final-directories", type=int, default=PINNED_FINAL_PAYLOAD_DIRECTORIES
    )
    extract.add_argument("--expected-final-bytes", type=int, default=PINNED_FINAL_PAYLOAD_BYTES)
    extract.set_defaults(handler=verify_extract)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        args.handler(args)
    except (AssetPackageError, OSError, tarfile.TarError, zipfile.BadZipFile) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
