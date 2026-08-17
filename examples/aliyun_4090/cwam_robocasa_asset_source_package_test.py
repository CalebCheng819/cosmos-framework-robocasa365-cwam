from __future__ import annotations

import argparse
import io
import json
import tarfile
import zipfile
from pathlib import Path

import cwam_robocasa_asset_source_package as asset_package
import pytest

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


SKELETON_DIRECTORIES = (
    "repo/",
    "repo/.cwam_provenance/",
    "repo/.git/",
    "repo/robocasa/",
    "repo/robocasa/models/",
    "repo/robocasa/models/assets/",
    "repo/robocasa/models/assets/objects/",
)
PROVENANCE_PAYLOADS = {
    "repo/.cwam_provenance/asset-tree-manifest.jsonl": b'{"path":"robocasa"}\n',
    "repo/.cwam_provenance/asset-tree-receipt.json": b'{"status":"passed"}\n',
    "repo/.cwam_provenance/headless-physics-smoke-receipt.json": b'{"status":"passed"}\n',
}


def _zip_payload(
    archive_name: str,
    *,
    traversal: bool = False,
    root_metadata_override: bytes | None = None,
    unexpected_root_member: bool = False,
) -> bytes:
    root = asset_package.ZIP_ROOTS[archive_name]
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
        if archive_name in asset_package.ZIP_ROOT_METADATA:
            payload = (
                root_metadata_override
                if root_metadata_override is not None
                else asset_package.PINNED_ASSET_README
            )
            archive.writestr("README.md", payload)
        if unexpected_root_member:
            archive.writestr("unexpected.txt", b"not admitted\n")
        archive.writestr(f"{root}/", b"")
        member = f"{root}/../escape.bin" if traversal else f"{root}/item.bin"
        archive.writestr(member, f"payload:{archive_name}\n".encode())
    return stream.getvalue()


def _build_package(
    path: Path,
    *,
    traversal_archive: str | None = None,
    metadata_override_archive: str | None = None,
    unexpected_root_archive: str | None = None,
) -> dict[str, int]:
    files = {
        **PROVENANCE_PAYLOADS,
        "repo/.git/HEAD": b"ref: refs/heads/main\n",
        "repo/robocasa/__init__.py": b'VERSION = "test"\n',
        "repo/robocasa/models/assets/objects/README.md": asset_package.PINNED_ASSET_README,
    }
    expanded_payload_bytes = 0
    for archive_name in sorted(asset_package.ZIP_TARGETS):
        files[f"zips/{archive_name}"] = _zip_payload(
            archive_name,
            traversal=archive_name == traversal_archive,
            root_metadata_override=(
                b"tampered metadata\n" if archive_name == metadata_override_archive else None
            ),
            unexpected_root_member=archive_name == unexpected_root_archive,
        )
        expanded_payload_bytes += len(f"payload:{archive_name}\n".encode())

    with tarfile.open(path, "w") as archive:
        for name in SKELETON_DIRECTORIES:
            info = tarfile.TarInfo(name)
            info.type = tarfile.DIRTYPE
            info.mode = 0o755
            archive.addfile(info)
        for name, payload in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            info.mode = 0o444
            archive.addfile(info, io.BytesIO(payload))

    return {
        "archive_bytes": path.stat().st_size,
        "package_files": len(files),
        "package_directories": len(SKELETON_DIRECTORIES),
        "package_bytes": sum(len(payload) for payload in files.values()),
        "final_files": 3 + len(asset_package.ZIP_TARGETS),
        "final_directories": 4 + len(asset_package.ZIP_TARGETS),
        "final_bytes": (
            len(files["repo/robocasa/__init__.py"])
            + expanded_payload_bytes
            + 2 * len(asset_package.PINNED_ASSET_README)
        ),
    }


def _arguments(
    archive: Path,
    stage: Path,
    receipt: Path,
    inventory: dict[str, int],
) -> argparse.Namespace:
    return argparse.Namespace(
        archive=archive,
        stage=stage,
        receipt_output=receipt,
        expected_archive_bytes=inventory["archive_bytes"],
        expected_package_files=inventory["package_files"],
        expected_package_directories=inventory["package_directories"],
        expected_package_bytes=inventory["package_bytes"],
        expected_final_files=inventory["final_files"],
        expected_final_directories=inventory["final_directories"],
        expected_final_bytes=inventory["final_bytes"],
    )


def test_verify_extract_preserves_provenance_and_expands_fixed_zips(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "robocasa-assets.tar"
    inventory = _build_package(archive)
    stage = tmp_path / "stage"
    stage.mkdir()
    receipt = tmp_path / "receipt.json"

    asset_package.verify_extract(_arguments(archive, stage, receipt, inventory))

    value = json.loads(receipt.read_text(encoding="utf-8"))
    assert value["status"] == "passed"
    assert value["final_payload_inventory"]["files"] == inventory["final_files"]
    assert (stage / ".cwam_provenance/asset-tree-receipt.json").is_file()
    assert (stage / ".git/HEAD").is_file()
    assert not (stage / ".source-zips").exists()
    for archive_name, target in asset_package.ZIP_TARGETS.items():
        root = asset_package.ZIP_ROOTS[archive_name]
        assert stage.joinpath(*target.parts, root, "item.bin").is_file()
    assert (stage / "robocasa/models/assets/README.md").read_bytes() == (
        asset_package.PINNED_ASSET_README
    )
    assert (stage / "robocasa/models/assets/objects/README.md").read_bytes() == (
        asset_package.PINNED_ASSET_README
    )


def test_verify_extract_rejects_zip_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "robocasa-assets.tar"
    inventory = _build_package(archive, traversal_archive="fixtures.zip")
    stage = tmp_path / "stage"
    stage.mkdir()
    receipt = tmp_path / "receipt.json"

    with pytest.raises(asset_package.AssetPackageError, match="non-canonical"):
        asset_package.verify_extract(_arguments(archive, stage, receipt, inventory))

    assert not (tmp_path / "escape.bin").exists()
    assert not receipt.exists()


def test_safe_relative_path_rejects_tar_aliases() -> None:
    for unsafe in ("../repo", "repo/../escape", "repo//module.py", "/repo/module.py"):
        with pytest.raises(asset_package.AssetPackageError):
            asset_package._safe_relative_path(unsafe, "test")


def test_verify_extract_rejects_changed_pinned_zip_metadata(tmp_path: Path) -> None:
    archive = tmp_path / "robocasa-assets.tar"
    inventory = _build_package(archive, metadata_override_archive="fixtures.zip")
    stage = tmp_path / "stage"
    stage.mkdir()

    with pytest.raises(asset_package.AssetPackageError, match="pinned ZIP metadata mismatch"):
        asset_package.verify_extract(
            _arguments(archive, stage, tmp_path / "receipt.json", inventory)
        )


def test_verify_extract_rejects_unknown_zip_root_member(tmp_path: Path) -> None:
    archive = tmp_path / "robocasa-assets.tar"
    inventory = _build_package(archive, unexpected_root_archive="textures.zip")
    stage = tmp_path / "stage"
    stage.mkdir()

    with pytest.raises(asset_package.AssetPackageError, match="unexpected root"):
        asset_package.verify_extract(
            _arguments(archive, stage, tmp_path / "receipt.json", inventory)
        )
