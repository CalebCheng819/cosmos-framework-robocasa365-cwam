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
STATIC_PROVENANCE_PAYLOADS = {
    "repo/.cwam_provenance/asset-tree-receipt.json": b'{"status":"passed"}\n',
    "repo/.cwam_provenance/headless-physics-smoke-receipt.json": b'{"status":"passed"}\n',
}
OVERLAY_DIRECTORY = "robocasa/models/assets/fixtures/legacy"
OVERLAY_FILE = f"{OVERLAY_DIRECTORY}/legacy.xml"
OVERLAY_PAYLOAD = b"<mujoco/>\n"


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
    directory_records = {
        "robocasa",
        "robocasa/models",
        "robocasa/models/assets",
        "robocasa/models/assets/objects",
        OVERLAY_DIRECTORY,
    }
    file_records = {
        "robocasa/__init__.py": b'VERSION = "test"\n',
        "robocasa/models/assets/objects/README.md": asset_package.PINNED_ASSET_README,
        OVERLAY_FILE: OVERLAY_PAYLOAD,
    }
    for archive_name, target in asset_package.ZIP_TARGETS.items():
        root = asset_package.ZIP_ROOTS[archive_name]
        directory_records.add((target / root).as_posix())
        file_records[(target / root / "item.bin").as_posix()] = (
            f"payload:{archive_name}\n".encode()
        )
    manifest_lines = [
        json.dumps(
            {
                "excluded_top_level_paths": [".cwam_provenance", ".git"],
                "kind": "cwam_robocasa_asset_tree_manifest",
                "payload_root": "/test/robocasa",
                "schema_version": 1,
            },
            separators=(",", ":"),
            sort_keys=True,
        )
    ]
    manifest_lines.extend(
        json.dumps(
            {"mode": 0o755, "path": path, "type": "directory"},
            separators=(",", ":"),
            sort_keys=True,
        )
        for path in sorted(directory_records)
    )
    manifest_lines.extend(
        json.dumps(
            {
                "mode": 0o644,
                "path": path,
                "sha256": "publisher-value-not-read-by-metadata-verifier",
                "size_bytes": len(payload),
                "type": "file",
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        for path, payload in sorted(file_records.items())
    )
    manifest_payload = ("\n".join(manifest_lines) + "\n").encode()
    files = {
        **STATIC_PROVENANCE_PAYLOADS,
        "repo/.cwam_provenance/asset-tree-manifest.jsonl": manifest_payload,
        "repo/.git/HEAD": b"ref: refs/heads/main\n",
        "repo/robocasa/__init__.py": file_records["robocasa/__init__.py"],
        "repo/robocasa/models/assets/objects/README.md": file_records[
            "robocasa/models/assets/objects/README.md"
        ],
    }
    for archive_name in sorted(asset_package.ZIP_TARGETS):
        files[f"zips/{archive_name}"] = _zip_payload(
            archive_name,
            traversal=archive_name == traversal_archive,
            root_metadata_override=(
                b"tampered metadata\n" if archive_name == metadata_override_archive else None
            ),
            unexpected_root_member=archive_name == unexpected_root_archive,
        )

    with tarfile.open(path, "w") as archive:
        for name in SKELETON_DIRECTORIES:
            info = tarfile.TarInfo(name)
            info.type = tarfile.DIRTYPE
            info.mode = 0o755
            archive.addfile(info)
        for name, payload in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            info.mode = 0o444 if "/.cwam_provenance/" in name else 0o644
            archive.addfile(info, io.BytesIO(payload))

    return {
        "archive_bytes": path.stat().st_size,
        "package_files": len(files),
        "package_directories": len(SKELETON_DIRECTORIES),
        "package_bytes": sum(len(payload) for payload in files.values()),
        "final_files": len(file_records),
        "final_directories": len(directory_records),
        "final_bytes": sum(len(payload) for payload in file_records.values()),
    }


def _build_overlay(path: Path, *, extra_member: bool = False) -> dict[str, int]:
    with tarfile.open(path, "w") as archive:
        directory = tarfile.TarInfo(f"{OVERLAY_DIRECTORY}/")
        directory.type = tarfile.DIRTYPE
        directory.mode = 0o755
        archive.addfile(directory)
        file_info = tarfile.TarInfo(OVERLAY_FILE)
        file_info.size = len(OVERLAY_PAYLOAD)
        file_info.mode = 0o644
        archive.addfile(file_info, io.BytesIO(OVERLAY_PAYLOAD))
        if extra_member:
            extra = tarfile.TarInfo(f"{OVERLAY_DIRECTORY}/extra.xml")
            extra.size = 1
            extra.mode = 0o644
            archive.addfile(extra, io.BytesIO(b"x"))
    return {
        "overlay_archive_bytes": path.stat().st_size,
        "overlay_files": 1,
        "overlay_directories": 1,
        "overlay_payload_bytes": len(OVERLAY_PAYLOAD),
    }


def _arguments(
    archive: Path,
    overlay: Path,
    stage: Path,
    receipt: Path,
    inventory: dict[str, int],
) -> argparse.Namespace:
    return argparse.Namespace(
        archive=archive,
        overlay=overlay,
        stage=stage,
        receipt_output=receipt,
        expected_archive_bytes=inventory["archive_bytes"],
        expected_package_files=inventory["package_files"],
        expected_package_directories=inventory["package_directories"],
        expected_package_bytes=inventory["package_bytes"],
        expected_overlay_bytes=inventory["overlay_archive_bytes"],
        expected_overlay_files=inventory["overlay_files"],
        expected_overlay_directories=inventory["overlay_directories"],
        expected_overlay_payload_bytes=inventory["overlay_payload_bytes"],
        expected_final_files=inventory["final_files"],
        expected_final_directories=inventory["final_directories"],
        expected_final_bytes=inventory["final_bytes"],
    )


def test_verify_extract_preserves_provenance_and_expands_fixed_zips(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "robocasa-assets.tar"
    inventory = _build_package(archive)
    overlay = tmp_path / "legacy-fixture-overlay.tar"
    inventory.update(_build_overlay(overlay))
    stage = tmp_path / "stage"
    stage.mkdir()
    receipt = tmp_path / "receipt.json"

    asset_package.verify_extract(_arguments(archive, overlay, stage, receipt, inventory))

    value = json.loads(receipt.read_text(encoding="utf-8"))
    assert value["status"] == "passed"
    assert value["final_payload_inventory"]["files"] == inventory["final_files"]
    assert (stage / ".cwam_provenance/asset-tree-receipt.json").is_file()
    assert (stage / ".git/HEAD").is_file()
    assert not (stage / ".source-zips").exists()
    for archive_name, target in asset_package.ZIP_TARGETS.items():
        root = asset_package.ZIP_ROOTS[archive_name]
        assert stage.joinpath(*target.parts, root, "item.bin").is_file()
    assert not (stage / "robocasa/models/assets/README.md").exists()
    assert (stage / "robocasa/models/assets/objects/README.md").read_bytes() == (
        asset_package.PINNED_ASSET_README
    )
    assert (stage / OVERLAY_FILE).read_bytes() == OVERLAY_PAYLOAD
    assert (stage / OVERLAY_FILE).stat().st_mode & 0o777 == 0o644
    assert (stage / OVERLAY_DIRECTORY).stat().st_mode & 0o777 == 0o755
    assert value["schema_version"] == "metadata_no_hash_v1"
    assert value["final_validation"] == "publisher_manifest_path_type_mode_size_exact"


def test_verify_live_metadata_rechecks_the_publisher_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = tmp_path / "robocasa-assets.tar"
    inventory = _build_package(archive)
    overlay = tmp_path / "legacy-fixture-overlay.tar"
    inventory.update(_build_overlay(overlay))
    stage = tmp_path / "stage"
    stage.mkdir()
    asset_package.verify_extract(
        _arguments(archive, overlay, stage, tmp_path / "extraction.json", inventory)
    )
    monkeypatch.setattr(
        asset_package,
        "_git_commit",
        lambda root: asset_package.PINNED_ROBOCASA_COMMIT,
    )
    output = tmp_path / "live.json"

    asset_package.verify_live_metadata(
        argparse.Namespace(
            asset_root=stage,
            output=output,
            expected_robocasa_commit=asset_package.PINNED_ROBOCASA_COMMIT,
            expected_final_files=inventory["final_files"],
            expected_final_directories=inventory["final_directories"],
            expected_final_bytes=inventory["final_bytes"],
        )
    )

    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["schema_version"] == "metadata_no_hash_v1"
    assert receipt["robocasa_git_commit"] == asset_package.PINNED_ROBOCASA_COMMIT
    assert receipt["validation"] == "publisher_manifest_path_type_mode_size_exact"


def test_verify_extract_rejects_zip_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "robocasa-assets.tar"
    inventory = _build_package(archive, traversal_archive="fixtures.zip")
    overlay = tmp_path / "legacy-fixture-overlay.tar"
    inventory.update(_build_overlay(overlay))
    stage = tmp_path / "stage"
    stage.mkdir()
    receipt = tmp_path / "receipt.json"

    with pytest.raises(asset_package.AssetPackageError, match="non-canonical"):
        asset_package.verify_extract(_arguments(archive, overlay, stage, receipt, inventory))

    assert not (tmp_path / "escape.bin").exists()
    assert not receipt.exists()


def test_safe_relative_path_rejects_tar_aliases() -> None:
    for unsafe in ("../repo", "repo/../escape", "repo//module.py", "/repo/module.py"):
        with pytest.raises(asset_package.AssetPackageError):
            asset_package._safe_relative_path(unsafe, "test")


def test_verify_extract_rejects_changed_pinned_zip_metadata(tmp_path: Path) -> None:
    archive = tmp_path / "robocasa-assets.tar"
    inventory = _build_package(archive, metadata_override_archive="fixtures.zip")
    overlay = tmp_path / "legacy-fixture-overlay.tar"
    inventory.update(_build_overlay(overlay))
    stage = tmp_path / "stage"
    stage.mkdir()

    with pytest.raises(asset_package.AssetPackageError, match="pinned ZIP metadata mismatch"):
        asset_package.verify_extract(
            _arguments(archive, overlay, stage, tmp_path / "receipt.json", inventory)
        )


def test_verify_extract_rejects_unknown_zip_root_member(tmp_path: Path) -> None:
    archive = tmp_path / "robocasa-assets.tar"
    inventory = _build_package(archive, unexpected_root_archive="textures.zip")
    overlay = tmp_path / "legacy-fixture-overlay.tar"
    inventory.update(_build_overlay(overlay))
    stage = tmp_path / "stage"
    stage.mkdir()

    with pytest.raises(asset_package.AssetPackageError, match="unexpected root"):
        asset_package.verify_extract(
            _arguments(archive, overlay, stage, tmp_path / "receipt.json", inventory)
        )


def test_verify_extract_rejects_overlay_path_outside_exact_missing_set(tmp_path: Path) -> None:
    archive = tmp_path / "robocasa-assets.tar"
    inventory = _build_package(archive)
    overlay = tmp_path / "legacy-fixture-overlay.tar"
    inventory.update(_build_overlay(overlay, extra_member=True))
    stage = tmp_path / "stage"
    stage.mkdir()

    with pytest.raises(asset_package.AssetPackageError, match="not an expected missing path"):
        asset_package.verify_extract(
            _arguments(archive, overlay, stage, tmp_path / "receipt.json", inventory)
        )
