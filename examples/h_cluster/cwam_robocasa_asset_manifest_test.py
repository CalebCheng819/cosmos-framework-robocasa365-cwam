# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""CPU-only tests for the strict RoboCasa asset-tree verifier."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path, PurePosixPath

import pytest

from examples.h_cluster import cwam_robocasa_asset_manifest as asset_manifest


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _payload_records(root: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []

    def scan(directory: Path, relative_base: PurePosixPath) -> None:
        for child in sorted(directory.iterdir(), key=lambda path: path.name):
            if not relative_base.parts and child.name in asset_manifest.EXCLUDED_TOP_LEVEL_PATHS:
                continue
            relative = relative_base / child.name
            metadata = child.lstat()
            if stat.S_ISDIR(metadata.st_mode):
                records.append(
                    {
                        "mode": stat.S_IMODE(metadata.st_mode),
                        "path": relative.as_posix(),
                        "type": "directory",
                    }
                )
                scan(child, relative)
            else:
                records.append(
                    {
                        "mode": stat.S_IMODE(metadata.st_mode),
                        "path": relative.as_posix(),
                        "sha256": _sha256(child),
                        "size_bytes": metadata.st_size,
                        "type": "file",
                    }
                )

    scan(root, PurePosixPath())
    return records


def _write_readonly_json(path: Path, value: object) -> None:
    if path.exists():
        path.chmod(0o644)
    path.write_text(f"{json.dumps(value, indent=2, sort_keys=True)}\n", encoding="utf-8")
    path.chmod(asset_manifest.PROVENANCE_FILE_MODE)


def _write_readonly_manifest(path: Path, values: list[object]) -> None:
    if path.exists():
        path.chmod(0o644)
    payload = "".join(
        f"{json.dumps(value, ensure_ascii=False, separators=(',', ':'), sort_keys=True)}\n" for value in values
    )
    path.write_text(payload, encoding="utf-8")
    path.chmod(asset_manifest.PROVENANCE_FILE_MODE)


def _manifest_path(root: Path) -> Path:
    return root / asset_manifest.MANIFEST_RELATIVE_PATH


def _receipt_path(root: Path) -> Path:
    return root / asset_manifest.PUBLISHER_RECEIPT_RELATIVE_PATH


def _read_manifest(root: Path) -> list[dict]:
    return [json.loads(line) for line in _manifest_path(root).read_text(encoding="utf-8").splitlines()]


def _read_receipt(root: Path) -> dict:
    return json.loads(_receipt_path(root).read_text(encoding="utf-8"))


def _rewrite_receipt(root: Path, receipt: dict) -> None:
    _write_readonly_json(_receipt_path(root), receipt)


def _rewrite_manifest(root: Path, values: list[object], *, rebind_receipt: bool = True) -> None:
    _write_readonly_manifest(_manifest_path(root), values)
    if rebind_receipt:
        receipt = _read_receipt(root)
        receipt["manifest_sha256"] = _sha256(_manifest_path(root))
        _rewrite_receipt(root, receipt)


@pytest.fixture
def published_tree(tmp_path: Path) -> Path:
    root = tmp_path / "robocasa-assets"
    root.mkdir(mode=0o755)
    root.chmod(0o755)
    git_directory = root / ".git"
    git_directory.mkdir(mode=0o755)
    (git_directory / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")

    provenance = root / asset_manifest.PROVENANCE_DIRECTORY
    provenance.mkdir(mode=0o755)
    assets = root / "assets"
    assets.mkdir(mode=0o755)
    nested = assets / "nested"
    nested.mkdir(mode=0o755)
    (root / "README.md").write_bytes(b"fixture readme\n")
    (assets / "texture.png").write_bytes(b"\x89PNG\r\nfixture")
    (nested / "model.bin").write_bytes(b"\x00\x01\x02\x03")
    for path in (root / "README.md", assets / "texture.png", nested / "model.bin"):
        path.chmod(0o644)

    header = {
        "excluded_top_level_paths": list(asset_manifest.EXCLUDED_TOP_LEVEL_PATHS),
        "kind": asset_manifest.MANIFEST_KIND,
        "payload_root": str(root),
        "schema_version": asset_manifest.SCHEMA_VERSION,
    }
    records = _payload_records(root)
    manifest_path = _manifest_path(root)
    _write_readonly_manifest(manifest_path, [header, *records])
    summary = {
        "directories": sum(record["type"] == "directory" for record in records),
        "files": sum(record["type"] == "file" for record in records),
        "total_bytes": sum(int(record["size_bytes"]) for record in records if record["type"] == "file"),
    }
    source_root = tmp_path / "source-archive"
    receipt = {
        "excluded_top_level_paths": list(asset_manifest.EXCLUDED_TOP_LEVEL_PATHS),
        "generated_at_utc": "2026-07-28T18:09:13.021394Z",
        "generator_id": asset_manifest.GENERATOR_ID,
        "git_commit": "a" * 40,
        "git_status_porcelain": "",
        "kind": asset_manifest.PUBLISHER_RECEIPT_KIND,
        "manifest_relative_path": asset_manifest.MANIFEST_RELATIVE_PATH,
        "manifest_sha256": _sha256(manifest_path),
        "payload_directory_count": summary["directories"],
        "payload_file_count": summary["files"],
        "payload_root": str(root),
        "payload_symlink_count": 0,
        "payload_total_bytes": summary["total_bytes"],
        "schema_version": asset_manifest.SCHEMA_VERSION,
        "source_archive_directory": str(source_root),
        "source_sha256sums_path": str(source_root / "SHA256SUMS"),
        "source_sha256sums_sha256": "b" * 64,
        "source_sizes_path": str(source_root / "SIZES"),
        "source_sizes_sha256": "c" * 64,
        "status": "passed",
    }
    _write_readonly_json(_receipt_path(root), receipt)
    return root


def test_verify_returns_runner_ready_structured_receipt(published_tree: Path) -> None:
    result = asset_manifest.verify_robocasa_asset_tree(published_tree)
    receipt_path = _receipt_path(published_tree)
    manifest_path = _manifest_path(published_tree)

    assert result == {
        "schema_version": 1,
        "kind": "cwam_robocasa_asset_tree_verification_receipt",
        "status": "passed",
        "asset_root": str(published_tree),
        "asset_root_mode": 0o755,
        "excluded_top_level_paths": [".cwam_provenance", ".git"],
        "manifest_path": str(manifest_path),
        "manifest_size_bytes": manifest_path.stat().st_size,
        "manifest_sha256": _sha256(manifest_path),
        "publisher_receipt_path": str(receipt_path),
        "publisher_receipt_size_bytes": receipt_path.stat().st_size,
        "publisher_receipt_sha256": _sha256(receipt_path),
        "publisher_generator_id": asset_manifest.GENERATOR_ID,
        "publisher_generated_at_utc": "2026-07-28T18:09:13.021394Z",
        "robocasa_git_commit": "a" * 40,
        "payload_directory_count": 2,
        "payload_file_count": 3,
        "payload_symlink_count": 0,
        "payload_total_bytes": 32,
        "source_archive_directory": str(published_tree.parent / "source-archive"),
        "source_sha256sums_sha256": "b" * 64,
        "source_sizes_sha256": "c" * 64,
    }


def test_cli_prints_verification_receipt(
    published_tree: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert asset_manifest.main(["verify", "--asset-root", str(published_tree)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "passed"
    assert output["manifest_sha256"] == _sha256(_manifest_path(published_tree))


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_receipt_requires_exact_schema(published_tree: Path, mutation: str) -> None:
    receipt = _read_receipt(published_tree)
    if mutation == "missing":
        del receipt["git_status_porcelain"]
    else:
        receipt["unexpected"] = True
    _rewrite_receipt(published_tree, receipt)

    with pytest.raises(asset_manifest.RoboCasaAssetManifestError, match="keys mismatch"):
        asset_manifest.verify_robocasa_asset_tree(published_tree)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("schema_version", True, "schema_version"),
        ("kind", "wrong", "kind"),
        ("status", "failed", "status"),
        ("generator_id", "future-generator", "supported generator"),
        ("git_commit", "A" * 40, "full lowercase Git commit"),
        ("git_status_porcelain", " M payload", "must be empty"),
        ("payload_symlink_count", 1, "must be zero"),
        ("generated_at_utc", "2026-07-28T18:09:13+08:00", "UTC Z suffix"),
    ],
)
def test_receipt_rejects_invalid_identity_fields(
    published_tree: Path,
    field: str,
    value: object,
    match: str,
) -> None:
    receipt = _read_receipt(published_tree)
    receipt[field] = value
    _rewrite_receipt(published_tree, receipt)

    with pytest.raises(asset_manifest.RoboCasaAssetManifestError, match=match):
        asset_manifest.verify_robocasa_asset_tree(published_tree)


def test_receipt_binds_exact_exclusion_policy(published_tree: Path) -> None:
    receipt = _read_receipt(published_tree)
    receipt["excluded_top_level_paths"] = [".git", ".cwam_provenance"]
    _rewrite_receipt(published_tree, receipt)

    with pytest.raises(asset_manifest.RoboCasaAssetManifestError, match="exact exclusion policy"):
        asset_manifest.verify_robocasa_asset_tree(published_tree)


def test_manifest_header_binds_exact_exclusion_policy(published_tree: Path) -> None:
    manifest = _read_manifest(published_tree)
    manifest[0]["excluded_top_level_paths"] = [".git", ".cwam_provenance"]
    _rewrite_manifest(published_tree, manifest)

    with pytest.raises(asset_manifest.RoboCasaAssetManifestError, match="exact exclusion policy"):
        asset_manifest.verify_robocasa_asset_tree(published_tree)


def test_manifest_and_receipt_bind_exact_asset_root(published_tree: Path) -> None:
    receipt = _read_receipt(published_tree)
    receipt["payload_root"] = str(published_tree.parent)
    _rewrite_receipt(published_tree, receipt)
    with pytest.raises(asset_manifest.RoboCasaAssetManifestError, match="payload_root"):
        asset_manifest.verify_robocasa_asset_tree(published_tree)

    receipt["payload_root"] = str(published_tree)
    _rewrite_receipt(published_tree, receipt)
    manifest = _read_manifest(published_tree)
    manifest[0]["payload_root"] = str(published_tree.parent)
    _rewrite_manifest(published_tree, manifest)
    with pytest.raises(asset_manifest.RoboCasaAssetManifestError, match="payload_root"):
        asset_manifest.verify_robocasa_asset_tree(published_tree)


def test_manifest_full_sha_is_bound_by_receipt(published_tree: Path) -> None:
    manifest_path = _manifest_path(published_tree)
    manifest_path.chmod(0o644)
    manifest_path.write_bytes(manifest_path.read_bytes() + b" ")
    manifest_path.chmod(asset_manifest.PROVENANCE_FILE_MODE)

    with pytest.raises(asset_manifest.RoboCasaAssetManifestError, match="does not match publisher receipt"):
        asset_manifest.verify_robocasa_asset_tree(published_tree)


def test_manifest_rejects_duplicate_json_keys(published_tree: Path) -> None:
    values = _read_manifest(published_tree)
    header_line = json.dumps(values[0], separators=(",", ":"), sort_keys=True)
    record = values[1]
    duplicate_line = (
        f'{{"mode":{record["mode"]},"path":"{record["path"]}",'
        f'"path":"{record["path"]}","sha256":"{record["sha256"]}",'
        f'"size_bytes":{record["size_bytes"]},"type":"file"}}'
    )
    remaining = [json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True) for value in values[2:]]
    manifest_path = _manifest_path(published_tree)
    manifest_path.chmod(0o644)
    manifest_path.write_text("\n".join([header_line, duplicate_line, *remaining]) + "\n", encoding="utf-8")
    manifest_path.chmod(asset_manifest.PROVENANCE_FILE_MODE)
    receipt = _read_receipt(published_tree)
    receipt["manifest_sha256"] = _sha256(manifest_path)
    _rewrite_receipt(published_tree, receipt)

    with pytest.raises(asset_manifest.RoboCasaAssetManifestError, match="duplicate object key"):
        asset_manifest.verify_robocasa_asset_tree(published_tree)


@pytest.mark.parametrize(
    "invalid_path",
    [
        "../escape",
        "/absolute",
        "assets//texture.png",
        "assets/.incomplete-stage/file",
        "assets\\texture.png",
    ],
)
def test_manifest_rejects_noncanonical_and_incomplete_paths(
    published_tree: Path,
    invalid_path: str,
) -> None:
    manifest = _read_manifest(published_tree)
    manifest[1]["path"] = invalid_path
    _rewrite_manifest(published_tree, manifest)

    with pytest.raises(asset_manifest.RoboCasaAssetManifestError, match="path"):
        asset_manifest.verify_robocasa_asset_tree(published_tree)


def test_manifest_rejects_duplicate_payload_paths(published_tree: Path) -> None:
    manifest = _read_manifest(published_tree)
    manifest.append(dict(manifest[1]))
    _rewrite_manifest(published_tree, manifest)

    with pytest.raises(asset_manifest.RoboCasaAssetManifestError, match="duplicate path"):
        asset_manifest.verify_robocasa_asset_tree(published_tree)


def test_manifest_rejects_unknown_record_keys_and_types(published_tree: Path) -> None:
    manifest = _read_manifest(published_tree)
    manifest[1]["unexpected"] = 1
    _rewrite_manifest(published_tree, manifest)
    with pytest.raises(asset_manifest.RoboCasaAssetManifestError, match="keys mismatch"):
        asset_manifest.verify_robocasa_asset_tree(published_tree)

    manifest = _read_manifest(published_tree)
    manifest[1].pop("unexpected")
    manifest[1]["type"] = "symlink"
    _rewrite_manifest(published_tree, manifest)
    with pytest.raises(asset_manifest.RoboCasaAssetManifestError, match="type must be"):
        asset_manifest.verify_robocasa_asset_tree(published_tree)


def test_manifest_rejects_nondeterministic_record_order(published_tree: Path) -> None:
    manifest = _read_manifest(published_tree)
    manifest[1], manifest[2] = manifest[2], manifest[1]
    _rewrite_manifest(published_tree, manifest)

    with pytest.raises(asset_manifest.RoboCasaAssetManifestError, match="record order"):
        asset_manifest.verify_robocasa_asset_tree(published_tree)


def test_receipt_counts_and_total_bytes_must_match_manifest(published_tree: Path) -> None:
    for field in ("payload_directory_count", "payload_file_count", "payload_total_bytes"):
        receipt = _read_receipt(published_tree)
        receipt[field] += 1
        _rewrite_receipt(published_tree, receipt)
        with pytest.raises(asset_manifest.RoboCasaAssetManifestError, match="does not match publisher receipt"):
            asset_manifest.verify_robocasa_asset_tree(published_tree)
        receipt[field] -= 1
        _rewrite_receipt(published_tree, receipt)


def test_receipt_source_descriptors_are_exactly_bound(published_tree: Path) -> None:
    receipt = _read_receipt(published_tree)
    receipt["source_sizes_path"] = str(published_tree.parent / "elsewhere" / "SIZES")
    _rewrite_receipt(published_tree, receipt)

    with pytest.raises(asset_manifest.RoboCasaAssetManifestError, match="source_sizes_path"):
        asset_manifest.verify_robocasa_asset_tree(published_tree)


def test_live_tree_rejects_extra_and_missing_entries(published_tree: Path) -> None:
    extra = published_tree / "unexpected.txt"
    extra.write_text("extra", encoding="utf-8")
    extra.chmod(0o644)
    with pytest.raises(asset_manifest.RoboCasaAssetManifestError, match="extra=.*unexpected.txt"):
        asset_manifest.verify_robocasa_asset_tree(published_tree)
    extra.unlink()

    (published_tree / "README.md").unlink()
    with pytest.raises(asset_manifest.RoboCasaAssetManifestError, match="missing=.*README.md"):
        asset_manifest.verify_robocasa_asset_tree(published_tree)


def test_live_tree_rejects_symlink(published_tree: Path) -> None:
    payload_file = published_tree / "README.md"
    payload_file.unlink()
    payload_file.symlink_to("assets/texture.png")

    with pytest.raises(asset_manifest.RoboCasaAssetManifestError, match="must not contain symlinks"):
        asset_manifest.verify_robocasa_asset_tree(published_tree)


def test_live_tree_rejects_special_file(published_tree: Path) -> None:
    fifo = published_tree / "payload.fifo"
    os.mkfifo(fifo)

    with pytest.raises(asset_manifest.RoboCasaAssetManifestError, match="special file"):
        asset_manifest.verify_robocasa_asset_tree(published_tree)


def test_live_tree_rejects_incomplete_path(published_tree: Path) -> None:
    incomplete = published_tree / "payload.incomplete-stage"
    incomplete.write_text("not published", encoding="utf-8")

    with pytest.raises(asset_manifest.RoboCasaAssetManifestError, match=r"\.incomplete"):
        asset_manifest.verify_robocasa_asset_tree(published_tree)


def test_live_tree_rejects_type_and_mode_drift(published_tree: Path) -> None:
    readme = published_tree / "README.md"
    readme.chmod(0o600)
    with pytest.raises(asset_manifest.RoboCasaAssetManifestError, match="mode mismatch"):
        asset_manifest.verify_robocasa_asset_tree(published_tree)

    readme.unlink()
    readme.mkdir(mode=0o644)
    with pytest.raises(asset_manifest.RoboCasaAssetManifestError, match="type mismatch"):
        asset_manifest.verify_robocasa_asset_tree(published_tree)


def test_live_tree_rejects_size_and_full_sha_drift(published_tree: Path) -> None:
    readme = published_tree / "README.md"
    original = readme.read_bytes()
    readme.write_bytes(original + b"x")
    with pytest.raises(asset_manifest.RoboCasaAssetManifestError, match="size mismatch"):
        asset_manifest.verify_robocasa_asset_tree(published_tree)

    readme.write_bytes(b"X" * len(original))
    with pytest.raises(asset_manifest.RoboCasaAssetManifestError, match="SHA-256 mismatch"):
        asset_manifest.verify_robocasa_asset_tree(published_tree)


def test_manifest_record_mode_drift_is_rejected(published_tree: Path) -> None:
    manifest = _read_manifest(published_tree)
    manifest[1]["mode"] = 0o600
    _rewrite_manifest(published_tree, manifest)

    with pytest.raises(asset_manifest.RoboCasaAssetManifestError, match="mode mismatch"):
        asset_manifest.verify_robocasa_asset_tree(published_tree)


def test_provenance_files_must_be_readonly_regular_files(published_tree: Path) -> None:
    receipt_path = _receipt_path(published_tree)
    receipt_path.chmod(0o644)
    with pytest.raises(asset_manifest.RoboCasaAssetManifestError, match="mode mismatch"):
        asset_manifest.verify_robocasa_asset_tree(published_tree)

    receipt_bytes = receipt_path.read_bytes()
    receipt_path.unlink()
    target = published_tree.parent / "receipt.json"
    target.write_bytes(receipt_bytes)
    target.chmod(asset_manifest.PROVENANCE_FILE_MODE)
    receipt_path.symlink_to(target)
    with pytest.raises(asset_manifest.RoboCasaAssetManifestError, match="symlink components"):
        asset_manifest.verify_robocasa_asset_tree(published_tree)


def test_required_excluded_directories_must_exist_and_not_be_symlinks(published_tree: Path) -> None:
    git_directory = published_tree / ".git"
    (git_directory / "HEAD").unlink()
    git_directory.rmdir()

    with pytest.raises(asset_manifest.RoboCasaAssetManifestError, match="excluded top-level paths are missing"):
        asset_manifest.verify_robocasa_asset_tree(published_tree)


def test_verification_receipt_is_readonly_and_never_deletes_a_racing_owner(
    published_tree: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = asset_manifest.verify_robocasa_asset_tree(published_tree)
    output = tmp_path / "control" / "asset-verification.json"
    output.parent.mkdir()
    digest = asset_manifest.write_verification_receipt(output, result)
    assert hashlib.sha256(output.read_bytes()).hexdigest() == digest
    assert stat.S_IMODE(output.stat().st_mode) == asset_manifest.PROVENANCE_FILE_MODE

    racing_output = output.with_name("asset-verification-race.json")

    def racing_link(_source: Path, target: Path, **_kwargs: object) -> None:
        Path(target).write_text("competing owner\n", encoding="utf-8")
        raise FileExistsError(target)

    monkeypatch.setattr(asset_manifest.os, "link", racing_link)
    with pytest.raises(FileExistsError):
        asset_manifest.write_verification_receipt(racing_output, result)
    assert racing_output.read_text(encoding="utf-8") == "competing owner\n"
