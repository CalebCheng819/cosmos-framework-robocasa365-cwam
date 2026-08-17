from __future__ import annotations

import json
from pathlib import Path

import cwam_stage_source_snapshot as staging
import pytest


def _write_manifest(source: Path, entries: dict[str, bytes]) -> Path:
    files = []
    for relative, payload in entries.items():
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        files.append({"path": relative, "bytes": len(payload), "sha256": "legacy-not-consumed"})
    manifest = source / "SNAPSHOT_MANIFEST.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_dirty": False,
                "source_head": "0" * 40,
                "file_count": len(files),
                "files": files,
            }
        ),
        encoding="utf-8",
    )
    return manifest


def test_stage_source_snapshot_copies_only_manifest_files_and_seals_tree(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    manifest = _write_manifest(source, {"sitecustomize.py": b"pass\n", "pkg/module.py": b"VALUE = 1\n"})
    cache = source / "__pycache__"
    cache.mkdir()
    (cache / "sitecustomize.cpython-311.pyc").write_bytes(b"untracked")
    destination = tmp_path / "worker" / "source"

    receipt = staging.stage_source_snapshot(
        source_root=source,
        manifest_path=manifest,
        destination_root=destination,
    )

    assert receipt["manifest_file_count"] == 2
    assert not (destination / "__pycache__").exists()
    assert (destination / "pkg/module.py").read_bytes() == b"VALUE = 1\n"
    assert (destination.stat().st_mode & 0o777) == 0o555
    assert ((destination / "pkg").stat().st_mode & 0o777) == 0o555
    assert ((destination / "pkg/module.py").stat().st_mode & 0o777) == 0o444
    assert ((destination / "SNAPSHOT_MANIFEST.json").stat().st_mode & 0o777) == 0o444


def test_stage_source_snapshot_rejects_path_traversal(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    manifest = source / "SNAPSHOT_MANIFEST.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_dirty": False,
                "file_count": 1,
                "files": [{"path": "../escape.py", "bytes": 0}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(staging.SourceStagingError, match="unsafe"):
        staging.stage_source_snapshot(
            source_root=source,
            manifest_path=manifest,
            destination_root=tmp_path / "destination",
        )


def test_stage_source_snapshot_rejects_symlinked_parent(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "module.py").write_bytes(b"pass\n")
    source = tmp_path / "source"
    source.mkdir()
    (source / "pkg").symlink_to(outside, target_is_directory=True)
    manifest = source / "SNAPSHOT_MANIFEST.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_dirty": False,
                "file_count": 1,
                "files": [{"path": "pkg/module.py", "bytes": 5}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(staging.SourceStagingError, match="symlinked parent"):
        staging.stage_source_snapshot(
            source_root=source,
            manifest_path=manifest,
            destination_root=tmp_path / "destination",
        )
