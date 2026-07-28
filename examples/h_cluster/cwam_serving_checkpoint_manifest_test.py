# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""CPU-only tests for strict CWAM serving checkpoint manifests."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import pytest

from examples.h_cluster import cwam_serving_checkpoint_manifest as checkpoint_manifest


def _checkpoint(tmp_path: Path, name: str = "checkpoint") -> Path:
    root = tmp_path / name
    (root / "nested").mkdir(parents=True)
    (root / "model.bin").write_bytes(b"model-weights")
    (root / "nested" / "config.json").write_text('{"layers": 2}\n', encoding="utf-8")
    return root


def _tree_sha256(files: list[dict[str, object]]) -> str:
    payload = json.dumps(
        files,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def test_create_and_verify_exact_checkpoint_tree(tmp_path: Path) -> None:
    root = _checkpoint(tmp_path)

    manifest = checkpoint_manifest.create_serving_checkpoint_manifest(root)
    repeated = checkpoint_manifest.create_serving_checkpoint_manifest(root)

    assert manifest == repeated
    assert set(manifest) == {
        "schema_version",
        "kind",
        "checkpoint_path",
        "files",
        "tree_sha256",
    }
    assert manifest["schema_version"] == 1
    assert manifest["kind"] == "cwam_serving_checkpoint_manifest"
    assert manifest["checkpoint_path"] == str(root.resolve())
    assert [record["relative_path"] for record in manifest["files"]] == [
        "model.bin",
        "nested/config.json",
    ]

    report = checkpoint_manifest.verify_serving_checkpoint_manifest(manifest, root)
    assert report == {
        "status": "passed",
        "checkpoint_path": str(root.resolve()),
        "file_count": 2,
        "tree_sha256": manifest["tree_sha256"],
    }


def test_verify_rehashes_every_file_and_rejects_same_size_tamper(tmp_path: Path) -> None:
    root = _checkpoint(tmp_path)
    manifest = checkpoint_manifest.create_serving_checkpoint_manifest(root)
    target = root / "model.bin"
    target.write_bytes(b"MODEL-WEIGHTS")
    assert target.stat().st_size == manifest["files"][0]["size_bytes"]

    with pytest.raises(checkpoint_manifest.ServingCheckpointManifestError, match="SHA-256 mismatch"):
        checkpoint_manifest.verify_serving_checkpoint_manifest(manifest, root)


def test_verify_rejects_size_drift(tmp_path: Path) -> None:
    root = _checkpoint(tmp_path)
    manifest = checkpoint_manifest.create_serving_checkpoint_manifest(root)
    (root / "model.bin").write_bytes(b"larger-model-weights")

    with pytest.raises(checkpoint_manifest.ServingCheckpointManifestError, match="size mismatch"):
        checkpoint_manifest.verify_serving_checkpoint_manifest(manifest, root)


def test_verify_rejects_extra_and_missing_files(tmp_path: Path) -> None:
    root = _checkpoint(tmp_path, "extra")
    manifest = checkpoint_manifest.create_serving_checkpoint_manifest(root)
    (root / "extra.bin").write_bytes(b"unexpected")

    with pytest.raises(checkpoint_manifest.ServingCheckpointManifestError, match=r"extra=\['extra.bin'\]"):
        checkpoint_manifest.verify_serving_checkpoint_manifest(manifest, root)

    root = _checkpoint(tmp_path, "missing")
    manifest = checkpoint_manifest.create_serving_checkpoint_manifest(root)
    (root / "model.bin").unlink()

    with pytest.raises(checkpoint_manifest.ServingCheckpointManifestError, match=r"missing=\['model.bin'\]"):
        checkpoint_manifest.verify_serving_checkpoint_manifest(manifest, root)


def test_rejects_symlinks_in_tree_and_at_checkpoint_root(tmp_path: Path) -> None:
    root = _checkpoint(tmp_path, "tree-link")
    (root / "alias.bin").symlink_to("model.bin")

    with pytest.raises(checkpoint_manifest.ServingCheckpointManifestError, match="must not contain symlinks"):
        checkpoint_manifest.create_serving_checkpoint_manifest(root)

    target = _checkpoint(tmp_path, "target")
    alias = tmp_path / "checkpoint-link"
    alias.symlink_to(target, target_is_directory=True)

    with pytest.raises(checkpoint_manifest.ServingCheckpointManifestError, match="no symlink components"):
        checkpoint_manifest.create_serving_checkpoint_manifest(alias)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO creation is not supported")
def test_rejects_special_files(tmp_path: Path) -> None:
    root = _checkpoint(tmp_path)
    os.mkfifo(root / "named-pipe")

    with pytest.raises(checkpoint_manifest.ServingCheckpointManifestError, match="special file"):
        checkpoint_manifest.create_serving_checkpoint_manifest(root)


@pytest.mark.parametrize(
    "relative_path",
    [
        ".incomplete",
        ".incomplete-stage/shard.bin",
        "weights.incomplete",
        "nested/model.incomplete-copy",
    ],
)
def test_manifest_rejects_every_incomplete_name(tmp_path: Path, relative_path: str) -> None:
    root = _checkpoint(tmp_path)
    manifest = checkpoint_manifest.create_serving_checkpoint_manifest(root)
    manifest["files"][0]["relative_path"] = relative_path

    with pytest.raises(checkpoint_manifest.ServingCheckpointManifestError, match=r"forbidden \.incomplete"):
        checkpoint_manifest.verify_serving_checkpoint_manifest(manifest, root)


def test_tree_scan_rejects_incomplete_names_and_empty_trees(tmp_path: Path) -> None:
    root = _checkpoint(tmp_path, "with-incomplete")
    (root / ".incomplete-shard").write_bytes(b"partial")
    with pytest.raises(checkpoint_manifest.ServingCheckpointManifestError, match=r"forbidden \.incomplete"):
        checkpoint_manifest.create_serving_checkpoint_manifest(root)

    root = _checkpoint(tmp_path, "with-incomplete-directory")
    (root / "stage.incomplete-123").mkdir()
    with pytest.raises(checkpoint_manifest.ServingCheckpointManifestError, match=r"forbidden \.incomplete"):
        checkpoint_manifest.create_serving_checkpoint_manifest(root)

    incomplete_root = _checkpoint(tmp_path, "checkpoint.incomplete-build")
    with pytest.raises(checkpoint_manifest.ServingCheckpointManifestError, match=r"forbidden \.incomplete"):
        checkpoint_manifest.create_serving_checkpoint_manifest(incomplete_root)

    empty = tmp_path / "empty"
    (empty / "directory-only").mkdir(parents=True)
    with pytest.raises(checkpoint_manifest.ServingCheckpointManifestError, match="at least one regular file"):
        checkpoint_manifest.create_serving_checkpoint_manifest(empty)


def test_manifest_rejects_duplicate_and_unsorted_paths(tmp_path: Path) -> None:
    root = _checkpoint(tmp_path)
    manifest = checkpoint_manifest.create_serving_checkpoint_manifest(root)
    manifest["files"].append(copy.deepcopy(manifest["files"][0]))
    manifest["tree_sha256"] = _tree_sha256(manifest["files"])

    with pytest.raises(checkpoint_manifest.ServingCheckpointManifestError, match="duplicate relative_path"):
        checkpoint_manifest.verify_serving_checkpoint_manifest(manifest, root)

    manifest = checkpoint_manifest.create_serving_checkpoint_manifest(root)
    manifest["files"].reverse()
    manifest["tree_sha256"] = _tree_sha256(manifest["files"])

    with pytest.raises(checkpoint_manifest.ServingCheckpointManifestError, match="sorted by relative_path"):
        checkpoint_manifest.verify_serving_checkpoint_manifest(manifest, root)


@pytest.mark.parametrize(
    "relative_path",
    [
        "",
        "/absolute.bin",
        "../escape.bin",
        "nested/../escape.bin",
        "./model.bin",
        "nested//config.json",
        "nested\\config.json",
        "model.bin/",
        "bad\nname.bin",
    ],
)
def test_manifest_rejects_noncanonical_relative_paths(tmp_path: Path, relative_path: str) -> None:
    root = _checkpoint(tmp_path)
    manifest = checkpoint_manifest.create_serving_checkpoint_manifest(root)
    manifest["files"][0]["relative_path"] = relative_path

    with pytest.raises(checkpoint_manifest.ServingCheckpointManifestError, match="relative_path"):
        checkpoint_manifest.verify_serving_checkpoint_manifest(manifest, root)


def test_verify_requires_exact_resolved_checkpoint_path(tmp_path: Path) -> None:
    first = _checkpoint(tmp_path, "first")
    second = _checkpoint(tmp_path, "second")
    manifest = checkpoint_manifest.create_serving_checkpoint_manifest(first)

    with pytest.raises(checkpoint_manifest.ServingCheckpointManifestError, match="exact resolved checkpoint path"):
        checkpoint_manifest.verify_serving_checkpoint_manifest(manifest, second)

    manifest["checkpoint_path"] = f"{first.resolve()}/"
    with pytest.raises(checkpoint_manifest.ServingCheckpointManifestError, match="exact resolved checkpoint path"):
        checkpoint_manifest.verify_serving_checkpoint_manifest(manifest, first)


def test_manifest_rejects_schema_and_tree_digest_drift(tmp_path: Path) -> None:
    root = _checkpoint(tmp_path)

    manifest = checkpoint_manifest.create_serving_checkpoint_manifest(root)
    manifest["schema_version"] = True
    with pytest.raises(checkpoint_manifest.ServingCheckpointManifestError, match="schema_version"):
        checkpoint_manifest.verify_serving_checkpoint_manifest(manifest, root)

    manifest = checkpoint_manifest.create_serving_checkpoint_manifest(root)
    manifest["kind"] = "other"
    with pytest.raises(checkpoint_manifest.ServingCheckpointManifestError, match="manifest kind"):
        checkpoint_manifest.verify_serving_checkpoint_manifest(manifest, root)

    manifest = checkpoint_manifest.create_serving_checkpoint_manifest(root)
    manifest["tree_sha256"] = "0" * 64
    with pytest.raises(checkpoint_manifest.ServingCheckpointManifestError, match="tree_sha256 mismatch"):
        checkpoint_manifest.verify_serving_checkpoint_manifest(manifest, root)


def test_manifest_rejects_missing_extra_or_invalid_file_fields(tmp_path: Path) -> None:
    root = _checkpoint(tmp_path)

    manifest = checkpoint_manifest.create_serving_checkpoint_manifest(root)
    del manifest["files"][0]["size_bytes"]
    with pytest.raises(checkpoint_manifest.ServingCheckpointManifestError, match="keys mismatch"):
        checkpoint_manifest.verify_serving_checkpoint_manifest(manifest, root)

    manifest = checkpoint_manifest.create_serving_checkpoint_manifest(root)
    manifest["files"][0]["unexpected"] = "field"
    with pytest.raises(checkpoint_manifest.ServingCheckpointManifestError, match="keys mismatch"):
        checkpoint_manifest.verify_serving_checkpoint_manifest(manifest, root)

    manifest = checkpoint_manifest.create_serving_checkpoint_manifest(root)
    manifest["files"][0]["size_bytes"] = True
    with pytest.raises(checkpoint_manifest.ServingCheckpointManifestError, match="non-negative integer"):
        checkpoint_manifest.verify_serving_checkpoint_manifest(manifest, root)

    manifest = checkpoint_manifest.create_serving_checkpoint_manifest(root)
    manifest["files"][0]["sha256"] = "A" * 64
    with pytest.raises(checkpoint_manifest.ServingCheckpointManifestError, match="lowercase SHA-256"):
        checkpoint_manifest.verify_serving_checkpoint_manifest(manifest, root)


def test_manifest_rejects_missing_extra_top_level_fields_and_empty_files(tmp_path: Path) -> None:
    root = _checkpoint(tmp_path)

    manifest = checkpoint_manifest.create_serving_checkpoint_manifest(root)
    del manifest["kind"]
    with pytest.raises(checkpoint_manifest.ServingCheckpointManifestError, match="keys mismatch"):
        checkpoint_manifest.verify_serving_checkpoint_manifest(manifest, root)

    manifest = checkpoint_manifest.create_serving_checkpoint_manifest(root)
    manifest["unexpected"] = "field"
    with pytest.raises(checkpoint_manifest.ServingCheckpointManifestError, match="keys mismatch"):
        checkpoint_manifest.verify_serving_checkpoint_manifest(manifest, root)

    manifest = checkpoint_manifest.create_serving_checkpoint_manifest(root)
    manifest["files"] = []
    manifest["tree_sha256"] = _tree_sha256([])
    with pytest.raises(checkpoint_manifest.ServingCheckpointManifestError, match="non-empty JSON array"):
        checkpoint_manifest.verify_serving_checkpoint_manifest(manifest, root)


def test_loader_rejects_duplicate_json_keys_symlinks_and_incomplete_names(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version": 1, "schema_version": 1}\n', encoding="utf-8")
    with pytest.raises(checkpoint_manifest.ServingCheckpointManifestError, match="duplicate object key"):
        checkpoint_manifest.load_serving_checkpoint_manifest(duplicate)

    target = tmp_path / "manifest.json"
    target.write_text("{}\n", encoding="utf-8")
    alias = tmp_path / "manifest-link.json"
    alias.symlink_to(target)
    with pytest.raises(checkpoint_manifest.ServingCheckpointManifestError, match="must not be a symlink"):
        checkpoint_manifest.load_serving_checkpoint_manifest(alias)

    incomplete = tmp_path / "manifest.incomplete.json"
    incomplete.write_text("{}\n", encoding="utf-8")
    with pytest.raises(checkpoint_manifest.ServingCheckpointManifestError, match=r"forbidden \.incomplete"):
        checkpoint_manifest.load_serving_checkpoint_manifest(incomplete)


def test_cli_create_and_verify_are_cpu_only(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = _checkpoint(tmp_path)
    assert checkpoint_manifest.main(["create", "--checkpoint-path", str(root)]) == 0
    manifest = json.loads(capsys.readouterr().out)

    manifest_path = tmp_path / "serving-checkpoint-manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert (
        checkpoint_manifest.main(
            [
                "verify",
                "--checkpoint-path",
                str(root),
                "--manifest",
                str(manifest_path),
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "passed"
