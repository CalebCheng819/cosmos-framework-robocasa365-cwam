# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""CPU-only tests for strict Python environment manifests."""

from __future__ import annotations

import codecs
import copy
import json
import os
from pathlib import Path

import pytest

from examples.h_cluster import cwam_python_environment_manifest as environment_manifest


class _Distribution:
    def __init__(self, name: object, version: object) -> None:
        self.metadata = {"Name": name}
        self.version = version


@pytest.fixture(scope="module")
def live_manifest() -> dict:
    return environment_manifest.create_python_environment_manifest()


def _write_manifest(path: Path, manifest: object) -> Path:
    path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    return path


def _rehash_distributions(manifest: dict) -> None:
    manifest["distribution_count"] = len(manifest["distributions"])
    manifest["distributions_sha256"] = environment_manifest._distributions_sha256(manifest["distributions"])


def test_create_and_verify_current_environment_are_deterministic(live_manifest: dict) -> None:
    repeated = environment_manifest.create_python_environment_manifest()

    assert repeated == live_manifest
    assert set(live_manifest) == {
        "schema_version",
        "kind",
        "interpreter",
        "distribution_count",
        "distributions",
        "distributions_sha256",
    }
    assert live_manifest["schema_version"] == 1
    assert live_manifest["kind"] == "cwam_python_environment_manifest"
    assert live_manifest["distribution_count"] == len(live_manifest["distributions"])
    assert list(live_manifest["distributions"]) == sorted(live_manifest["distributions"])

    report = environment_manifest.verify_python_environment_manifest(live_manifest)
    assert report["status"] == "passed"
    assert report["interpreter_resolved_path"] == live_manifest["interpreter"]["resolved_path"]
    assert report["interpreter_sha256"] == live_manifest["interpreter"]["sha256"]
    assert report["python_version"] == live_manifest["interpreter"]["python_version"]
    assert report["distribution_count"] == live_manifest["distribution_count"]
    assert report["distributions_sha256"] == live_manifest["distributions_sha256"]


def test_manifest_binds_exact_resolved_interpreter_and_runtime(live_manifest: dict) -> None:
    interpreter = live_manifest["interpreter"]

    assert Path(interpreter["resolved_path"]).is_absolute()
    assert Path(interpreter["resolved_path"]).resolve(strict=True) == Path(interpreter["resolved_path"])
    assert Path(interpreter["prefix"]).resolve(strict=True) == Path(interpreter["prefix"])
    assert Path(interpreter["base_prefix"]).resolve(strict=True) == Path(interpreter["base_prefix"])
    assert len(interpreter["sha256"]) == 64
    assert interpreter["size_bytes"] > 0
    version = interpreter["version_info"]
    assert interpreter["python_version"] == f"{version['major']}.{version['minor']}.{version['micro']}"


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("sha256", "0" * 64, "interpreter_fields"),
        ("size_bytes", 1, "interpreter_fields"),
        ("implementation", "different", "interpreter_fields"),
        ("cache_tag", "different-000", "interpreter_fields"),
    ],
)
def test_verify_rejects_interpreter_identity_drift(
    live_manifest: dict,
    field: str,
    value: object,
    match: str,
) -> None:
    manifest = copy.deepcopy(live_manifest)
    manifest["interpreter"][field] = value

    with pytest.raises(environment_manifest.PythonEnvironmentManifestError, match=match):
        environment_manifest.verify_python_environment_manifest(manifest)


def test_verify_rejects_python_version_drift(live_manifest: dict) -> None:
    manifest = copy.deepcopy(live_manifest)
    manifest["interpreter"]["version_info"]["micro"] += 1
    version = manifest["interpreter"]["version_info"]
    manifest["interpreter"]["python_version"] = f"{version['major']}.{version['minor']}.{version['micro']}"

    with pytest.raises(
        environment_manifest.PythonEnvironmentManifestError,
        match="interpreter_fields",
    ):
        environment_manifest.verify_python_environment_manifest(manifest)


def test_verify_rejects_distribution_add_remove_and_version_drift(live_manifest: dict) -> None:
    added = copy.deepcopy(live_manifest)
    added["distributions"]["definitely-not-installed"] = {
        "name": "definitely-not-installed",
        "version": "1.0",
    }
    _rehash_distributions(added)
    with pytest.raises(environment_manifest.PythonEnvironmentManifestError, match="missing="):
        environment_manifest.verify_python_environment_manifest(added)

    if live_manifest["distributions"]:
        name = next(iter(live_manifest["distributions"]))
        removed = copy.deepcopy(live_manifest)
        del removed["distributions"][name]
        _rehash_distributions(removed)
        with pytest.raises(environment_manifest.PythonEnvironmentManifestError, match="extra="):
            environment_manifest.verify_python_environment_manifest(removed)

        changed = copy.deepcopy(live_manifest)
        changed["distributions"][name]["version"] += ".changed"
        _rehash_distributions(changed)
        with pytest.raises(environment_manifest.PythonEnvironmentManifestError, match="changed="):
            environment_manifest.verify_python_environment_manifest(changed)


def test_distribution_collection_is_sorted_and_normalizes_pep503_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        environment_manifest.importlib.metadata,
        "distributions",
        lambda: [
            _Distribution("Zoo_Pkg", "2.0"),
            _Distribution("alpha.pkg", "1.0"),
        ],
    )

    assert environment_manifest._collect_distribution_map_once() == {
        "alpha-pkg": {"name": "alpha.pkg", "version": "1.0"},
        "zoo-pkg": {"name": "Zoo_Pkg", "version": "2.0"},
    }


@pytest.mark.parametrize(
    ("distributions", "match"),
    [
        (
            [_Distribution("Foo_Bar", "1"), _Distribution("foo-bar", "1")],
            "duplicate canonical distribution name",
        ),
        ([_Distribution(None, "1")], "non-empty, trimmed"),
        ([_Distribution("bad name", "1")], "whitespace/control"),
        ([_Distribution("valid-name", None)], "non-empty, trimmed"),
        ([_Distribution("valid-name", "1 beta")], "whitespace/control"),
    ],
)
def test_distribution_collection_rejects_ambiguous_or_invalid_metadata(
    monkeypatch: pytest.MonkeyPatch,
    distributions: list[_Distribution],
    match: str,
) -> None:
    monkeypatch.setattr(
        environment_manifest.importlib.metadata,
        "distributions",
        lambda: distributions,
    )

    with pytest.raises(environment_manifest.PythonEnvironmentManifestError, match=match):
        environment_manifest._collect_distribution_map_once()


def test_create_rejects_inventory_or_interpreter_mutation(
    monkeypatch: pytest.MonkeyPatch,
    live_manifest: dict,
) -> None:
    interpreter = live_manifest["interpreter"]
    monkeypatch.setattr(
        environment_manifest,
        "_collect_interpreter_record",
        lambda: copy.deepcopy(interpreter),
    )
    inventories = iter(
        [
            {"alpha": {"name": "alpha", "version": "1"}},
            {"alpha": {"name": "alpha", "version": "2"}},
        ]
    )
    monkeypatch.setattr(
        environment_manifest,
        "_collect_distribution_map_once",
        lambda: next(inventories),
    )
    with pytest.raises(environment_manifest.PythonEnvironmentManifestError, match="inventory changed"):
        environment_manifest.create_python_environment_manifest()

    interpreter_records = iter(
        [
            copy.deepcopy(interpreter),
            {**copy.deepcopy(interpreter), "sha256": "0" * 64},
        ]
    )
    monkeypatch.setattr(
        environment_manifest,
        "_collect_interpreter_record",
        lambda: next(interpreter_records),
    )
    monkeypatch.setattr(
        environment_manifest,
        "_collect_distribution_map_once",
        lambda: {"alpha": {"name": "alpha", "version": "1"}},
    )
    with pytest.raises(environment_manifest.PythonEnvironmentManifestError, match="identity changed"):
        environment_manifest.create_python_environment_manifest()


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda value: value.update(schema_version=True), "schema_version"),
        (lambda value: value.update(kind="other"), "manifest.kind"),
        (lambda value: value.update(distribution_count=True), "non-negative integer"),
        (lambda value: value.update(unexpected="field"), "keys mismatch"),
        (lambda value: value["interpreter"].update(unexpected="field"), "keys mismatch"),
        (
            lambda value: value["interpreter"]["version_info"].update(unexpected="field"),
            "keys mismatch",
        ),
    ],
)
def test_manifest_requires_exact_schema(
    live_manifest: dict,
    mutation,
    match: str,
) -> None:
    manifest = copy.deepcopy(live_manifest)
    mutation(manifest)

    with pytest.raises(environment_manifest.PythonEnvironmentManifestError, match=match):
        environment_manifest.verify_python_environment_manifest(manifest)


@pytest.mark.parametrize("value", [[], None, True, "manifest"])
def test_verify_rejects_non_object_manifests(value: object) -> None:
    with pytest.raises(environment_manifest.PythonEnvironmentManifestError, match="must be a JSON object"):
        environment_manifest.verify_python_environment_manifest(value)


def test_manifest_rejects_distribution_count_hash_and_record_drift(live_manifest: dict) -> None:
    manifest = copy.deepcopy(live_manifest)
    manifest["distribution_count"] += 1
    with pytest.raises(environment_manifest.PythonEnvironmentManifestError, match="count mismatch"):
        environment_manifest.verify_python_environment_manifest(manifest)

    manifest = copy.deepcopy(live_manifest)
    manifest["distributions_sha256"] = "0" * 64
    with pytest.raises(environment_manifest.PythonEnvironmentManifestError, match="sha256 mismatch"):
        environment_manifest.verify_python_environment_manifest(manifest)

    if live_manifest["distributions"]:
        name = next(iter(live_manifest["distributions"]))
        manifest = copy.deepcopy(live_manifest)
        manifest["distributions"][name]["unexpected"] = "field"
        with pytest.raises(environment_manifest.PythonEnvironmentManifestError, match="keys mismatch"):
            environment_manifest.verify_python_environment_manifest(manifest)

        manifest = copy.deepcopy(live_manifest)
        manifest["distributions"][name]["name"] = "different-name"
        _rehash_distributions(manifest)
        with pytest.raises(environment_manifest.PythonEnvironmentManifestError, match="key/name mismatch"):
            environment_manifest.verify_python_environment_manifest(manifest)


def test_manifest_bound_paths_reject_symlinks_and_incomplete_names(
    live_manifest: dict,
    tmp_path: Path,
) -> None:
    interpreter = Path(live_manifest["interpreter"]["resolved_path"])
    interpreter_alias = tmp_path / "python-alias"
    interpreter_alias.symlink_to(interpreter)
    manifest = copy.deepcopy(live_manifest)
    manifest["interpreter"]["resolved_path"] = str(interpreter_alias)
    with pytest.raises(environment_manifest.PythonEnvironmentManifestError, match="no symlink components"):
        environment_manifest.verify_python_environment_manifest(manifest)

    prefix_alias = tmp_path / "prefix-alias"
    prefix_alias.symlink_to(Path(live_manifest["interpreter"]["prefix"]), target_is_directory=True)
    manifest = copy.deepcopy(live_manifest)
    manifest["interpreter"]["prefix"] = str(prefix_alias)
    with pytest.raises(environment_manifest.PythonEnvironmentManifestError, match="no symlink components"):
        environment_manifest.verify_python_environment_manifest(manifest)

    manifest = copy.deepcopy(live_manifest)
    manifest["interpreter"]["resolved_path"] = str(tmp_path / "python.incomplete-copy")
    with pytest.raises(environment_manifest.PythonEnvironmentManifestError, match=r"\.incomplete"):
        environment_manifest.verify_python_environment_manifest(manifest)


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        (b'{"schema_version":1,"schema_version":1}', "duplicate object key"),
        (b'{"outer":{"name":"first","\\u006eame":"second"}}', "duplicate object key"),
        (b'{"value":NaN}', "forbidden constant"),
        (codecs.BOM_UTF8 + b"{}", "UTF-8 BOM"),
        (b"\xff", "valid UTF-8"),
        (b"[]", "root must be an object"),
        (b"{", "valid JSON"),
    ],
)
def test_manifest_loader_rejects_ambiguous_or_invalid_json(
    tmp_path: Path,
    payload: bytes,
    match: str,
) -> None:
    path = tmp_path / "manifest.json"
    path.write_bytes(payload)

    with pytest.raises(environment_manifest.PythonEnvironmentManifestError, match=match):
        environment_manifest.load_python_environment_manifest(path)


def test_manifest_loader_rejects_symlink_components_incomplete_special_and_oversize(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "manifest.json"
    target.write_text("{}\n", encoding="utf-8")
    alias = tmp_path / "manifest-alias.json"
    alias.symlink_to(target)
    with pytest.raises(environment_manifest.PythonEnvironmentManifestError, match="no symlink components"):
        environment_manifest.load_python_environment_manifest(alias)

    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    nested = real_parent / "manifest.json"
    nested.write_text("{}\n", encoding="utf-8")
    parent_alias = tmp_path / "parent-alias"
    parent_alias.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(environment_manifest.PythonEnvironmentManifestError, match="no symlink components"):
        environment_manifest.load_python_environment_manifest(parent_alias / "manifest.json")

    broken = tmp_path / "broken-manifest.json"
    broken.symlink_to(tmp_path / "missing-manifest.json")
    with pytest.raises(environment_manifest.PythonEnvironmentManifestError, match="does not exist"):
        environment_manifest.load_python_environment_manifest(broken)

    first_loop = tmp_path / "first-loop.json"
    second_loop = tmp_path / "second-loop.json"
    first_loop.symlink_to(second_loop)
    second_loop.symlink_to(first_loop)
    with pytest.raises(environment_manifest.PythonEnvironmentManifestError, match="cannot resolve"):
        environment_manifest.load_python_environment_manifest(first_loop)

    incomplete = tmp_path / "manifest.incomplete.json"
    incomplete.write_text("{}\n", encoding="utf-8")
    with pytest.raises(environment_manifest.PythonEnvironmentManifestError, match=r"\.incomplete"):
        environment_manifest.load_python_environment_manifest(incomplete)

    if hasattr(os, "mkfifo"):
        fifo = tmp_path / "manifest.fifo"
        os.mkfifo(fifo)
        with pytest.raises(environment_manifest.PythonEnvironmentManifestError, match="regular file"):
            environment_manifest.load_python_environment_manifest(fifo)

    oversized = tmp_path / "oversized.json"
    oversized.write_bytes(b"{} ")
    monkeypatch.setattr(environment_manifest, "MAX_MANIFEST_BYTES", 2)
    with pytest.raises(environment_manifest.PythonEnvironmentManifestError, match="exceeds"):
        environment_manifest.load_python_environment_manifest(oversized)


def test_manifest_loader_handles_short_regular_file_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "manifest.json"
    path.write_text('{"value":"complete"}\n', encoding="utf-8")
    real_read = os.read
    monkeypatch.setattr(
        environment_manifest.os,
        "read",
        lambda descriptor, count: real_read(descriptor, min(count, 1)),
    )

    assert environment_manifest.load_python_environment_manifest(path) == {"value": "complete"}


def test_cli_create_and_verify_use_the_active_interpreter(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert environment_manifest.main(["create"]) == 0
    manifest = json.loads(capsys.readouterr().out)
    manifest_path = _write_manifest(tmp_path / "environment-manifest.json", manifest)

    assert environment_manifest.main(["verify", "--manifest", str(manifest_path)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "passed"
    assert report["interpreter_resolved_path"] == manifest["interpreter"]["resolved_path"]


def test_cli_returns_two_for_a_rejected_manifest(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest = tmp_path / "manifest.incomplete.json"
    manifest.write_text("{}\n", encoding="utf-8")

    assert environment_manifest.main(["verify", "--manifest", str(manifest)]) == 2
    assert ".incomplete" in capsys.readouterr().err
