# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""CPU tests for DSW checkpoint relocation and immutable input receipts."""

from __future__ import annotations

import hashlib
import json
import stat
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
H_HELPERS = HERE.parent / "h_cluster"
sys.path.insert(0, str(H_HELPERS))
sys.path.insert(0, str(HERE))

import cwam_dsw_input_provenance as provenance  # noqa: E402
import cwam_serving_checkpoint_manifest as checkpoint_manifest  # noqa: E402

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def _write_json(path: Path, value: object, *, mode: int = 0o444) -> str:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    path.chmod(mode)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_immutable_receipt_retries_only_strict_cpfs_identity_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    real_stable_file_descriptor = provenance.stable_file_descriptor
    attempts = 0

    def transient_cpfs_descriptor(
        path: Path,
        *,
        label: str = "artifact",
        expected_mode: int | None = None,
    ) -> dict:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise provenance.DswInputProvenanceError(
                f"{label} changed while opening: {path}"
            )
        return real_stable_file_descriptor(
            path,
            label=label,
            expected_mode=expected_mode,
        )

    monkeypatch.setattr(
        provenance,
        "stable_file_descriptor",
        transient_cpfs_descriptor,
    )
    monkeypatch.setattr(provenance.time, "sleep", lambda _seconds: None)

    output = tmp_path / "receipt.json"
    descriptor = provenance._atomic_write_immutable(output, {"status": "passed"})

    assert attempts == 3
    assert descriptor["path"] == str(output)
    assert stat.S_IMODE(output.stat().st_mode) == 0o444


def test_immutable_receipt_does_not_retry_nontransient_readback_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    attempts = 0

    def rejected_descriptor(
        _path: Path,
        *,
        label: str = "artifact",
        expected_mode: int | None = None,
    ) -> dict:
        del label, expected_mode
        nonlocal attempts
        attempts += 1
        raise provenance.DswInputProvenanceError("receipt mode mismatch")

    monkeypatch.setattr(provenance, "stable_file_descriptor", rejected_descriptor)
    with pytest.raises(
        provenance.DswInputProvenanceError,
        match="receipt mode mismatch",
    ):
        provenance._atomic_write_immutable(
            tmp_path / "rejected-receipt.json",
            {"status": "passed"},
        )
    assert attempts == 1


def _target_model(tmp_path: Path) -> tuple[Path, dict, str]:
    model = tmp_path / "downloaded-serving-artifact" / "model"
    model.mkdir(parents=True)
    (model / ".metadata").write_bytes(b"metadata")
    for index in range(8):
        (model / f"__{index}_0.distcp").write_bytes(f"shard-{index}".encode())
    live = checkpoint_manifest.create_serving_checkpoint_manifest(model)
    h_checkpoint_path = "/mnt/shared-storage-gpfs2/source/artifact/model"
    h_manifest = {**live, "checkpoint_path": h_checkpoint_path}
    h_manifest_path = tmp_path / "h-serving-checkpoint-manifest.json"
    h_manifest_sha = _write_json(h_manifest_path, h_manifest)
    return model, h_manifest, h_manifest_sha


def _ema_receipts(
    tmp_path: Path,
    *,
    h_manifest: dict,
    h_manifest_sha: str,
    config_sha: str,
) -> dict[str, tuple[Path, str]]:
    h_model_path = h_manifest["checkpoint_path"]
    keys = []
    for index in range(809):
        chunks = [{"chunk_index": chunk} for chunk in range(8)]
        keys.append(
            {
                "key": f"net_ema.layer_{index:04d}",
                "source_dtype": "torch.float32",
                "target_dtype": "torch.bfloat16",
                "chunk_count": len(chunks),
                "chunks": chunks,
            }
        )
    key_manifest = {
        "schema_version": 1,
        "kind": "cwam_serving_ema_bf16_key_manifest",
        "selection": "net_ema.* only; key names preserved",
        "key_count": 809,
        "target_model_path": h_model_path,
        "keys": keys,
    }
    key_path = tmp_path / "ema-bf16-key-manifest.json"
    key_sha = _write_json(key_path, key_manifest)

    samples = [
        {
            "dtype": "torch.bfloat16",
            "explicit_target_matches_source_ema_cast": True,
            "custom_load_planner_matches_source_ema_cast": True,
        }
        for _ in range(3)
    ]
    validation = {
        "schema_version": 1,
        "kind": "cwam_serving_ema_bf16_validation",
        "status": "passed",
        "target_model_path": h_model_path,
        "full_readback": {
            "status": "passed",
            "verified_key_count": 809,
            "verified_chunk_count": 6472,
            "verified_shard_count": 8,
            "every_serialized_chunk_sha256_recorded": True,
            "every_bf16_tensor_sha256_matched": True,
            "every_target_file_sha256_recorded": True,
            "checkpoint_tree_sha256": h_manifest["tree_sha256"],
        },
        "custom_load_planner_samples": samples,
    }
    validation_path = tmp_path / "ema-bf16-validation.json"
    validation_sha = _write_json(validation_path, validation)

    source_records = []
    source_names = [".metadata", *(f"__{index}_0.distcp" for index in range(8))]
    for index, name in enumerate(source_names):
        source_records.append(
            {
                "relative_path": name,
                "size_bytes": 91_061_947_649 if index == 0 else 0,
                "sha256": hashlib.sha256(name.encode()).hexdigest(),
            }
        )
    source_records.sort(key=lambda record: record["relative_path"])
    source_tree_sha = provenance._canonical_records_digest(source_records)
    source_manifest_sha = "1" * 64
    source_model_path = "/mnt/shared-storage-gpfs2/source/original/model"
    export_receipt = {
        "schema_version": 1,
        "kind": "cwam_serving_ema_bf16_export_receipt",
        "status": "succeeded",
        "selection": {
            "included_prefix": "net_ema.",
            "excluded_prefix": "net.",
            "key_names_preserved": True,
            "source_dtype": "torch.float32",
            "target_dtype": "torch.bfloat16",
            "key_count": 809,
            "chunk_count": 6472,
        },
        "target": {
            "shard_count": 8,
            "checkpoint_tree_sha256": h_manifest["tree_sha256"],
            "serving_manifest_sha256": h_manifest_sha,
            "key_manifest_sha256": key_sha,
            "validation_sha256": validation_sha,
        },
        "source": {
            "model_path": source_model_path,
            "origin_manifest_sha256": source_manifest_sha,
            "origin_tree_sha256": source_tree_sha,
            "config_sha256": config_sha,
        },
        "publication": {"final_model_path": h_model_path},
    }
    export_path = tmp_path / "ema-bf16-export-receipt.json"
    export_sha = _write_json(export_path, export_receipt)

    source_receipt = {
        "schema_version": 1,
        "kind": "cwam_source_dcp_full_verification",
        "status": "passed",
        "started_at": "2026-07-31T00:00:00Z",
        "completed_at": "2026-07-31T00:01:00Z",
        "elapsed_seconds": 60.0,
        "source": {
            "model_path": source_model_path,
            "file_count": 9,
            "total_bytes": 91_061_947_649,
            "files": source_records,
            "full_file_sha256_readback": True,
            "origin_manifest_path": "/mnt/shared-storage-gpfs2/source/origin.json",
            "origin_manifest_sha256": source_manifest_sha,
            "origin_tree_sha256": source_tree_sha,
        },
        "derived_target_binding": {
            "export_receipt_path": "/mnt/shared-storage-gpfs2/source/export.json",
            "export_receipt_sha256": export_sha,
            "target_manifest_path": "/mnt/shared-storage-gpfs2/source/target.json",
            "target_manifest_sha256": h_manifest_sha,
            "target_tree_sha256": h_manifest["tree_sha256"],
        },
        "verifier": {
            "source_path": "/source/verify.py",
            "sha256": "2" * 64,
            "python": "3.test",
        },
    }
    source_path = tmp_path / "source-dcp-full-verification.json"
    source_sha = _write_json(source_path, source_receipt)
    return {
        "export": (export_path, export_sha),
        "keys": (key_path, key_sha),
        "validation": (validation_path, validation_sha),
        "source": (source_path, source_sha),
    }


def test_checkpoint_relocation_rebinds_path_and_preserves_exact_tree(tmp_path: Path) -> None:
    model, h_manifest, h_manifest_sha = _target_model(tmp_path)
    config = tmp_path / "config.yaml"
    config.write_bytes(b"config")
    config_sha = hashlib.sha256(config.read_bytes()).hexdigest()
    receipts = _ema_receipts(
        tmp_path,
        h_manifest=h_manifest,
        h_manifest_sha=h_manifest_sha,
        config_sha=config_sha,
    )
    destination_manifest = tmp_path / "control" / "destination-manifest.json"
    relocation_receipt = tmp_path / "control" / "relocation.json"
    destination_manifest.parent.mkdir()

    receipt = provenance.relocate_checkpoint(
        checkpoint_path=model,
        h_manifest_path=tmp_path / "h-serving-checkpoint-manifest.json",
        expected_h_manifest_sha256=h_manifest_sha,
        expected_target_tree_sha256=h_manifest["tree_sha256"],
        export_receipt_path=receipts["export"][0],
        expected_export_receipt_sha256=receipts["export"][1],
        key_manifest_path=receipts["keys"][0],
        expected_key_manifest_sha256=receipts["keys"][1],
        validation_path=receipts["validation"][0],
        expected_validation_sha256=receipts["validation"][1],
        source_dcp_receipt_path=receipts["source"][0],
        expected_source_dcp_receipt_sha256=receipts["source"][1],
        training_config_path=config,
        expected_training_config_sha256=config_sha,
        destination_manifest_output=destination_manifest,
        relocation_receipt_output=relocation_receipt,
    )

    destination = json.loads(destination_manifest.read_text(encoding="utf-8"))
    assert destination["checkpoint_path"] == str(model)
    assert destination["checkpoint_path"] != h_manifest["checkpoint_path"]
    assert destination["files"] == h_manifest["files"]
    assert destination["tree_sha256"] == h_manifest["tree_sha256"]
    assert receipt["ema_selection"]["selected_tree"] == "net_ema"
    assert receipt["ema_selection"]["key_count"] == 809
    assert stat.S_IMODE(destination_manifest.stat().st_mode) == 0o444
    assert stat.S_IMODE(relocation_receipt.stat().st_mode) == 0o444


def test_source_snapshot_verifier_rehashes_every_file_and_binds_head(
    tmp_path: Path,
) -> None:
    root = tmp_path / "snapshots" / "cosmos"
    root.mkdir(parents=True)
    payload = root / "module.py"
    payload.write_bytes(b"VALUE = 1\n")
    commit = "a" * 40
    manifest = {
        "schema_version": 1,
        "source_head": commit,
        "source_dirty": False,
        "file_count": 1,
        "files": [
            {
                "path": "module.py",
                "bytes": payload.stat().st_size,
                "sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
            }
        ],
    }
    manifest_path = root / "SNAPSHOT_MANIFEST.json"
    manifest_sha = _write_json(manifest_path, manifest)
    payload.chmod(0o444)
    root.chmod(0o555)
    output_parent = tmp_path / "control"
    output_parent.mkdir()

    receipt = provenance.verify_source_snapshot(
        phase="pre",
        snapshot_root=root,
        manifest_path=manifest_path,
        expected_manifest_sha256=manifest_sha,
        expected_source_commit=commit,
        output=output_parent / "source-pre.json",
    )
    assert receipt["file_count"] == 1
    assert receipt["source_commit"] == commit

    payload.chmod(0o644)
    payload.write_bytes(b"VALUE = 2\n")
    payload.chmod(0o444)
    with pytest.raises(provenance.DswInputProvenanceError, match="checksum mismatch"):
        provenance.verify_source_snapshot(
            phase="post",
            snapshot_root=root,
            manifest_path=manifest_path,
            expected_manifest_sha256=manifest_sha,
            expected_source_commit=commit,
            output=output_parent / "source-post.json",
        )


def _fake_environment(tmp_path: Path, name: str) -> tuple[Path, Path]:
    environment = tmp_path / name
    executable = environment / "bin" / "python"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"")
    return environment, executable


def _fake_torch(
    *,
    version: str,
    cuda_version: str,
    gpu_count: int,
) -> ModuleType:
    torch = ModuleType("torch")
    torch.__version__ = version
    torch.version = SimpleNamespace(cuda=cuda_version)
    properties = SimpleNamespace(
        name="NVIDIA GeForce RTX 4090",
        major=8,
        minor=9,
    )
    torch.cuda = SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: gpu_count,
        get_device_properties=lambda _index: properties,
    )
    return torch


def test_cosmos_native_environment_marker_binds_env_spec_commit_and_live_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    environment, executable = _fake_environment(tmp_path, "cosmos-env")
    source_commit = "9" * 40
    marker = environment / ".cwam_env_ready.json"
    marker_sha = _write_json(
        marker,
        {
            "schema_version": 1,
            "status": "ready",
            "python": "3.11.14",
            "torch": "2.10.0+cu128",
            "torch_cuda": "12.8",
            "cuda_available": True,
            "packages": {"pydantic": "2.test", "pyyaml": "6.test"},
            "source_head": source_commit,
            "pyproject_sha256": "1" * 64,
            "uv_lock_sha256": "2" * 64,
            "uv": "0.11.28",
            "default_index": "https://pypi.org/simple",
            "cudart_alias": "libcudart.so.12",
        },
    )
    fake_torch = _fake_torch(
        version="2.10.0+cu128",
        cuda_version="12.8",
        gpu_count=4,
    )
    fake_pydantic = ModuleType("pydantic")
    fake_pydantic.__version__ = "2.test"
    fake_yaml = ModuleType("yaml")
    fake_yaml.__version__ = "6.test"
    modules = {
        "torch": fake_torch,
        "pydantic": fake_pydantic,
        "yaml": fake_yaml,
    }
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(provenance.importlib, "import_module", modules.__getitem__)
    monkeypatch.setattr(
        provenance.importlib.metadata,
        "version",
        {"pydantic": "2.test", "pyyaml": "6.test"}.__getitem__,
    )
    monkeypatch.setattr(provenance.platform, "python_version", lambda: "3.11.14")
    monkeypatch.setattr(provenance.sys, "prefix", str(environment))
    monkeypatch.setattr(provenance.sys, "executable", str(executable))

    receipt = provenance.verify_environment(
        phase="pre",
        environment=environment,
        environment_type="cosmos",
        installation_mode="native_exact_rebuild",
        marker_path=marker,
        expected_marker_sha256=marker_sha,
        expected_source_commit=source_commit,
        expected_robocasa_commit=None,
        expected_robosuite_commit=None,
        import_probes=("torch", "pydantic", "yaml"),
        expect_cuda_gpus=4,
        output=tmp_path / "cosmos-env-receipt.json",
    )

    assert receipt["installation_mode"] == "native_exact_rebuild"
    assert receipt["live_runtime"]["cuda"]["device_count"] == 4
    assert receipt["validation"]["live_python_matches_marker"] is True
    assert receipt["validation"]["live_marker_package_versions_match"] is True
    assert "live_distribution_inventory_matches_marker" not in receipt["validation"]


def test_robocasa_native_environment_marker_matches_live_pinned_packages(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    environment, executable = _fake_environment(tmp_path, "robocasa-env")
    robocasa_commit = "3" * 40
    robosuite_commit = "4" * 40
    marker = environment / ".cwam_env_ready.json"
    marker_sha = _write_json(
        marker,
        {
            "schema_version": 1,
            "status": "ready",
            "python": "3.11.14",
            "torch": "2.7.1+cu128",
            "torch_cuda": "12.8",
            "cuda_available": True,
            "mujoco": "3.3.1",
            "robocasa": "1.0.1",
            "robosuite": "1.5.2",
            "distribution_count": 123,
            "distributions_sha256": "5" * 64,
            "origin_environment_manifest_sha256": "6" * 64,
            "origin_requirements_sha256": "7" * 64,
            "relocated_requirements_sha256": "8" * 64,
            "robocasa_commit": robocasa_commit,
            "robosuite_commit": robosuite_commit,
            "package_source_note": provenance._EXPECTED_ROBOCASA_PACKAGE_SOURCE_NOTE,
        },
    )
    fake_torch = _fake_torch(
        version="2.7.1+cu128",
        cuda_version="12.8",
        gpu_count=4,
    )
    modules: dict[str, ModuleType] = {}
    for name, version in (
        ("mujoco", "3.3.1"),
        ("robocasa", "1.0.1"),
        ("robosuite", "1.5.2"),
    ):
        module = ModuleType(name)
        module.__version__ = version
        modules[name] = module
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(provenance.importlib, "import_module", modules.__getitem__)
    monkeypatch.setattr(
        provenance,
        "_live_distribution_summary",
        lambda: (123, "5" * 64),
    )
    monkeypatch.setattr(provenance.platform, "python_version", lambda: "3.11.14")
    monkeypatch.setattr(provenance.sys, "prefix", str(environment))
    monkeypatch.setattr(provenance.sys, "executable", str(executable))

    receipt = provenance.verify_environment(
        phase="pre",
        environment=environment,
        environment_type="robocasa",
        installation_mode="native_exact_rebuild",
        marker_path=marker,
        expected_marker_sha256=marker_sha,
        expected_source_commit=None,
        expected_robocasa_commit=robocasa_commit,
        expected_robosuite_commit=robosuite_commit,
        import_probes=("mujoco", "robocasa", "robosuite"),
        expect_cuda_gpus=0,
        output=tmp_path / "robocasa-env-receipt.json",
    )

    assert receipt["live_runtime"]["cuda"]["torch"] == "2.7.1+cu128"
    assert receipt["validation"]["live_framework_versions_match_marker"] is True
    assert receipt["validation"]["live_distribution_inventory_matches_marker"] is True
    assert "live_marker_package_versions_match" not in receipt["validation"]

    monkeypatch.setattr(
        provenance,
        "_live_distribution_summary",
        lambda: (122, "5" * 64),
    )
    with pytest.raises(
        provenance.DswInputProvenanceError,
        match="distribution inventory differs",
    ):
        provenance.verify_environment(
            phase="post",
            environment=environment,
            environment_type="robocasa",
            installation_mode="native_exact_rebuild",
            marker_path=marker,
            expected_marker_sha256=marker_sha,
            expected_source_commit=None,
            expected_robocasa_commit=robocasa_commit,
            expected_robosuite_commit=robosuite_commit,
            import_probes=("mujoco", "robocasa", "robosuite"),
            expect_cuda_gpus=0,
            output=tmp_path / "robocasa-drift-receipt.json",
        )
