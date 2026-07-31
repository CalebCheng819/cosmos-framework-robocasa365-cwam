# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""CPU tests for fail-closed DSW checkpoint publication."""

from __future__ import annotations

import hashlib
import json
import stat
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import publish_cwam_ema_checkpoint_dsw as publisher  # noqa: E402

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def test_formal_receipt_and_lock_are_pinned_to_persistent_cpfs() -> None:
    expected_parent = Path(
        "/mnt/workspace/cwam_cosmos3_robocasa365_eval/control/receipts"
    )
    assert publisher.DEFAULT_RECEIPT == (
        expected_parent / "ema-checkpoint-atomic-publication.json"
    )
    assert publisher.DEFAULT_LOCK == (
        expected_parent / ".ema-checkpoint-atomic-publication.lock"
    )


def _write_bytes(path: Path, payload: bytes) -> publisher.FileExpectation:
    path.write_bytes(payload)
    path.chmod(0o444)
    return publisher.FileExpectation(
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def _write_json(path: Path, value: object) -> publisher.FileExpectation:
    return _write_bytes(
        path,
        (json.dumps(value, indent=2, sort_keys=True) + "\n").encode(),
    )


def _identity_item(
    object_path: str,
    expected: publisher.FileExpectation,
) -> dict[str, object]:
    return {
        "object_path": object_path,
        "encoding": "identity",
        "raw_size_bytes": expected.size_bytes,
        "raw_sha256": expected.sha256,
        "transfer_size_bytes": expected.size_bytes,
        "transfer_sha256": expected.sha256,
        "sha256_sidecar_path": f"{object_path}.sha256",
        "sha256_sidecar_size_bytes": 1,
        "sha256_sidecar_sha256": "a" * 64,
        "remote_size_verified": True,
    }


def _fake_closure(
    tmp_path: Path,
) -> tuple[publisher.PublicationContract, Path, Path, Path, Path]:
    tmp_path.chmod(0o755)
    stage = tmp_path / ".formal.incomplete-transfer-20260731"
    final = tmp_path / "formal"
    model = stage / "model"
    model.mkdir(parents=True)
    model_payloads = {
        ".metadata": b"metadata",
        **{f"__{index}_0.distcp": f"shard-{index}".encode() for index in range(8)},
    }
    model_records = []
    for name in sorted(model_payloads):
        expected = _write_bytes(model / name, model_payloads[name])
        model_records.append(
            {
                "relative_path": name,
                "size_bytes": expected.size_bytes,
                "sha256": expected.sha256,
            }
        )
    tree_sha = publisher._records_digest(model_records)
    source_model_path = "/h/formal/model"

    root_files: dict[str, publisher.FileExpectation] = {}
    root_files["config.yaml"] = _write_bytes(stage / "config.yaml", b"config\n")
    h_manifest = {
        "schema_version": 1,
        "kind": "cwam_serving_checkpoint_manifest",
        "checkpoint_path": source_model_path,
        "files": model_records,
        "tree_sha256": tree_sha,
    }
    root_files["h-serving-checkpoint-manifest.json"] = _write_json(
        stage / "h-serving-checkpoint-manifest.json",
        h_manifest,
    )

    key_records = [
        {
            "key": "net_ema.a",
            "source_dtype": "torch.float32",
            "target_dtype": "torch.bfloat16",
            "chunk_count": 1,
            "chunks": [{}],
        },
        {
            "key": "net_ema.b",
            "source_dtype": "torch.float32",
            "target_dtype": "torch.bfloat16",
            "chunk_count": 2,
            "chunks": [{}, {}],
        },
    ]
    key_manifest = {
        "schema_version": 1,
        "kind": "cwam_serving_ema_bf16_key_manifest",
        "selection": "net_ema.* only; key names preserved",
        "key_count": 2,
        "target_model_path": source_model_path,
        "keys": key_records,
    }
    root_files["ema-bf16-key-manifest.json"] = _write_json(
        stage / "ema-bf16-key-manifest.json",
        key_manifest,
    )
    validation = {
        "schema_version": 1,
        "kind": "cwam_serving_ema_bf16_validation",
        "status": "passed",
        "full_readback": {
            "status": "passed",
            "verified_key_count": 2,
            "verified_chunk_count": 3,
            "verified_shard_count": 8,
            "checkpoint_tree_sha256": tree_sha,
            "every_serialized_chunk_sha256_recorded": True,
            "every_bf16_tensor_sha256_matched": True,
            "every_target_file_sha256_recorded": True,
        },
        "custom_load_planner_samples": [
            {
                "dtype": "torch.bfloat16",
                "explicit_target_matches_source_ema_cast": True,
                "custom_load_planner_matches_source_ema_cast": True,
            }
            for _ in range(3)
        ],
    }
    root_files["ema-bf16-validation.json"] = _write_json(
        stage / "ema-bf16-validation.json",
        validation,
    )
    export = {
        "schema_version": 1,
        "kind": "cwam_serving_ema_bf16_export_receipt",
        "status": "succeeded",
        "selection": {
            "included_prefix": "net_ema.",
            "excluded_prefix": "net.",
            "key_names_preserved": True,
            "source_dtype": "torch.float32",
            "target_dtype": "torch.bfloat16",
            "key_count": 2,
            "chunk_count": 3,
        },
        "target": {
            "shard_count": 8,
            "checkpoint_tree_sha256": tree_sha,
            "metadata_sha256": str(model_records[0]["sha256"]),
            "serving_manifest_sha256": root_files[
                "h-serving-checkpoint-manifest.json"
            ].sha256,
            "key_manifest_sha256": root_files[
                "ema-bf16-key-manifest.json"
            ].sha256,
            "validation_sha256": root_files["ema-bf16-validation.json"].sha256,
        },
        "source": {"config_sha256": root_files["config.yaml"].sha256},
        "publication": {"final_model_path": source_model_path},
    }
    root_files["ema-bf16-export-receipt.json"] = _write_json(
        stage / "ema-bf16-export-receipt.json",
        export,
    )
    source_receipt = {
        "schema_version": 1,
        "kind": "cwam_source_dcp_full_verification",
        "status": "passed",
        "source": {
            "file_count": 9,
            "total_bytes": 91_061_947_649,
            "full_file_sha256_readback": True,
        },
        "derived_target_binding": {
            "export_receipt_sha256": root_files[
                "ema-bf16-export-receipt.json"
            ].sha256,
            "target_manifest_sha256": root_files[
                "h-serving-checkpoint-manifest.json"
            ].sha256,
            "target_tree_sha256": tree_sha,
        },
    }
    root_files["source-dcp-full-verification.json"] = _write_json(
        stage / "source-dcp-full-verification.json",
        source_receipt,
    )

    exporter_sha = hashlib.sha256(b"exporter").hexdigest()
    h_sums = {
        "config.yaml": root_files["config.yaml"].sha256,
        "ema-bf16-export-receipt.json": root_files[
            "ema-bf16-export-receipt.json"
        ].sha256,
        "ema-bf16-key-manifest.json": root_files[
            "ema-bf16-key-manifest.json"
        ].sha256,
        "ema-bf16-validation.json": root_files["ema-bf16-validation.json"].sha256,
        "exporter-source.py": exporter_sha,
        **{
            f"model/{record['relative_path']}": str(record["sha256"])
            for record in model_records
        },
        "serving-checkpoint-manifest.json": root_files[
            "h-serving-checkpoint-manifest.json"
        ].sha256,
    }
    sums_payload = "".join(
        f"{digest}  {relative}\n" for relative, digest in sorted(h_sums.items())
    ).encode()
    root_files["H-SHA256SUMS"] = _write_bytes(
        stage / "H-SHA256SUMS",
        sums_payload,
    )

    identity_expectations = {
        "SHA256SUMS": root_files["H-SHA256SUMS"],
        "config.yaml": root_files["config.yaml"],
        "ema-bf16-export-receipt.json": root_files[
            "ema-bf16-export-receipt.json"
        ],
        "ema-bf16-key-manifest.json": root_files[
            "ema-bf16-key-manifest.json"
        ],
        "ema-bf16-validation.json": root_files["ema-bf16-validation.json"],
        "exporter-source.py": publisher.FileExpectation(55_882, exporter_sha),
        "serving-checkpoint-manifest.json": root_files[
            "h-serving-checkpoint-manifest.json"
        ],
        "model/.metadata": publisher.FileExpectation(
            int(model_records[0]["size_bytes"]),
            str(model_records[0]["sha256"]),
        ),
        "source-dcp-full-verification.json": root_files[
            "source-dcp-full-verification.json"
        ],
    }
    transfer_items = [
        _identity_item(path, expected)
        for path, expected in identity_expectations.items()
    ]
    by_name = {str(record["relative_path"]): record for record in model_records}
    for index in range(8):
        name = f"__{index}_0.distcp"
        record = by_name[name]
        transfer_items.append(
            {
                "object_path": f"model/{name}.gz",
                "encoding": "gzip-pigz-1-n",
                "raw_object_path": f"model/{name}",
                "raw_size_bytes": record["size_bytes"],
                "raw_sha256": record["sha256"],
                "transfer_size_bytes": 1,
                "transfer_sha256": hashlib.sha256(name.encode()).hexdigest(),
                "sha256_sidecar_path": f"model/{name}.gz.sha256",
                "sha256_sidecar_size_bytes": 1,
                "sha256_sidecar_sha256": "b" * 64,
                "remote_size_verified": True,
            }
        )
    transfer_items.sort(key=lambda item: str(item["object_path"]))
    transfer = {
        "schema_version": 1,
        "kind": "cwam_ema_serving_dcp_ceph_transfer_manifest",
        "status": "uploaded_remote_size_verified",
        "item_count": 17,
        "items": transfer_items,
        "bindings": {
            "export_receipt_sha256": root_files[
                "ema-bf16-export-receipt.json"
            ].sha256,
            "source_dcp_full_verification_sha256": root_files[
                "source-dcp-full-verification.json"
            ].sha256,
            "target_tree_sha256": tree_sha,
        },
        "verification_boundary": {
            "local_transfer_sha256_computed_before_upload": True,
            "remote_object_set_and_size_verified": True,
            "remote_content_sha256_readback": False,
            "consumer_must_verify_transfer_sha256_after_download": True,
            "consumer_must_verify_raw_sha256_after_decompression": True,
        },
    }
    root_files["ema-serving-dcp-transfer-manifest.json"] = _write_json(
        stage / "ema-serving-dcp-transfer-manifest.json",
        transfer,
    )
    stage.chmod(0o755)
    model.chmod(0o755)
    contract = publisher.PublicationContract(
        stage_root=stage,
        final_root=final,
        source_model_path=source_model_path,
        tree_sha256=tree_sha,
        root_files=root_files,
        model_files=tuple(model_records),
        h_sums=h_sums,
        ema_key_count=2,
        ema_chunk_count=3,
    )
    receipts = tmp_path / "receipts"
    receipts.mkdir(mode=0o755)
    receipts.chmod(0o755)
    return (
        contract,
        stage,
        final,
        receipts / "publication.json",
        receipts / "publication.lock",
    )


def test_audit_only_is_read_only(tmp_path: Path) -> None:
    contract, stage, final, receipt, lock = _fake_closure(tmp_path)

    result = publisher.publish_checkpoint(
        stage_root=stage,
        final_root=final,
        receipt_output=receipt,
        lock_path=lock,
        publish=False,
        contract=contract,
    )

    assert result["status"] == "audit_passed"
    assert result["prepublication"]["model"]["tree_sha256"] == contract.tree_sha256
    assert stage.is_dir()
    assert not final.exists()
    assert not receipt.exists()
    assert not lock.exists()


def test_publish_uses_no_replace_rename_and_writes_receipt(tmp_path: Path) -> None:
    contract, stage, final, receipt, lock = _fake_closure(tmp_path)

    result = publisher.publish_checkpoint(
        stage_root=stage,
        final_root=final,
        receipt_output=receipt,
        lock_path=lock,
        publish=True,
        contract=contract,
    )

    assert result["status"] == "published"
    assert result["publication"]["primitive"] == "renameat2(RENAME_NOREPLACE)"
    assert not stage.exists()
    assert final.is_dir()
    assert receipt.is_file()
    assert stat.S_IMODE(receipt.stat().st_mode) == 0o444
    persisted = json.loads(receipt.read_text(encoding="utf-8"))
    assert persisted["status"] == "published"
    assert (
        persisted["postpublication"]["model"]["tree_sha256"]
        == contract.tree_sha256
    )
    assert persisted["publication"]["receipt_required_for_success"] is True
    assert (
        persisted["publication"]["rollback_on_postpublication_failure"]
        is True
    )


def test_extra_top_level_file_fails_closed(tmp_path: Path) -> None:
    contract, stage, final, receipt, lock = _fake_closure(tmp_path)
    (stage / "unexpected").write_bytes(b"x")

    with pytest.raises(
        publisher.CheckpointPublicationError,
        match="top-level set mismatch",
    ):
        publisher.publish_checkpoint(
            stage_root=stage,
            final_root=final,
            receipt_output=receipt,
            lock_path=lock,
            publish=False,
            contract=contract,
        )
    assert stage.is_dir()
    assert not final.exists()


@pytest.mark.parametrize("mutation", ["hash", "mode"])
def test_model_hash_or_mode_failure_never_publishes(
    tmp_path: Path,
    mutation: str,
) -> None:
    contract, stage, final, receipt, lock = _fake_closure(tmp_path)
    shard = stage / "model" / "__0_0.distcp"
    if mutation == "hash":
        shard.chmod(0o644)
        shard.write_bytes(b"wrong")
        shard.chmod(0o444)
    else:
        shard.chmod(0o644)

    with pytest.raises(publisher.CheckpointPublicationError):
        publisher.publish_checkpoint(
            stage_root=stage,
            final_root=final,
            receipt_output=receipt,
            lock_path=lock,
            publish=True,
            contract=contract,
        )
    assert stage.is_dir()
    assert not final.exists()
    assert not receipt.exists()


def test_existing_final_is_never_replaced(tmp_path: Path) -> None:
    contract, stage, final, receipt, lock = _fake_closure(tmp_path)
    final.mkdir()
    sentinel = final / "sentinel"
    sentinel.write_bytes(b"keep")

    with pytest.raises(
        publisher.CheckpointPublicationError,
        match="hidden stage and final checkpoint both exist",
    ):
        publisher.publish_checkpoint(
            stage_root=stage,
            final_root=final,
            receipt_output=receipt,
            lock_path=lock,
            publish=True,
            contract=contract,
        )
    assert sentinel.read_bytes() == b"keep"
    assert stage.is_dir()


def test_tamper_after_rename_rolls_final_back_to_hidden_stage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    contract, stage, final, receipt, lock = _fake_closure(tmp_path)
    real_rename_noreplace = publisher._rename_noreplace
    injected = False

    def rename_then_tamper(source: Path, destination: Path) -> None:
        nonlocal injected
        real_rename_noreplace(source, destination)
        if destination == final and not injected:
            injected = True
            shard = final / "model" / "__0_0.distcp"
            shard.chmod(0o644)
            shard.write_bytes(b"tampered-after-preaudit")
            shard.chmod(0o444)

    monkeypatch.setattr(publisher, "_rename_noreplace", rename_then_tamper)
    with pytest.raises(
        publisher.CheckpointPublicationError,
        match="content mismatch",
    ):
        publisher.publish_checkpoint(
            stage_root=stage,
            final_root=final,
            receipt_output=receipt,
            lock_path=lock,
            publish=True,
            contract=contract,
        )

    assert injected is True
    assert stage.is_dir()
    assert not final.exists()
    assert not receipt.exists()


def test_receipt_failure_rolls_final_back_to_hidden_stage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    contract, stage, final, receipt, lock = _fake_closure(tmp_path)

    def fail_receipt(_path: Path, _value: object) -> dict:
        raise publisher.CheckpointPublicationError("injected receipt failure")

    monkeypatch.setattr(publisher, "_atomic_write_read_only", fail_receipt)
    with pytest.raises(
        publisher.CheckpointPublicationError,
        match="injected receipt failure",
    ):
        publisher.publish_checkpoint(
            stage_root=stage,
            final_root=final,
            receipt_output=receipt,
            lock_path=lock,
            publish=True,
            contract=contract,
        )

    assert stage.is_dir()
    assert not final.exists()
    assert not receipt.exists()
    assert (
        publisher.audit_checkpoint_root(stage, contract=contract)["model"][
            "tree_sha256"
        ]
        == contract.tree_sha256
    )


def test_valid_interrupted_final_without_receipt_is_recovered(
    tmp_path: Path,
) -> None:
    contract, stage, final, receipt, lock = _fake_closure(tmp_path)
    publisher._rename_noreplace(stage, final)

    result = publisher.publish_checkpoint(
        stage_root=stage,
        final_root=final,
        receipt_output=receipt,
        lock_path=lock,
        publish=True,
        contract=contract,
    )

    assert result["status"] == "published"
    assert result["recovered_uncommitted_final_before_publication"] is True
    assert not stage.exists()
    assert final.is_dir()
    assert receipt.is_file()


def test_receipt_readback_failure_removes_only_created_receipt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    tmp_path.chmod(0o755)
    receipt = tmp_path / "receipt.json"

    def fail_readback(
        path: Path,
        *,
        label: str,
        expected: publisher.FileExpectation | None = None,
        capture: bool = False,
    ) -> tuple[dict, bytes | None]:
        del expected, capture
        raise publisher.CheckpointPublicationError(
            f"{label} content mismatch: {path}"
        )

    monkeypatch.setattr(publisher, "_stable_file", fail_readback)
    with pytest.raises(
        publisher.CheckpointPublicationError,
        match="content mismatch",
    ):
        publisher._atomic_write_read_only(receipt, {"status": "published"})
    assert not receipt.exists()
