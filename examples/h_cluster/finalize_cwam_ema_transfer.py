#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Finalize and publish the auditable Ceph transfer manifest for EMA DCP."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import stat
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

IDENTITY_PATHS = (
    "SHA256SUMS",
    "config.yaml",
    "ema-bf16-export-receipt.json",
    "ema-bf16-key-manifest.json",
    "ema-bf16-validation.json",
    "exporter-source.py",
    "serving-checkpoint-manifest.json",
    "model/.metadata",
    "source-dcp-full-verification.json",
)


class TransferManifestError(RuntimeError):
    """Raised when local or remote transfer state violates the contract."""


def _sha256(path: Path) -> tuple[int, str]:
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    after = path.stat()
    if (
        not stat.S_ISREG(before.st_mode)
        or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
    ):
        raise TransferManifestError(f"file changed while hashing: {path}")
    return before.st_size, digest.hexdigest()


def _load_worker_records(stage: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for path in sorted(stage.glob("upload-worker-*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            value = json.loads(line)
            if value.get("event") != "object_uploaded":
                raise TransferManifestError(f"unexpected worker record in {path}: {value!r}")
            object_path = value.get("object")
            if not isinstance(object_path, str) or object_path in records:
                raise TransferManifestError(f"invalid or duplicate worker object: {object_path!r}")
            records[object_path] = value
    expected = {f"model/__{index}_0.distcp.gz" for index in range(8)}
    if set(records) != expected:
        raise TransferManifestError(
            f"worker object set mismatch: missing={sorted(expected - set(records))}, "
            f"extra={sorted(set(records) - expected)}"
        )
    return records


def _remote_files(prefix: str) -> dict[str, int]:
    process = subprocess.run(
        ["rclone", "lsjson", prefix, "--recursive", "--files-only"],
        check=True,
        capture_output=True,
        text=True,
    )
    values = json.loads(process.stdout)
    result: dict[str, int] = {}
    for value in values:
        path = value.get("Path")
        size = value.get("Size")
        if not isinstance(path, str) or type(size) is not int or path in result:
            raise TransferManifestError(f"invalid remote listing entry: {value!r}")
        result[path] = size
    return result


def _write_atomic_read_only(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise TransferManifestError(f"refusing to overwrite: {path}")
    temporary = path.with_name(f".{path.name}.incomplete-{uuid.uuid4().hex}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o444)
        os.replace(temporary, path)
    finally:
        os.close(descriptor)


def finalize(
    *,
    artifact: Path,
    source_receipt: Path,
    stage: Path,
    remote_prefix: str,
) -> dict[str, Any]:
    artifact = artifact.resolve(strict=True)
    source_receipt = source_receipt.resolve(strict=True)
    stage = stage.resolve(strict=True)
    worker_records = _load_worker_records(stage)
    items: list[dict[str, Any]] = []

    identity_sources = {
        relative: artifact / relative for relative in IDENTITY_PATHS if relative != "source-dcp-full-verification.json"
    }
    identity_sources["source-dcp-full-verification.json"] = source_receipt
    for relative in IDENTITY_PATHS:
        source_size, source_sha = _sha256(identity_sources[relative])
        transfer_size, transfer_sha = _sha256(stage / relative)
        if (source_size, source_sha) != (transfer_size, transfer_sha):
            raise TransferManifestError(f"identity transfer differs from source: {relative}")
        sidecar = stage / f"{relative}.sha256"
        sidecar_size, sidecar_sha = _sha256(sidecar)
        items.append(
            {
                "object_path": relative,
                "encoding": "identity",
                "raw_size_bytes": source_size,
                "raw_sha256": source_sha,
                "transfer_size_bytes": transfer_size,
                "transfer_sha256": transfer_sha,
                "sha256_sidecar_path": f"{relative}.sha256",
                "sha256_sidecar_size_bytes": sidecar_size,
                "sha256_sidecar_sha256": sidecar_sha,
            }
        )

    for object_path, worker_record in sorted(worker_records.items()):
        raw_path = artifact / str(worker_record["raw_object"])
        raw_size, raw_sha = _sha256(raw_path)
        transfer_size, transfer_sha = _sha256(stage / object_path)
        if (
            raw_size != worker_record["raw_size"]
            or raw_sha != worker_record["raw_sha256"]
            or transfer_size != worker_record["transfer_size"]
            or transfer_sha != worker_record["transfer_sha256"]
        ):
            raise TransferManifestError(f"worker ledger mismatch for {object_path}")
        sidecar = stage / f"{object_path}.sha256"
        sidecar_size, sidecar_sha = _sha256(sidecar)
        items.append(
            {
                "object_path": object_path,
                "encoding": "gzip-pigz-1-n",
                "raw_object_path": worker_record["raw_object"],
                "raw_size_bytes": raw_size,
                "raw_sha256": raw_sha,
                "transfer_size_bytes": transfer_size,
                "transfer_sha256": transfer_sha,
                "sha256_sidecar_path": f"{object_path}.sha256",
                "sha256_sidecar_size_bytes": sidecar_size,
                "sha256_sidecar_sha256": sidecar_sha,
            }
        )
    items.sort(key=lambda item: item["object_path"])

    remote_before = _remote_files(remote_prefix)
    expected_before: dict[str, int] = {}
    for item in items:
        expected_before[item["object_path"]] = item["transfer_size_bytes"]
        expected_before[item["sha256_sidecar_path"]] = item["sha256_sidecar_size_bytes"]
    if remote_before != expected_before:
        raise TransferManifestError(
            f"remote object/size set mismatch before manifest: expected={expected_before}, actual={remote_before}"
        )
    for item in items:
        item["remote_size_verified"] = True

    export_receipt_sha = _sha256(artifact / "ema-bf16-export-receipt.json")[1]
    source_receipt_sha = _sha256(source_receipt)[1]
    serving_manifest = json.loads((artifact / "serving-checkpoint-manifest.json").read_text())
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "kind": "cwam_ema_serving_dcp_ceph_transfer_manifest",
        "status": "uploaded_remote_size_verified",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "remote_prefix": remote_prefix,
        "item_count": len(items),
        "items": items,
        "bindings": {
            "export_receipt_sha256": export_receipt_sha,
            "source_dcp_full_verification_sha256": source_receipt_sha,
            "target_tree_sha256": serving_manifest["tree_sha256"],
        },
        "verification_boundary": {
            "local_transfer_sha256_computed_before_upload": True,
            "remote_object_set_and_size_verified": True,
            "remote_content_sha256_readback": False,
            "consumer_must_verify_transfer_sha256_after_download": True,
            "consumer_must_verify_raw_sha256_after_decompression": True,
        },
    }
    manifest_path = stage / "transfer-manifest.json"
    _write_atomic_read_only(
        manifest_path,
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(),
    )
    manifest_size, manifest_sha = _sha256(manifest_path)
    sidecar_path = stage / "transfer-manifest.json.sha256"
    _write_atomic_read_only(
        sidecar_path,
        f"{manifest_sha}  transfer-manifest.json\n".encode(),
    )
    subprocess.run(
        ["rclone", "copyto", str(manifest_path), f"{remote_prefix}/transfer-manifest.json"],
        check=True,
    )
    subprocess.run(
        ["rclone", "copyto", str(sidecar_path), f"{remote_prefix}/transfer-manifest.json.sha256"],
        check=True,
    )

    remote_after = _remote_files(remote_prefix)
    expected_after = {
        **expected_before,
        "transfer-manifest.json": manifest_size,
        "transfer-manifest.json.sha256": sidecar_path.stat().st_size,
    }
    if remote_after != expected_after:
        raise TransferManifestError("remote object/size set mismatch after manifest publication")
    result = {
        "status": "uploaded_remote_size_verified",
        "remote_prefix": remote_prefix,
        "object_count": len(remote_after),
        "payload_count": len(items),
        "transfer_manifest_path": str(manifest_path),
        "transfer_manifest_sha256": manifest_sha,
        "target_tree_sha256": serving_manifest["tree_sha256"],
    }
    print(json.dumps(result, sort_keys=True))
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--source-receipt", type=Path, required=True)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--remote-prefix", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        finalize(
            artifact=args.artifact,
            source_receipt=args.source_receipt,
            stage=args.stage,
            remote_prefix=args.remote_prefix,
        )
    except (
        OSError,
        ValueError,
        KeyError,
        json.JSONDecodeError,
        subprocess.CalledProcessError,
        TransferManifestError,
    ) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
