#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Fully rehash a source DCP and issue a cross-bound, read-only receipt."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import stat
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from examples.h_cluster import cwam_serving_checkpoint_manifest as serving_manifest


class SourceDcpVerificationError(RuntimeError):
    """Raised when full source verification or receipt publication fails."""


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _hash_file(path: Path) -> tuple[int, str]:
    try:
        return serving_manifest._hash_regular_file(path)
    except serving_manifest.ServingCheckpointManifestError as error:
        raise SourceDcpVerificationError(str(error)) from error


def _sha256(path: Path) -> str:
    return _hash_file(path)[1]


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return serving_manifest.load_serving_checkpoint_manifest(path)
    except serving_manifest.ServingCheckpointManifestError as error:
        raise SourceDcpVerificationError(str(error)) from error


def _require_sha(actual: str, expected: str, label: str) -> None:
    if actual != expected:
        raise SourceDcpVerificationError(f"{label} SHA-256 mismatch: expected={expected}, actual={actual}")


def verify_and_issue_receipt(
    *,
    source_model: Path,
    origin_manifest_path: Path,
    export_receipt_path: Path,
    target_manifest_path: Path,
    output_path: Path,
    expected_origin_manifest_sha256: str,
    expected_origin_tree_sha256: str,
    expected_export_receipt_sha256: str,
    expected_target_manifest_sha256: str,
    expected_target_tree_sha256: str,
) -> dict[str, Any]:
    source_model = source_model.resolve(strict=True)
    origin_manifest_path = origin_manifest_path.resolve(strict=True)
    export_receipt_path = export_receipt_path.resolve(strict=True)
    target_manifest_path = target_manifest_path.resolve(strict=True)
    output_path = Path(os.path.abspath(output_path))
    if output_path.exists() or output_path.is_symlink():
        raise SourceDcpVerificationError(f"output already exists; refusing to overwrite: {output_path}")
    if output_path.parent.resolve(strict=True) != output_path.parent:
        raise SourceDcpVerificationError(f"output parent must be canonical: {output_path.parent}")

    _require_sha(_sha256(origin_manifest_path), expected_origin_manifest_sha256, "origin manifest")
    manifest = _load_json(origin_manifest_path)
    try:
        expected_files, origin_tree = serving_manifest._validated_manifest(
            manifest,
            checkpoint_path=str(source_model),
        )
    except serving_manifest.ServingCheckpointManifestError as error:
        raise SourceDcpVerificationError(str(error)) from error
    _require_sha(origin_tree, expected_origin_tree_sha256, "origin tree")

    actual_names = sorted(path.name for path in source_model.iterdir())
    expected_names = [str(record["relative_path"]) for record in expected_files]
    if actual_names != expected_names:
        raise SourceDcpVerificationError(
            f"source file set mismatch: expected={expected_names}, actual={actual_names}"
        )
    if any("/" in name for name in expected_names):
        raise SourceDcpVerificationError("source DCP verification expects one flat directory")

    total_bytes = sum(int(record["size_bytes"]) for record in expected_files)
    completed_bytes = 0
    started_at = _utc_now()
    started = time.monotonic()
    actual_files: list[dict[str, str | int]] = []
    print(
        json.dumps(
            {
                "event": "source_dcp_full_verification_start",
                "source_model": str(source_model),
                "file_count": len(expected_files),
                "total_bytes": total_bytes,
                "started_at": started_at,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    for index, expected in enumerate(expected_files, start=1):
        relative = str(expected["relative_path"])
        size, sha256 = _hash_file(source_model / relative)
        if size != expected["size_bytes"] or sha256 != expected["sha256"]:
            raise SourceDcpVerificationError(
                f"live source mismatch for {relative}: "
                f"expected_size={expected['size_bytes']}, actual_size={size}, "
                f"expected_sha256={expected['sha256']}, actual_sha256={sha256}"
            )
        actual_files.append(
            {
                "relative_path": relative,
                "size_bytes": size,
                "sha256": sha256,
            }
        )
        completed_bytes += size
        elapsed = max(time.monotonic() - started, 1e-9)
        rate = completed_bytes / elapsed
        print(
            json.dumps(
                {
                    "event": "source_dcp_file_verified",
                    "relative_path": relative,
                    "file_index": index,
                    "file_count": len(expected_files),
                    "completed_bytes": completed_bytes,
                    "total_bytes": total_bytes,
                    "percent": round(100.0 * completed_bytes / total_bytes, 3),
                    "bytes_per_second": round(rate, 3),
                    "eta_seconds": round((total_bytes - completed_bytes) / rate, 3),
                    "at": _utc_now(),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    actual_tree = serving_manifest._tree_sha256(actual_files)
    _require_sha(actual_tree, expected_origin_tree_sha256, "live source tree")
    _require_sha(_sha256(export_receipt_path), expected_export_receipt_sha256, "export receipt")
    export_receipt = _load_json(export_receipt_path)
    if (
        export_receipt.get("kind") != "cwam_serving_ema_bf16_export_receipt"
        or export_receipt.get("status") != "succeeded"
    ):
        raise SourceDcpVerificationError("export receipt kind/status mismatch")
    source_binding = export_receipt.get("source")
    target_binding = export_receipt.get("target")
    if not isinstance(source_binding, dict) or not isinstance(target_binding, dict):
        raise SourceDcpVerificationError("export receipt lacks source/target bindings")
    if (
        source_binding.get("model_path") != str(source_model)
        or source_binding.get("origin_manifest_sha256") != expected_origin_manifest_sha256
        or source_binding.get("origin_tree_sha256") != expected_origin_tree_sha256
    ):
        raise SourceDcpVerificationError("export receipt source binding mismatch")
    if target_binding.get("checkpoint_tree_sha256") != expected_target_tree_sha256:
        raise SourceDcpVerificationError("export receipt target-tree binding mismatch")

    _require_sha(_sha256(target_manifest_path), expected_target_manifest_sha256, "target manifest")
    target_manifest = _load_json(target_manifest_path)
    if target_manifest.get("tree_sha256") != expected_target_tree_sha256:
        raise SourceDcpVerificationError("target manifest tree binding mismatch")

    script_path = Path(__file__).resolve(strict=True)
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "kind": "cwam_source_dcp_full_verification",
        "status": "passed",
        "started_at": started_at,
        "completed_at": _utc_now(),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "source": {
            "model_path": str(source_model),
            "file_count": len(actual_files),
            "total_bytes": total_bytes,
            "files": actual_files,
            "full_file_sha256_readback": True,
            "origin_manifest_path": str(origin_manifest_path),
            "origin_manifest_sha256": expected_origin_manifest_sha256,
            "origin_tree_sha256": actual_tree,
        },
        "derived_target_binding": {
            "export_receipt_path": str(export_receipt_path),
            "export_receipt_sha256": expected_export_receipt_sha256,
            "target_manifest_path": str(target_manifest_path),
            "target_manifest_sha256": expected_target_manifest_sha256,
            "target_tree_sha256": expected_target_tree_sha256,
        },
        "verifier": {
            "source_path": str(script_path),
            "sha256": _sha256(script_path),
            "python": sys.version,
        },
    }
    payload = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode()
    temporary = output_path.with_name(f".{output_path.name}.incomplete-{uuid.uuid4().hex}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o444)
        os.replace(temporary, output_path)
        parent_descriptor = os.open(output_path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    finally:
        os.close(descriptor)
    metadata = output_path.lstat()
    if not stat.S_ISREG(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) != 0o444:
        raise SourceDcpVerificationError(f"published receipt is not a mode-0444 regular file: {output_path}")
    result = {
        "status": "passed",
        "receipt_path": str(output_path),
        "receipt_sha256": _sha256(output_path),
        "source_tree_sha256": actual_tree,
        "target_tree_sha256": expected_target_tree_sha256,
        "file_count": len(actual_files),
        "total_bytes": total_bytes,
        "elapsed_seconds": receipt["elapsed_seconds"],
    }
    print(json.dumps({"event": "source_dcp_full_verification_complete", **result}, sort_keys=True), flush=True)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--origin-manifest", type=Path, required=True)
    parser.add_argument("--export-receipt", type=Path, required=True)
    parser.add_argument("--target-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-origin-manifest-sha256", required=True)
    parser.add_argument("--expected-origin-tree-sha256", required=True)
    parser.add_argument("--expected-export-receipt-sha256", required=True)
    parser.add_argument("--expected-target-manifest-sha256", required=True)
    parser.add_argument("--expected-target-tree-sha256", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        verify_and_issue_receipt(
            source_model=args.source_model,
            origin_manifest_path=args.origin_manifest,
            export_receipt_path=args.export_receipt,
            target_manifest_path=args.target_manifest,
            output_path=args.output,
            expected_origin_manifest_sha256=args.expected_origin_manifest_sha256,
            expected_origin_tree_sha256=args.expected_origin_tree_sha256,
            expected_export_receipt_sha256=args.expected_export_receipt_sha256,
            expected_target_manifest_sha256=args.expected_target_manifest_sha256,
            expected_target_tree_sha256=args.expected_target_tree_sha256,
        )
    except SourceDcpVerificationError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
