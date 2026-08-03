#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Validate and query a formal CWAM EMA object-transfer manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
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
GZIP_PATHS = tuple(f"model/__{index}_0.distcp.gz" for index in range(8))
EXPECTED_OBJECT_PATHS = frozenset((*IDENTITY_PATHS, *GZIP_PATHS))
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_PATH_RE = re.compile(r"^[A-Za-z0-9._/-]+$")
_SAFE_REMOTE_RE = re.compile(r"^eailab-hdd2:eailab/chengjuntao/[A-Za-z0-9._/-]+$")


class TransferContractError(RuntimeError):
    """Raised when a transfer manifest violates the formal contract."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise TransferContractError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _stable_payload(path: Path) -> tuple[bytes, str]:
    supplied = Path(os.path.abspath(path.expanduser()))
    try:
        before = supplied.lstat()
    except FileNotFoundError:
        raise TransferContractError(f"manifest is missing: {supplied}") from None
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise TransferContractError("manifest must be a regular non-symlink file")
    if stat.S_IMODE(before.st_mode) != 0o444:
        raise TransferContractError(
            f"manifest mode mismatch: expected=0o444 actual={oct(stat.S_IMODE(before.st_mode))}"
        )
    payload = supplied.read_bytes()
    after = supplied.lstat()

    def identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
        return (
            value.st_dev,
            value.st_ino,
            value.st_mode,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )

    if identity(before) != identity(after) or len(payload) != before.st_size:
        raise TransferContractError("manifest changed while reading")
    return payload, hashlib.sha256(payload).hexdigest()


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise TransferContractError(f"{label} must be a lowercase SHA-256")
    return value


def _require_size(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise TransferContractError(f"{label} must be a non-negative integer")
    return value


def _require_safe_path(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or _SAFE_PATH_RE.fullmatch(value) is None
        or value.startswith("/")
        or "//" in value
        or any(part in ("", ".", "..") for part in value.split("/"))
    ):
        raise TransferContractError(f"{label} is not a safe relative path")
    return value


def _validate_item(raw: object) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise TransferContractError("transfer item must be an object")
    common = {
        "object_path",
        "encoding",
        "raw_size_bytes",
        "raw_sha256",
        "transfer_size_bytes",
        "transfer_sha256",
        "sha256_sidecar_path",
        "sha256_sidecar_size_bytes",
        "sha256_sidecar_sha256",
        "remote_size_verified",
    }
    encoding = raw.get("encoding")
    expected_keys = common if encoding == "identity" else common | {"raw_object_path"}
    if encoding not in ("identity", "gzip-pigz-1-n") or set(raw) != expected_keys:
        raise TransferContractError("transfer item encoding or key set is invalid")
    object_path = _require_safe_path(raw["object_path"], "object_path")
    sidecar_path = _require_safe_path(raw["sha256_sidecar_path"], "sidecar path")
    if sidecar_path != f"{object_path}.sha256":
        raise TransferContractError(f"sidecar path mismatch for {object_path}")
    raw_size = _require_size(raw["raw_size_bytes"], f"{object_path} raw size")
    transfer_size = _require_size(raw["transfer_size_bytes"], f"{object_path} transfer size")
    _require_size(raw["sha256_sidecar_size_bytes"], f"{object_path} sidecar size")
    raw_sha = _require_sha256(raw["raw_sha256"], f"{object_path} raw SHA-256")
    transfer_sha = _require_sha256(raw["transfer_sha256"], f"{object_path} transfer SHA-256")
    _require_sha256(raw["sha256_sidecar_sha256"], f"{object_path} sidecar SHA-256")
    if raw.get("remote_size_verified") is not True:
        raise TransferContractError(f"remote size was not verified for {object_path}")
    if encoding == "identity":
        if (raw_size, raw_sha) != (transfer_size, transfer_sha):
            raise TransferContractError(f"identity object differs from raw data: {object_path}")
    else:
        raw_object_path = _require_safe_path(raw["raw_object_path"], f"{object_path} raw object path")
        if object_path != f"{raw_object_path}.gz":
            raise TransferContractError(f"gzip object path mismatch: {object_path}")
    return dict(raw)


def load_transfer_manifest(
    path: Path,
    *,
    expected_sha256: str,
    expected_remote_prefix: str,
) -> dict[str, Any]:
    """Load and strictly validate one immutable transfer manifest."""

    _require_sha256(expected_sha256, "expected manifest SHA-256")
    if _SAFE_REMOTE_RE.fullmatch(expected_remote_prefix) is None:
        raise TransferContractError("remote prefix must stay below eailab-hdd2:eailab/chengjuntao/")
    payload, actual_sha = _stable_payload(path)
    if actual_sha != expected_sha256:
        raise TransferContractError(f"manifest SHA-256 mismatch: expected={expected_sha256} actual={actual_sha}")
    try:
        value = json.loads(payload, object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise TransferContractError(f"manifest is invalid JSON: {error}") from error
    if not isinstance(value, dict):
        raise TransferContractError("manifest root must be an object")
    required = {
        "schema_version",
        "kind",
        "status",
        "created_at",
        "remote_prefix",
        "item_count",
        "items",
        "bindings",
        "verification_boundary",
    }
    if set(value) != required:
        raise TransferContractError("manifest top-level key set is invalid")
    if (
        value["schema_version"] != SCHEMA_VERSION
        or value["kind"] != "cwam_ema_serving_dcp_ceph_transfer_manifest"
        or value["status"] != "uploaded_remote_size_verified"
        or value["remote_prefix"] != expected_remote_prefix
        or value["item_count"] != len(EXPECTED_OBJECT_PATHS)
    ):
        raise TransferContractError("manifest header does not match the formal contract")
    if not isinstance(value["created_at"], str) or not value["created_at"].endswith("Z"):
        raise TransferContractError("manifest created_at is invalid")
    raw_items = value["items"]
    if not isinstance(raw_items, list) or len(raw_items) != len(EXPECTED_OBJECT_PATHS):
        raise TransferContractError("manifest item count is invalid")
    items = [_validate_item(item) for item in raw_items]
    object_paths = [item["object_path"] for item in items]
    if object_paths != sorted(object_paths) or set(object_paths) != EXPECTED_OBJECT_PATHS:
        raise TransferContractError("manifest object path set or ordering is invalid")
    bindings = value["bindings"]
    if not isinstance(bindings, dict) or set(bindings) != {
        "export_receipt_sha256",
        "source_dcp_full_verification_sha256",
        "target_tree_sha256",
    }:
        raise TransferContractError("manifest bindings are invalid")
    for key, binding in bindings.items():
        _require_sha256(binding, f"binding {key}")
    expected_boundary = {
        "local_transfer_sha256_computed_before_upload": True,
        "remote_object_set_and_size_verified": True,
        "remote_content_sha256_readback": False,
        "consumer_must_verify_transfer_sha256_after_download": True,
        "consumer_must_verify_raw_sha256_after_decompression": True,
    }
    if value["verification_boundary"] != expected_boundary:
        raise TransferContractError("manifest verification boundary is invalid")
    value["items"] = items
    value["manifest_sha256"] = actual_sha
    return value


def item_by_object_path(manifest: Mapping[str, Any], object_path: str) -> Mapping[str, Any]:
    """Return one validated item from a validated manifest."""

    _require_safe_path(object_path, "requested object path")
    matches = [item for item in manifest["items"] if item["object_path"] == object_path]
    if len(matches) != 1:
        raise TransferContractError(f"requested object is absent or duplicated: {object_path}")
    return matches[0]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--expected-remote-prefix", required=True)
    parser.add_argument("--object-path")
    parser.add_argument("--output-format", choices=("json", "tsv"), default="json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest = load_transfer_manifest(
            args.manifest,
            expected_sha256=args.expected_sha256,
            expected_remote_prefix=args.expected_remote_prefix,
        )
        if args.object_path is None:
            result: Mapping[str, Any] = {
                "status": "passed",
                "manifest_sha256": manifest["manifest_sha256"],
                "remote_prefix": manifest["remote_prefix"],
                "item_count": manifest["item_count"],
                "target_tree_sha256": manifest["bindings"]["target_tree_sha256"],
            }
        else:
            result = item_by_object_path(manifest, args.object_path)
    except (OSError, TransferContractError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    if args.output_format == "tsv":
        if args.object_path is None:
            raise SystemExit("--output-format=tsv requires --object-path")
        print(
            "\t".join(
                str(result[key])
                for key in (
                    "object_path",
                    "encoding",
                    "raw_object_path" if result["encoding"] != "identity" else "object_path",
                    "raw_size_bytes",
                    "raw_sha256",
                    "transfer_size_bytes",
                    "transfer_sha256",
                )
            )
        )
    else:
        print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
