#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Fail-closed DSW publication of the formal CWAM EMA serving checkpoint.

The downloader writes only to a sibling hidden incomplete directory.  This
program audits the exact top-level closure and fully rehashes the model tree.
With ``--publish`` it reserves the canonical final directory with exclusive
``mkdir``, moves the exact top-level entries into that reservation, rehashes
the complete destination, removes the empty hidden stage, and finally writes
an immutable receipt outside the checkpoint.  CPFS exposes the reserved
directory while entries move, so the receipt--not directory visibility--is
the atomic commit point.

Without ``--publish`` the command is read-only and never creates the final
checkpoint directory.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
import re
import stat
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

EXPECTED_STAGE_ROOT = Path(
    "/mnt/workspace/cwam_cosmos3_robocasa365_eval/inputs/checkpoints/"
    ".R14-iter_000010000-serving-ema-bf16-v1.incomplete-transfer-20260731"
)
EXPECTED_FINAL_ROOT = Path(
    "/mnt/workspace/cwam_cosmos3_robocasa365_eval/inputs/checkpoints/"
    "R14-iter_000010000-serving-ema-bf16-v1"
)
DEFAULT_RECEIPT = Path(
    "/mnt/workspace/cwam_cosmos3_robocasa365_eval/control/receipts/"
    "ema-checkpoint-atomic-publication.json"
)
DEFAULT_LOCK = Path(
    "/mnt/workspace/cwam_cosmos3_robocasa365_eval/control/receipts/"
    ".ema-checkpoint-atomic-publication.lock"
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SUMS_LINE_RE = re.compile(r"^([0-9a-f]{64})  ([^\x00\r\n]+)$")


@dataclass(frozen=True)
class FileExpectation:
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class PublicationContract:
    stage_root: Path
    final_root: Path
    source_model_path: str
    tree_sha256: str
    root_files: Mapping[str, FileExpectation]
    model_files: tuple[Mapping[str, Any], ...]
    h_sums: Mapping[str, str]
    ema_key_count: int
    ema_chunk_count: int
    ema_source_logical_bytes: int | None = None
    ema_target_logical_bytes: int | None = None


MODEL_FILES: tuple[Mapping[str, Any], ...] = (
    {
        "relative_path": ".metadata",
        "size_bytes": 1_338_854,
        "sha256": "f1c9d43c8944fea444b61cd389f8f167cdd750975445e179d383af01cb7f0acd",
    },
    {
        "relative_path": "__0_0.distcp",
        "size_bytes": 3_794_555_281,
        "sha256": "3725441fc882ba66867347055529441ecc0288b0fa006f49544036adf97d6667",
    },
    {
        "relative_path": "__1_0.distcp",
        "size_bytes": 3_794_555_281,
        "sha256": "eeee9e22e47b1b6b084e955dd4f2fdb8476431fb6a52f275fd8a09c15d3b86c4",
    },
    {
        "relative_path": "__2_0.distcp",
        "size_bytes": 3_794_555_281,
        "sha256": "7b4a66e6f7b4ea6ccac73648bc3746471641ed96eb32fc86be60b9c57c4058ac",
    },
    {
        "relative_path": "__3_0.distcp",
        "size_bytes": 3_794_555_281,
        "sha256": "b4ebc4f7ea45d42749452203de7df42cfa6777da8c9f9e2130d2c31de6c27ee5",
    },
    {
        "relative_path": "__4_0.distcp",
        "size_bytes": 3_794_555_281,
        "sha256": "399d1cfd38eaa231df77a13e83ae972d44c50086af15d8b3247855754ae6e3ef",
    },
    {
        "relative_path": "__5_0.distcp",
        "size_bytes": 3_794_555_281,
        "sha256": "0651e25684f14f93ceded471f142d797508d041604588069cc697d173815a9c3",
    },
    {
        "relative_path": "__6_0.distcp",
        "size_bytes": 3_794_555_281,
        "sha256": "88e6ccc90b7052883c2a1fb83a432411c60d3bfdf86514fe9aa040ccb9a7d988",
    },
    {
        "relative_path": "__7_0.distcp",
        "size_bytes": 3_794_555_281,
        "sha256": "87faa9874fd8cd0cc1f1db7541ea6ca29747b358d189c4cf8700c5057c75b622",
    },
)

ROOT_FILES: Mapping[str, FileExpectation] = {
    "H-SHA256SUMS": FileExpectation(
        1_303,
        "2af2bbfa9b33bc931d3cde62e1135e4414dc8ad0ef58357091463d31b45c4226",
    ),
    "config.yaml": FileExpectation(
        20_578,
        "c8269c4f12c99ac98e71e85334933d9749cd66a494960d6afe6e91368134caaa",
    ),
    "h-serving-checkpoint-manifest.json": FileExpectation(
        1_820,
        "7bcb9494b9f00b93dbad7d2b533032d3c8086e134986ae70f1c03abb1858f500",
    ),
    "ema-bf16-export-receipt.json": FileExpectation(
        2_845,
        "26bc5f658aa71a2ac8018dabb53574c9917171bb69bff9e4091d5621961c4200",
    ),
    "ema-bf16-key-manifest.json": FileExpectation(
        5_394_417,
        "e2edc73c06b628bb9294b157af1f6d3be3ec11c626713a709c9f64667ca2d8fa",
    ),
    "ema-bf16-validation.json": FileExpectation(
        1_999,
        "887d7a962bda3f37f105d210945ec21bebed4becb0c42677b8c4ead256b673dd",
    ),
    "source-dcp-full-verification.json": FileExpectation(
        3_598,
        "1f9352ff4d97116d4861e2fec6dc5522a738a5ac58d5e15bf0025f3b1f45f646",
    ),
    "ema-serving-dcp-transfer-manifest.json": FileExpectation(
        11_154,
        "1f00b9d45f1d7ca2e6691fcd117d9da42c3444f9e517f5f3091362e1b3992f9b",
    ),
}

H_SUMS: Mapping[str, str] = {
    "config.yaml": ROOT_FILES["config.yaml"].sha256,
    "ema-bf16-export-receipt.json": ROOT_FILES["ema-bf16-export-receipt.json"].sha256,
    "ema-bf16-key-manifest.json": ROOT_FILES["ema-bf16-key-manifest.json"].sha256,
    "ema-bf16-validation.json": ROOT_FILES["ema-bf16-validation.json"].sha256,
    "exporter-source.py": "916c77a3373d201b0bcf79151857f039dbdc3c7231be37b8a8ec4c28cbf41400",
    **{
        f"model/{record['relative_path']}": str(record["sha256"])
        for record in MODEL_FILES
    },
    "serving-checkpoint-manifest.json": ROOT_FILES[
        "h-serving-checkpoint-manifest.json"
    ].sha256,
}

FORMAL_CONTRACT = PublicationContract(
    stage_root=EXPECTED_STAGE_ROOT,
    final_root=EXPECTED_FINAL_ROOT,
    source_model_path=(
        "/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chengjuntao/"
        "artifacts/cwam_e35/R14-iter_000010000-serving-ema-bf16-v1/model"
    ),
    tree_sha256="77d2a84a35e81438c87a6e31bedc09299b294fcbe9841f4e943b985bbcb6f846",
    root_files=ROOT_FILES,
    model_files=MODEL_FILES,
    h_sums=H_SUMS,
    ema_key_count=809,
    ema_chunk_count=6472,
    ema_source_logical_bytes=60_692_546_304,
    ema_target_logical_bytes=30_346_273_152,
)


class CheckpointPublicationError(RuntimeError):
    """Raised when the checkpoint closure or publication contract fails."""


class ReceiptCommitPointCrossedError(CheckpointPublicationError):
    """Raised after the immutable receipt name became externally visible."""


def _identity(metadata: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_mode),
        int(metadata.st_size),
        int(metadata.st_mtime_ns),
        int(metadata.st_ctime_ns),
    )


def _canonical_directory(path: Path, label: str, *, mode: int = 0o755) -> Path:
    supplied = Path(os.path.abspath(path.expanduser()))
    try:
        canonical = supplied.resolve(strict=True)
        metadata = supplied.lstat()
    except FileNotFoundError:
        raise CheckpointPublicationError(f"{label} does not exist: {supplied}") from None
    if (
        supplied != canonical
        or stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISDIR(metadata.st_mode)
    ):
        raise CheckpointPublicationError(
            f"{label} must be a canonical non-symlink directory: {supplied}"
        )
    if stat.S_IMODE(metadata.st_mode) != mode:
        raise CheckpointPublicationError(
            f"{label} mode mismatch: expected={oct(mode)}, "
            f"actual={oct(stat.S_IMODE(metadata.st_mode))}"
        )
    return canonical


def _stable_file(
    path: Path,
    *,
    label: str,
    expected: FileExpectation | None = None,
    capture: bool = False,
) -> tuple[dict[str, Any], bytes | None]:
    try:
        before = path.lstat()
    except FileNotFoundError:
        raise CheckpointPublicationError(f"{label} is missing: {path}") from None
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise CheckpointPublicationError(f"{label} must be a regular non-symlink file: {path}")
    if stat.S_IMODE(before.st_mode) != 0o444:
        raise CheckpointPublicationError(
            f"{label} mode mismatch: expected=0o444, actual={oct(stat.S_IMODE(before.st_mode))}"
        )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    digest = hashlib.sha256()
    size_bytes = 0
    payload = bytearray() if capture else None
    try:
        opened = os.fstat(descriptor)
        if _identity(opened) != _identity(before):
            raise CheckpointPublicationError(f"{label} changed while opening: {path}")
        while chunk := os.read(descriptor, 8 * 1024 * 1024):
            digest.update(chunk)
            size_bytes += len(chunk)
            if payload is not None:
                payload.extend(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    final = path.lstat()
    if not (_identity(before) == _identity(opened) == _identity(after) == _identity(final)):
        raise CheckpointPublicationError(f"{label} changed while hashing: {path}")
    observed = {
        "relative_path": path.name,
        "size_bytes": size_bytes,
        "sha256": digest.hexdigest(),
        "mode": "0444",
    }
    if expected is not None and (
        observed["size_bytes"] != expected.size_bytes
        or observed["sha256"] != expected.sha256
    ):
        raise CheckpointPublicationError(
            f"{label} content mismatch: expected=({expected.size_bytes}, {expected.sha256}), "
            f"actual=({observed['size_bytes']}, {observed['sha256']})"
        )
    return observed, bytes(payload) if payload is not None else None


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CheckpointPublicationError(f"JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _load_json(
    path: Path,
    *,
    label: str,
    expected: FileExpectation,
) -> tuple[dict[str, Any], dict[str, Any]]:
    descriptor, payload = _stable_file(path, label=label, expected=expected, capture=True)
    assert payload is not None
    try:
        value = json.loads(payload, object_pairs_hook=_reject_duplicate_json_keys)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise CheckpointPublicationError(f"{label} is invalid JSON: {error}") from error
    if not isinstance(value, dict):
        raise CheckpointPublicationError(f"{label} JSON root must be an object")
    return value, descriptor


def _records_digest(records: Sequence[Mapping[str, Any]]) -> str:
    payload = json.dumps(
        list(records),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _parse_h_sums(payload: bytes) -> dict[str, str]:
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeError as error:
        raise CheckpointPublicationError(f"H-SHA256SUMS is not UTF-8: {error}") from error
    result: dict[str, str] = {}
    for index, line in enumerate(lines, start=1):
        match = _SUMS_LINE_RE.fullmatch(line)
        if match is None:
            raise CheckpointPublicationError(f"H-SHA256SUMS line {index} is malformed")
        digest, relative = match.groups()
        if (
            relative.startswith("/")
            or "\\" in relative
            or any(part in ("", ".", "..") for part in relative.split("/"))
            or relative in result
        ):
            raise CheckpointPublicationError(
                f"H-SHA256SUMS line {index} has an unsafe or duplicate path"
            )
        result[relative] = digest
    return result


def _validate_h_manifest(
    value: Mapping[str, Any],
    *,
    contract: PublicationContract,
) -> None:
    if set(value) != {"schema_version", "kind", "checkpoint_path", "files", "tree_sha256"}:
        raise CheckpointPublicationError("H serving manifest has an unexpected key set")
    if (
        value["schema_version"] != 1
        or value["kind"] != "cwam_serving_checkpoint_manifest"
        or value["checkpoint_path"] != contract.source_model_path
        or value["tree_sha256"] != contract.tree_sha256
        or value["files"] != list(contract.model_files)
        or _records_digest(value["files"]) != contract.tree_sha256
    ):
        raise CheckpointPublicationError("H serving manifest contract mismatch")


def _validate_ema_jsons(
    *,
    export: Mapping[str, Any],
    keys: Mapping[str, Any],
    validation: Mapping[str, Any],
    source: Mapping[str, Any],
    contract: PublicationContract,
) -> None:
    root_files = contract.root_files
    selection = export.get("selection")
    target = export.get("target")
    export_source = export.get("source")
    publication = export.get("publication")
    expected_selection: dict[str, Any] = {
        "included_prefix": "net_ema.",
        "excluded_prefix": "net.",
        "key_names_preserved": True,
        "source_dtype": "torch.float32",
        "target_dtype": "torch.bfloat16",
        "key_count": contract.ema_key_count,
        "chunk_count": contract.ema_chunk_count,
    }
    if contract.ema_source_logical_bytes is not None:
        expected_selection["source_logical_bytes"] = contract.ema_source_logical_bytes
    if contract.ema_target_logical_bytes is not None:
        expected_selection["target_logical_bytes"] = contract.ema_target_logical_bytes
    if (
        export.get("schema_version") != 1
        or export.get("kind") != "cwam_serving_ema_bf16_export_receipt"
        or export.get("status") != "succeeded"
        or not isinstance(selection, Mapping)
        or any(selection.get(key) != expected for key, expected in expected_selection.items())
        or not isinstance(target, Mapping)
        or not isinstance(export_source, Mapping)
        or not isinstance(publication, Mapping)
        or target.get("shard_count") != 8
        or target.get("checkpoint_tree_sha256") != contract.tree_sha256
        or target.get("serving_manifest_sha256")
        != root_files["h-serving-checkpoint-manifest.json"].sha256
        or target.get("key_manifest_sha256")
        != root_files["ema-bf16-key-manifest.json"].sha256
        or target.get("validation_sha256")
        != root_files["ema-bf16-validation.json"].sha256
        or target.get("metadata_sha256") != contract.model_files[0]["sha256"]
        or export_source.get("config_sha256") != root_files["config.yaml"].sha256
        or publication.get("final_model_path") != contract.source_model_path
    ):
        raise CheckpointPublicationError("EMA export receipt binding mismatch")

    raw_keys = keys.get("keys")
    if (
        keys.get("schema_version") != 1
        or keys.get("kind") != "cwam_serving_ema_bf16_key_manifest"
        or keys.get("selection") != "net_ema.* only; key names preserved"
        or keys.get("key_count") != contract.ema_key_count
        or keys.get("target_model_path") != contract.source_model_path
        or not isinstance(raw_keys, list)
        or len(raw_keys) != contract.ema_key_count
    ):
        raise CheckpointPublicationError("EMA key manifest header mismatch")
    observed_names: list[str] = []
    observed_chunks = 0
    for index, record in enumerate(raw_keys):
        if not isinstance(record, Mapping):
            raise CheckpointPublicationError(f"EMA key record {index} is not an object")
        name = record.get("key")
        chunks = record.get("chunks")
        chunk_count = record.get("chunk_count")
        if (
            not isinstance(name, str)
            or not name.startswith("net_ema.")
            or record.get("source_dtype") != "torch.float32"
            or record.get("target_dtype") != "torch.bfloat16"
            or type(chunk_count) is not int
            or chunk_count < 1
            or not isinstance(chunks, list)
            or len(chunks) != chunk_count
        ):
            raise CheckpointPublicationError(f"EMA key record {index} is invalid")
        observed_names.append(name)
        observed_chunks += chunk_count
    if (
        observed_names != sorted(set(observed_names))
        or observed_chunks != contract.ema_chunk_count
    ):
        raise CheckpointPublicationError("EMA key ordering or chunk count mismatch")

    full = validation.get("full_readback")
    samples = validation.get("custom_load_planner_samples")
    if (
        validation.get("schema_version") != 1
        or validation.get("kind") != "cwam_serving_ema_bf16_validation"
        or validation.get("status") != "passed"
        or not isinstance(full, Mapping)
        or full.get("status") != "passed"
        or full.get("verified_key_count") != contract.ema_key_count
        or full.get("verified_chunk_count") != contract.ema_chunk_count
        or full.get("verified_shard_count") != 8
        or full.get("checkpoint_tree_sha256") != contract.tree_sha256
        or full.get("every_serialized_chunk_sha256_recorded") is not True
        or full.get("every_bf16_tensor_sha256_matched") is not True
        or full.get("every_target_file_sha256_recorded") is not True
        or not isinstance(samples, list)
        or len(samples) < 3
    ):
        raise CheckpointPublicationError("EMA full-readback validation mismatch")
    for sample in samples:
        if (
            not isinstance(sample, Mapping)
            or sample.get("dtype") != "torch.bfloat16"
            or sample.get("explicit_target_matches_source_ema_cast") is not True
            or sample.get("custom_load_planner_matches_source_ema_cast") is not True
        ):
            raise CheckpointPublicationError("EMA planner sample mismatch")

    derived = source.get("derived_target_binding")
    source_body = source.get("source")
    if (
        source.get("schema_version") != 1
        or source.get("kind") != "cwam_source_dcp_full_verification"
        or source.get("status") != "passed"
        or not isinstance(derived, Mapping)
        or not isinstance(source_body, Mapping)
        or source_body.get("file_count") != 9
        or source_body.get("total_bytes") != 91_061_947_649
        or source_body.get("full_file_sha256_readback") is not True
        or derived.get("export_receipt_sha256")
        != root_files["ema-bf16-export-receipt.json"].sha256
        or derived.get("target_manifest_sha256")
        != root_files["h-serving-checkpoint-manifest.json"].sha256
        or derived.get("target_tree_sha256") != contract.tree_sha256
    ):
        raise CheckpointPublicationError("source DCP full-verification binding mismatch")


def _validate_transfer_manifest(
    value: Mapping[str, Any],
    *,
    contract: PublicationContract,
) -> None:
    bindings = value.get("bindings")
    items = value.get("items")
    if (
        value.get("schema_version") != 1
        or value.get("kind") != "cwam_ema_serving_dcp_ceph_transfer_manifest"
        or value.get("status") != "uploaded_remote_size_verified"
        or value.get("item_count") != 17
        or not isinstance(items, list)
        or len(items) != 17
        or not isinstance(bindings, Mapping)
        or bindings.get("export_receipt_sha256")
        != contract.root_files["ema-bf16-export-receipt.json"].sha256
        or bindings.get("source_dcp_full_verification_sha256")
        != contract.root_files["source-dcp-full-verification.json"].sha256
        or bindings.get("target_tree_sha256") != contract.tree_sha256
    ):
        raise CheckpointPublicationError("Ceph transfer manifest header/binding mismatch")

    identity_expected: dict[str, FileExpectation] = {
        "SHA256SUMS": contract.root_files["H-SHA256SUMS"],
        "config.yaml": contract.root_files["config.yaml"],
        "ema-bf16-export-receipt.json": contract.root_files["ema-bf16-export-receipt.json"],
        "ema-bf16-key-manifest.json": contract.root_files["ema-bf16-key-manifest.json"],
        "ema-bf16-validation.json": contract.root_files["ema-bf16-validation.json"],
        "exporter-source.py": FileExpectation(55_882, contract.h_sums["exporter-source.py"]),
        "serving-checkpoint-manifest.json": contract.root_files[
            "h-serving-checkpoint-manifest.json"
        ],
        "model/.metadata": FileExpectation(
            int(contract.model_files[0]["size_bytes"]),
            str(contract.model_files[0]["sha256"]),
        ),
        "source-dcp-full-verification.json": contract.root_files[
            "source-dcp-full-verification.json"
        ],
    }
    model_by_path = {
        str(record["relative_path"]): record for record in contract.model_files
    }
    observed_paths: list[str] = []
    for item in items:
        if not isinstance(item, Mapping):
            raise CheckpointPublicationError("Ceph transfer manifest item is not an object")
        object_path = item.get("object_path")
        if not isinstance(object_path, str):
            raise CheckpointPublicationError("Ceph transfer manifest object_path is invalid")
        observed_paths.append(object_path)
        if item.get("remote_size_verified") is not True:
            raise CheckpointPublicationError(f"remote size was not verified for {object_path}")
        if object_path in identity_expected:
            expected = identity_expected[object_path]
            if (
                item.get("encoding") != "identity"
                or item.get("raw_size_bytes") != expected.size_bytes
                or item.get("raw_sha256") != expected.sha256
                or item.get("transfer_size_bytes") != expected.size_bytes
                or item.get("transfer_sha256") != expected.sha256
            ):
                raise CheckpointPublicationError(
                    f"identity transfer item mismatch for {object_path}"
                )
            continue
        match = re.fullmatch(r"model/(__[0-7]_0[.]distcp)[.]gz", object_path)
        if match is None:
            raise CheckpointPublicationError(
                f"unexpected Ceph transfer object: {object_path}"
            )
        raw_name = match.group(1)
        expected_record = model_by_path[raw_name]
        if (
            item.get("encoding") != "gzip-pigz-1-n"
            or item.get("raw_object_path") != f"model/{raw_name}"
            or item.get("raw_size_bytes") != expected_record["size_bytes"]
            or item.get("raw_sha256") != expected_record["sha256"]
            or type(item.get("transfer_size_bytes")) is not int
            or item["transfer_size_bytes"] <= 0
            or not isinstance(item.get("transfer_sha256"), str)
            or _SHA256_RE.fullmatch(item["transfer_sha256"]) is None
        ):
            raise CheckpointPublicationError(f"gzip transfer item mismatch for {object_path}")
    expected_paths = sorted(
        [*identity_expected, *(f"model/__{index}_0.distcp.gz" for index in range(8))]
    )
    if observed_paths != expected_paths:
        raise CheckpointPublicationError("Ceph transfer item ordering/set mismatch")

    boundary = value.get("verification_boundary")
    if (
        not isinstance(boundary, Mapping)
        or boundary.get("local_transfer_sha256_computed_before_upload") is not True
        or boundary.get("remote_object_set_and_size_verified") is not True
        or boundary.get("remote_content_sha256_readback") is not False
        or boundary.get("consumer_must_verify_transfer_sha256_after_download") is not True
        or boundary.get("consumer_must_verify_raw_sha256_after_decompression") is not True
    ):
        raise CheckpointPublicationError("Ceph transfer verification boundary mismatch")


def audit_checkpoint_root(
    root: Path,
    *,
    contract: PublicationContract = FORMAL_CONTRACT,
) -> dict[str, Any]:
    """Fully audit one hidden or published checkpoint closure."""

    root = _canonical_directory(root, "checkpoint artifact root")
    root_before = root.lstat()
    observed_names = {entry.name for entry in os.scandir(root)}
    expected_names = {*contract.root_files, "model"}
    if observed_names != expected_names:
        raise CheckpointPublicationError(
            "checkpoint top-level set mismatch: "
            f"missing={sorted(expected_names - observed_names)}, "
            f"extra={sorted(observed_names - expected_names)}"
        )

    descriptors: dict[str, dict[str, Any]] = {}
    json_values: dict[str, dict[str, Any]] = {}
    for name, expected in contract.root_files.items():
        path = root / name
        if name.endswith(".json"):
            value, descriptor = _load_json(
                path,
                label=f"checkpoint top-level {name}",
                expected=expected,
            )
            json_values[name] = value
        else:
            descriptor, payload = _stable_file(
                path,
                label=f"checkpoint top-level {name}",
                expected=expected,
                capture=name == "H-SHA256SUMS",
            )
            if name == "H-SHA256SUMS":
                assert payload is not None
                if _parse_h_sums(payload) != dict(contract.h_sums):
                    raise CheckpointPublicationError("H-SHA256SUMS inventory mismatch")
        descriptor["relative_path"] = name
        descriptors[name] = descriptor

    _validate_h_manifest(
        json_values["h-serving-checkpoint-manifest.json"],
        contract=contract,
    )
    _validate_ema_jsons(
        export=json_values["ema-bf16-export-receipt.json"],
        keys=json_values["ema-bf16-key-manifest.json"],
        validation=json_values["ema-bf16-validation.json"],
        source=json_values["source-dcp-full-verification.json"],
        contract=contract,
    )
    _validate_transfer_manifest(
        json_values["ema-serving-dcp-transfer-manifest.json"],
        contract=contract,
    )

    model = _canonical_directory(root / "model", "checkpoint model directory")
    model_before = model.lstat()
    model_names = {entry.name for entry in os.scandir(model)}
    expected_model_names = {
        str(record["relative_path"]) for record in contract.model_files
    }
    if model_names != expected_model_names:
        raise CheckpointPublicationError(
            "model file set mismatch: "
            f"missing={sorted(expected_model_names - model_names)}, "
            f"extra={sorted(model_names - expected_model_names)}"
        )
    model_records: list[dict[str, Any]] = []
    for expected_record in contract.model_files:
        name = str(expected_record["relative_path"])
        observed, _ = _stable_file(
            model / name,
            label=f"checkpoint model {name}",
            expected=FileExpectation(
                int(expected_record["size_bytes"]),
                str(expected_record["sha256"]),
            ),
        )
        model_records.append(
            {
                "relative_path": name,
                "size_bytes": observed["size_bytes"],
                "sha256": observed["sha256"],
            }
        )
    if (
        model_records != list(contract.model_files)
        or _records_digest(model_records) != contract.tree_sha256
    ):
        raise CheckpointPublicationError("model tree SHA-256 mismatch")

    if _identity(model_before) != _identity(model.lstat()):
        raise CheckpointPublicationError("model directory changed during audit")
    if _identity(root_before) != _identity(root.lstat()):
        raise CheckpointPublicationError("checkpoint artifact root changed during audit")
    return {
        "root": str(root),
        "top_level_entries": sorted(expected_names),
        "top_level_files": [descriptors[name] for name in sorted(descriptors)],
        "model": {
            "path": str(model),
            "file_count": len(model_records),
            "total_bytes": sum(int(record["size_bytes"]) for record in model_records),
            "files": model_records,
            "tree_sha256": contract.tree_sha256,
        },
        "validation": {
            "exact_top_level_allowlist": True,
            "all_top_level_files_mode_0444": True,
            "exact_model_allowlist": True,
            "all_model_files_mode_0444": True,
            "h_sha256sums_inventory_matches": True,
            "six_json_sha_and_semantics_match": True,
            "transfer_manifest_cross_bindings_match": True,
            "full_model_tree_rehashed": True,
        },
    }


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _inode_key(metadata: os.stat_result) -> tuple[int, int]:
    return int(metadata.st_dev), int(metadata.st_ino)


def _expected_top_level_names(contract: PublicationContract) -> set[str]:
    return {*contract.root_files, "model"}


def _assert_directory_inode(
    path: Path,
    expected_inode: tuple[int, int],
    *,
    label: str,
) -> os.stat_result:
    try:
        metadata = path.lstat()
        canonical = path.resolve(strict=True)
    except FileNotFoundError:
        raise CheckpointPublicationError(f"{label} disappeared: {path}") from None
    if (
        canonical != path
        or stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISDIR(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o755
        or _inode_key(metadata) != expected_inode
    ):
        raise CheckpointPublicationError(
            f"{label} inode/type/mode changed: {path}"
        )
    return metadata


def _directory_names(
    path: Path,
    expected_inode: tuple[int, int],
    *,
    label: str,
) -> set[str]:
    _assert_directory_inode(path, expected_inode, label=label)
    names = {entry.name for entry in os.scandir(path)}
    _assert_directory_inode(path, expected_inode, label=label)
    return names


def _mkdir_exclusive_directory(
    path: Path,
    *,
    parent: Path,
    label: str,
) -> tuple[Path, tuple[int, int]]:
    supplied = Path(os.path.abspath(path.expanduser()))
    if supplied.parent != parent:
        raise CheckpointPublicationError(f"{label} must be a direct child of {parent}")
    created_inode: tuple[int, int] | None = None
    try:
        previous_umask = os.umask(0)
        try:
            try:
                os.mkdir(supplied, 0o755)
            except FileExistsError:
                raise CheckpointPublicationError(
                    f"{label} already exists; refusing to replace it: {supplied}"
                ) from None
        finally:
            os.umask(previous_umask)
        if not os.path.lexists(supplied):
            raise CheckpointPublicationError(
                f"{label} exclusive mkdir returned without a directory"
            )
        metadata = supplied.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise CheckpointPublicationError(
                f"{label} exclusive mkdir did not create a directory"
            )
        created_inode = _inode_key(metadata)
        if stat.S_IMODE(metadata.st_mode) != 0o755:
            os.chmod(supplied, 0o755, follow_symlinks=False)
        _assert_directory_inode(supplied, created_inode, label=label)
        _fsync_directory(parent)
        _assert_directory_inode(supplied, created_inode, label=label)
        return supplied, created_inode
    except Exception:
        if created_inode is not None and os.path.lexists(supplied):
            current = supplied.lstat()
            if (
                _inode_key(current) == created_inode
                and stat.S_ISDIR(current.st_mode)
                and not stat.S_ISLNK(current.st_mode)
                and not any(os.scandir(supplied))
            ):
                os.rmdir(supplied)
                _fsync_directory(parent)
        raise


def _validate_expected_entry_metadata(
    metadata: os.stat_result,
    *,
    path: Path,
    name: str,
    contract: PublicationContract,
) -> None:
    if stat.S_ISLNK(metadata.st_mode):
        raise CheckpointPublicationError(
            f"transaction source entry must not be a symlink: {path}"
        )
    if name == "model":
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o755
        ):
            raise CheckpointPublicationError(
                f"transaction model entry type/mode mismatch: {path}"
            )
        return
    expected = contract.root_files[name]
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o444
        or metadata.st_size != expected.size_bytes
    ):
        raise CheckpointPublicationError(
            f"transaction top-level entry type/mode/size mismatch: {path}"
        )


def _move_entry_to_absent_destination(
    *,
    source_root: Path,
    source_inode: tuple[int, int],
    destination_root: Path,
    destination_inode: tuple[int, int],
    name: str,
    contract: PublicationContract,
    source_label: str,
    destination_label: str,
) -> None:
    _assert_directory_inode(source_root, source_inode, label=source_label)
    _assert_directory_inode(
        destination_root,
        destination_inode,
        label=destination_label,
    )
    source = source_root / name
    destination = destination_root / name
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    source_descriptor = os.open(source_root, flags)
    destination_descriptor = os.open(destination_root, flags)
    try:
        if _inode_key(os.fstat(source_descriptor)) != source_inode:
            raise CheckpointPublicationError(
                f"{source_label} descriptor inode mismatch"
            )
        if _inode_key(os.fstat(destination_descriptor)) != destination_inode:
            raise CheckpointPublicationError(
                f"{destination_label} descriptor inode mismatch"
            )
        try:
            source_metadata = os.stat(
                name,
                dir_fd=source_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            raise CheckpointPublicationError(
                f"transaction source entry disappeared: {source}"
            ) from None
        _validate_expected_entry_metadata(
            source_metadata,
            path=source,
            name=name,
            contract=contract,
        )
        source_entry_inode = _inode_key(source_metadata)
        try:
            os.stat(
                name,
                dir_fd=destination_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            pass
        else:
            raise CheckpointPublicationError(
                f"transaction destination entry already exists: {destination}"
            )
        os.rename(
            name,
            name,
            src_dir_fd=source_descriptor,
            dst_dir_fd=destination_descriptor,
        )
        try:
            os.stat(
                name,
                dir_fd=source_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            pass
        else:
            raise CheckpointPublicationError(
                f"transaction source entry remained after rename: {source}"
            )
        destination_metadata = os.stat(
            name,
            dir_fd=destination_descriptor,
            follow_symlinks=False,
        )
        _validate_expected_entry_metadata(
            destination_metadata,
            path=destination,
            name=name,
            contract=contract,
        )
        if _inode_key(destination_metadata) != source_entry_inode:
            raise CheckpointPublicationError(
                f"transaction entry inode changed during rename: {name}"
            )
        os.fsync(source_descriptor)
        os.fsync(destination_descriptor)
    finally:
        os.close(destination_descriptor)
        os.close(source_descriptor)
    _assert_directory_inode(source_root, source_inode, label=source_label)
    _assert_directory_inode(
        destination_root,
        destination_inode,
        label=destination_label,
    )


def _move_entries(
    *,
    source_root: Path,
    source_inode: tuple[int, int],
    destination_root: Path,
    destination_inode: tuple[int, int],
    names: Sequence[str],
    contract: PublicationContract,
    source_label: str,
    destination_label: str,
) -> None:
    for name in sorted(names):
        _move_entry_to_absent_destination(
            source_root=source_root,
            source_inode=source_inode,
            destination_root=destination_root,
            destination_inode=destination_inode,
            name=name,
            contract=contract,
            source_label=source_label,
            destination_label=destination_label,
        )


def _remove_empty_directory(
    path: Path,
    expected_inode: tuple[int, int],
    *,
    parent: Path,
    label: str,
) -> None:
    if _directory_names(path, expected_inode, label=label):
        raise CheckpointPublicationError(f"{label} is not empty: {path}")
    os.rmdir(path)
    _fsync_directory(parent)
    if os.path.lexists(path):
        raise CheckpointPublicationError(f"{label} still exists after rmdir: {path}")


def _optional_transaction_directory(
    path: Path,
    *,
    label: str,
) -> tuple[Path, tuple[int, int], set[str]] | None:
    if not os.path.lexists(path):
        return None
    canonical = _canonical_directory(path, label)
    inode = _inode_key(canonical.lstat())
    return canonical, inode, _directory_names(canonical, inode, label=label)


def _recover_uncommitted_transaction_to_stage(
    *,
    stage_root: Path,
    final_root: Path,
    parent: Path,
    contract: PublicationContract,
) -> tuple[Path, dict[str, Any], bool]:
    expected_names = _expected_top_level_names(contract)
    stage_state = _optional_transaction_directory(
        stage_root,
        label="hidden checkpoint transaction stage",
    )
    final_state = _optional_transaction_directory(
        final_root,
        label="uncommitted checkpoint final reservation",
    )
    if stage_state is None and final_state is None:
        raise CheckpointPublicationError(
            "hidden stage and uncommitted final reservation are both absent"
        )
    stage_names = set() if stage_state is None else stage_state[2]
    final_names = set() if final_state is None else final_state[2]
    extra = (stage_names | final_names) - expected_names
    overlap = stage_names & final_names
    missing = expected_names - (stage_names | final_names)
    if extra:
        raise CheckpointPublicationError(
            f"uncommitted checkpoint transaction has extra entries: {sorted(extra)}"
        )
    if overlap:
        raise CheckpointPublicationError(
            "uncommitted checkpoint transaction has overlapping entries: "
            f"{sorted(overlap)}"
        )
    if missing:
        raise CheckpointPublicationError(
            f"uncommitted checkpoint transaction is missing entries: {sorted(missing)}"
        )
    if final_state is None:
        assert stage_state is not None
        stage = stage_state[0]
        return stage, audit_checkpoint_root(stage, contract=contract), False

    final, final_inode, _ = final_state
    if stage_state is None:
        stage, stage_inode = _mkdir_exclusive_directory(
            stage_root,
            parent=parent,
            label="recovered hidden checkpoint stage",
        )
    else:
        stage, stage_inode, _ = stage_state
    if (
        stage.stat().st_dev != final.stat().st_dev
        or stage.stat().st_dev != parent.stat().st_dev
    ):
        raise CheckpointPublicationError(
            "checkpoint transaction directories must share one filesystem"
        )
    _move_entries(
        source_root=final,
        source_inode=final_inode,
        destination_root=stage,
        destination_inode=stage_inode,
        names=tuple(final_names),
        contract=contract,
        source_label="uncommitted checkpoint final reservation",
        destination_label="recovered hidden checkpoint stage",
    )
    if _directory_names(
        stage,
        stage_inode,
        label="recovered hidden checkpoint stage",
    ) != expected_names:
        raise CheckpointPublicationError(
            "recovered hidden checkpoint stage closure is incomplete"
        )
    if _directory_names(
        final,
        final_inode,
        label="uncommitted checkpoint final reservation",
    ):
        raise CheckpointPublicationError(
            "uncommitted checkpoint final reservation is not empty after recovery"
        )
    audit = audit_checkpoint_root(stage, contract=contract)
    _assert_directory_inode(
        stage,
        stage_inode,
        label="recovered hidden checkpoint stage",
    )
    _assert_directory_inode(
        final,
        final_inode,
        label="uncommitted checkpoint final reservation",
    )
    _remove_empty_directory(
        final,
        final_inode,
        parent=parent,
        label="uncommitted checkpoint final reservation",
    )
    _assert_directory_inode(
        stage,
        stage_inode,
        label="recovered hidden checkpoint stage",
    )
    return stage, audit, True


def _atomic_write_read_only(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    supplied = Path(os.path.abspath(path.expanduser()))
    parent = _canonical_directory(supplied.parent, "publication receipt parent")
    if supplied.parent != parent:
        raise CheckpointPublicationError("publication receipt parent must be canonical")
    if os.path.lexists(supplied):
        raise CheckpointPublicationError(f"refusing to overwrite publication receipt: {supplied}")
    payload = (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{supplied.name}.", suffix=".tmp", dir=parent
    )
    temporary = Path(temporary_name)
    committed = False
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fchmod(stream.fileno(), 0o444)
            os.fsync(stream.fileno())
            opened_metadata = os.fstat(stream.fileno())
            temporary_metadata = temporary.lstat()
            if (
                _identity(opened_metadata) != _identity(temporary_metadata)
                or not stat.S_ISREG(opened_metadata.st_mode)
                or stat.S_IMODE(opened_metadata.st_mode) != 0o444
            ):
                raise CheckpointPublicationError(
                    "publication receipt temporary inode/type/mode mismatch"
                )
            os.link(temporary, supplied, follow_symlinks=False)
            committed = True
            published_metadata = supplied.lstat()
            if (
                published_metadata.st_dev,
                published_metadata.st_ino,
            ) != (
                temporary_metadata.st_dev,
                temporary_metadata.st_ino,
            ):
                raise CheckpointPublicationError(
                    "publication receipt hard link inode mismatch"
                )
            # Finish both directory mutations, then persist the shared inode's
            # post-link metadata while the original descriptor is still pinned.
            temporary.unlink()
            os.fsync(stream.fileno())
        _fsync_directory(parent)
    except Exception as error:
        try:
            temporary.unlink(missing_ok=True)
        except Exception as cleanup_error:
            if not committed:
                raise CheckpointPublicationError(
                    "publication receipt preparation failed and temporary "
                    f"cleanup also failed: {cleanup_error}"
                ) from error
        if committed:
            raise ReceiptCommitPointCrossedError(
                "publication receipt commit point was crossed but durability "
                f"follow-up failed: {error}"
            ) from error
        raise
    expected = FileExpectation(len(payload), hashlib.sha256(payload).hexdigest())
    deadline = time.monotonic() + 10.0
    delay_seconds = 0.05
    try:
        while True:
            try:
                observed, _ = _stable_file(
                    supplied,
                    label="publication receipt",
                    expected=expected,
                )
                return observed
            except CheckpointPublicationError as error:
                message = str(error)
                if (
                    "publication receipt changed while opening:" not in message
                    and "publication receipt changed while hashing:" not in message
                ):
                    raise
                if time.monotonic() >= deadline:
                    raise CheckpointPublicationError(
                        "publication receipt metadata did not stabilize within "
                        "10 seconds"
                    ) from error
                time.sleep(delay_seconds)
                delay_seconds = min(delay_seconds * 2.0, 0.5)
    except Exception as error:
        raise ReceiptCommitPointCrossedError(
            "publication receipt commit point was crossed but immutable "
            f"readback failed: {error}"
        ) from error


def publish_checkpoint(
    *,
    stage_root: Path,
    final_root: Path,
    receipt_output: Path,
    lock_path: Path,
    publish: bool,
    contract: PublicationContract = FORMAL_CONTRACT,
) -> dict[str, Any]:
    """Audit the hidden closure and optionally commit a CPFS transaction."""

    stage_supplied = Path(os.path.abspath(stage_root.expanduser()))
    final_supplied = Path(os.path.abspath(final_root.expanduser()))
    if stage_supplied != contract.stage_root or final_supplied != contract.final_root:
        raise CheckpointPublicationError(
            "stage/final path differs from the pinned formal publication contract"
        )
    if stage_supplied.parent != final_supplied.parent:
        raise CheckpointPublicationError("stage and final roots must be siblings")
    parent = _canonical_directory(stage_supplied.parent, "checkpoint parent")
    if stage_supplied.parent != parent:
        raise CheckpointPublicationError("checkpoint parent must be canonical")

    if not publish:
        if os.path.lexists(final_supplied):
            raise CheckpointPublicationError(
                "final checkpoint already exists; refusing audit-only operation: "
                f"{final_supplied}"
            )
        stage = _canonical_directory(stage_supplied, "hidden incomplete checkpoint")
        if stage.stat().st_dev != parent.stat().st_dev:
            raise CheckpointPublicationError(
                "stage and final parent are not on one filesystem"
            )
        prepublication = audit_checkpoint_root(stage, contract=contract)
        return {
            "schema_version": 2,
            "kind": "cwam_dsw_ema_checkpoint_publication",
            "status": "audit_passed",
            "mode": "audit_only",
            "stage_root": str(stage),
            "final_root": str(final_supplied),
            "final_root_absent_before_publication": True,
            "prepublication": prepublication,
        }

    receipt_output = Path(os.path.abspath(receipt_output.expanduser()))
    lock_path = Path(os.path.abspath(lock_path.expanduser()))
    if contract is FORMAL_CONTRACT and (
        receipt_output != DEFAULT_RECEIPT or lock_path != DEFAULT_LOCK
    ):
        raise CheckpointPublicationError(
            "formal publication receipt and lock paths are pinned"
        )
    if receipt_output.parent != lock_path.parent:
        raise CheckpointPublicationError("publication receipt and lock must share one directory")
    lock_parent = _canonical_directory(lock_path.parent, "publication lock parent")
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    lock_descriptor = os.open(lock_parent / lock_path.name, flags, 0o600)
    try:
        try:
            fcntl.flock(lock_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise CheckpointPublicationError("another checkpoint publisher owns the lock") from None
        if os.path.lexists(receipt_output):
            raise CheckpointPublicationError(
                "publication receipt already exists; refusing to alter checkpoint state"
            )
        stage, prepublication, recovered_uncommitted_final = (
            _recover_uncommitted_transaction_to_stage(
                stage_root=stage_supplied,
                final_root=final_supplied,
                parent=parent,
                contract=contract,
            )
        )
        if stage.stat().st_dev != parent.stat().st_dev:
            raise CheckpointPublicationError(
                "stage and final parent are not on one filesystem"
            )
        result: dict[str, Any] = {
            "schema_version": 2,
            "kind": "cwam_dsw_ema_checkpoint_publication",
            "status": "publishing",
            "mode": "cpfs_transaction_publish",
            "stage_root": str(stage),
            "final_root": str(final_supplied),
            "final_root_absent_before_publication": True,
            "recovered_uncommitted_final_before_publication": (
                recovered_uncommitted_final
            ),
            "prepublication": prepublication,
        }
        if (
            receipt_output.parent != lock_parent
            or os.path.lexists(receipt_output)
        ):
            raise CheckpointPublicationError(
                "publication receipt path must be absent in the canonical lock directory"
            )
        transaction_started = False
        commit_point_crossed = False
        try:
            _fsync_directory(parent)
            if os.path.lexists(final_supplied):
                raise CheckpointPublicationError(
                    "final checkpoint appeared before exclusive reservation"
                )
            stage_inode = _inode_key(stage.lstat())
            final, final_inode = _mkdir_exclusive_directory(
                final_supplied,
                parent=parent,
                label="exclusive final checkpoint reservation",
            )
            transaction_started = True
            if (
                stage.stat().st_dev != final.stat().st_dev
                or final.stat().st_dev != parent.stat().st_dev
            ):
                raise CheckpointPublicationError(
                    "stage, final reservation, and parent must share one filesystem"
                )
            _move_entries(
                source_root=stage,
                source_inode=stage_inode,
                destination_root=final,
                destination_inode=final_inode,
                names=tuple(_expected_top_level_names(contract)),
                contract=contract,
                source_label="hidden checkpoint transaction stage",
                destination_label="exclusive final checkpoint reservation",
            )
            if _directory_names(
                stage,
                stage_inode,
                label="hidden checkpoint transaction stage",
            ):
                raise CheckpointPublicationError(
                    "hidden checkpoint stage is not empty after entry transaction"
                )
            if _directory_names(
                final,
                final_inode,
                label="exclusive final checkpoint reservation",
            ) != _expected_top_level_names(contract):
                raise CheckpointPublicationError(
                    "final checkpoint reservation closure is incomplete"
                )
            _fsync_directory(parent)
            postpublication = audit_checkpoint_root(
                final,
                contract=contract,
            )
            if (
                postpublication["model"]["files"]
                != prepublication["model"]["files"]
                or postpublication["model"]["tree_sha256"]
                != prepublication["model"]["tree_sha256"]
                or [
                    (item["relative_path"], item["size_bytes"], item["sha256"])
                    for item in postpublication["top_level_files"]
                ]
                != [
                    (item["relative_path"], item["size_bytes"], item["sha256"])
                    for item in prepublication["top_level_files"]
                ]
            ):
                raise CheckpointPublicationError(
                    "post-publication full readback differs"
                )
            _assert_directory_inode(
                final,
                final_inode,
                label="exclusive final checkpoint reservation",
            )
            _remove_empty_directory(
                stage,
                stage_inode,
                parent=parent,
                label="empty hidden checkpoint transaction stage",
            )
            _assert_directory_inode(
                final,
                final_inode,
                label="exclusive final checkpoint reservation",
            )
            result.update(
                {
                    "status": "published",
                    "completed_at": dt.datetime.now(dt.timezone.utc)
                    .isoformat()
                    .replace("+00:00", "Z"),
                    "publication": {
                        "primitive": (
                            "mkdir(final)+per-entry-rename+"
                            "exclusive-receipt-hardlink"
                        ),
                        "same_parent_same_filesystem": True,
                        "parent_fsync_before_and_after": True,
                        "stage_absent_after": True,
                        "final_present_after": True,
                        "directory_visibility_atomic": False,
                        "committed_visibility_atomic": True,
                        "final_root_exclusive_reservation": True,
                        "final_root_inode_verified_throughout": True,
                        "receipt_is_commit_point": True,
                        "receipt_required_for_success": True,
                        "rollback_on_precommit_failure": True,
                        "no_rollback_after_receipt_commit": True,
                    },
                    "postpublication": postpublication,
                }
            )
            try:
                receipt = _atomic_write_read_only(receipt_output, result)
            except ReceiptCommitPointCrossedError:
                commit_point_crossed = True
                raise
            commit_point_crossed = True
            _assert_directory_inode(
                final,
                final_inode,
                label="committed final checkpoint",
            )
            result["receipt"] = receipt
            return result
        except Exception as publication_error:
            if transaction_started and not commit_point_crossed:
                try:
                    _recover_uncommitted_transaction_to_stage(
                        stage_root=stage_supplied,
                        final_root=final_supplied,
                        parent=parent,
                        contract=contract,
                    )
                except Exception as rollback_error:
                    raise CheckpointPublicationError(
                        "checkpoint publication failed and rollback could not "
                        f"restore the hidden stage: {rollback_error}"
                    ) from publication_error
            raise
    finally:
        os.close(lock_descriptor)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-root", type=Path, default=EXPECTED_STAGE_ROOT)
    parser.add_argument("--final-root", type=Path, default=EXPECTED_FINAL_ROOT)
    parser.add_argument("--receipt-output", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--lock-path", type=Path, default=DEFAULT_LOCK)
    parser.add_argument(
        "--publish",
        action="store_true",
        help=(
            "perform the CPFS exclusive-reservation entry transaction and "
            "receipt commit after every audit passes"
        ),
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        result = publish_checkpoint(
            stage_root=args.stage_root,
            final_root=args.final_root,
            receipt_output=args.receipt_output,
            lock_path=args.lock_path,
            publish=args.publish,
        )
    except (CheckpointPublicationError, OSError) as error:
        raise SystemExit(f"checkpoint publication failed: {error}") from error
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
