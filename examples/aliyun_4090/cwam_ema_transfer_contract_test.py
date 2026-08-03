# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""CPU tests for the formal CWAM EMA transfer-manifest contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from examples.aliyun_4090 import cwam_ema_transfer_contract as contract

pytestmark = [pytest.mark.L0, pytest.mark.CPU]

REMOTE = "eailab-hdd2:eailab/chengjuntao/cwam-s1/ema-serving-dcp"


def _item(path: str, *, gzip: bool) -> dict[str, object]:
    raw_path = path.removesuffix(".gz")
    value: dict[str, object] = {
        "object_path": path,
        "encoding": "gzip-pigz-1-n" if gzip else "identity",
        "raw_size_bytes": 10,
        "raw_sha256": "1" * 64,
        "transfer_size_bytes": 8 if gzip else 10,
        "transfer_sha256": "2" * 64 if gzip else "1" * 64,
        "sha256_sidecar_path": f"{path}.sha256",
        "sha256_sidecar_size_bytes": 80,
        "sha256_sidecar_sha256": "3" * 64,
        "remote_size_verified": True,
    }
    if gzip:
        value["raw_object_path"] = raw_path
    return value


def _manifest() -> dict[str, object]:
    items = [
        *(_item(path, gzip=False) for path in contract.IDENTITY_PATHS),
        *(_item(path, gzip=True) for path in contract.GZIP_PATHS),
    ]
    items.sort(key=lambda item: str(item["object_path"]))
    return {
        "schema_version": 1,
        "kind": "cwam_ema_serving_dcp_ceph_transfer_manifest",
        "status": "uploaded_remote_size_verified",
        "created_at": "2026-08-03T00:00:00Z",
        "remote_prefix": REMOTE,
        "item_count": 17,
        "items": items,
        "bindings": {
            "export_receipt_sha256": "4" * 64,
            "source_dcp_full_verification_sha256": "5" * 64,
            "target_tree_sha256": "6" * 64,
        },
        "verification_boundary": {
            "local_transfer_sha256_computed_before_upload": True,
            "remote_object_set_and_size_verified": True,
            "remote_content_sha256_readback": False,
            "consumer_must_verify_transfer_sha256_after_download": True,
            "consumer_must_verify_raw_sha256_after_decompression": True,
        },
    }


def _write(path: Path, value: object) -> str:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    path.write_bytes(payload)
    path.chmod(0o444)
    return hashlib.sha256(payload).hexdigest()


def test_valid_manifest_is_bound_and_queryable(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    expected = _write(path, _manifest())

    value = contract.load_transfer_manifest(
        path,
        expected_sha256=expected,
        expected_remote_prefix=REMOTE,
    )

    assert value["manifest_sha256"] == expected
    item = contract.item_by_object_path(value, "model/__7_0.distcp.gz")
    assert item["raw_object_path"] == "model/__7_0.distcp"


@pytest.mark.parametrize("mutation", ["wrong_hash", "missing_item", "wrong_boundary"])
def test_manifest_mutations_fail_closed(tmp_path: Path, mutation: str) -> None:
    value = _manifest()
    path = tmp_path / "manifest.json"
    expected = _write(path, value)
    if mutation == "wrong_hash":
        expected = "0" * 64
    else:
        path.chmod(0o644)
        if mutation == "missing_item":
            value["items"] = value["items"][:-1]
        else:
            value["verification_boundary"]["remote_content_sha256_readback"] = True
        expected = _write(path, value)

    with pytest.raises(contract.TransferContractError):
        contract.load_transfer_manifest(
            path,
            expected_sha256=expected,
            expected_remote_prefix=REMOTE,
        )


def test_remote_prefix_cannot_escape_personal_hdd2_prefix(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    expected = _write(path, _manifest())
    with pytest.raises(contract.TransferContractError, match="chengjuntao"):
        contract.load_transfer_manifest(
            path,
            expected_sha256=expected,
            expected_remote_prefix="eailab-hdd2:eailab/another-user/checkpoint",
        )
