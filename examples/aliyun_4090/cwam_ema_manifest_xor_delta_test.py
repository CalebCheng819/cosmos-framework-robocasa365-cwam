#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""CPU tests for the manifest-aligned CWAM shard XOR codec."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cwam_ema_manifest_xor_delta as codec


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _chunk(index: int, offset: int, length: int, global_offset: list[int]) -> dict[str, object]:
    return {
        "chunk_index": index,
        "global_offset": global_offset,
        "target_storage": {
            "relative_path": "__0_0.distcp",
            "offset": offset,
            "length": length,
            "serialized_sha256": "0" * 64,
        },
    }


def _key(name: str, shape: list[int], chunks: list[dict[str, object]]) -> dict[str, object]:
    return {
        "key": name,
        "shape": shape,
        "chunk_count": len(chunks),
        "chunks": chunks,
    }


def _manifest(keys: list[dict[str, object]]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": "cwam_serving_ema_bf16_key_manifest",
        "key_count": len(keys),
        "keys": keys,
    }


def _fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, object], dict[str, object], bytes, bytes]:
    base_root = tmp_path / "base"
    target_root = tmp_path / "target"
    base_root.mkdir()
    target_root.mkdir()
    base = bytearray(b"b" * 96)
    base[8:16] = b"ABCDEFGH"
    base[40:48] = b"12345678"
    target = bytearray((index * 17 + 3) % 256 for index in range(112))
    target[24:32] = b"ABCDXFGH"
    target[64:72] = b"1234567Z"
    target[88:94] = b"EXTRA!"
    (base_root / "__0_0.distcp").write_bytes(base)
    (target_root / "__0_0.distcp").write_bytes(target)
    base_manifest = _manifest(
        [
            _key("net.a", [8], [_chunk(0, 8, 8, [0])]),
            _key("net.b", [8], [_chunk(0, 40, 8, [0])]),
        ]
    )
    target_manifest = _manifest(
        [
            _key("net.a", [8], [_chunk(0, 24, 8, [0])]),
            _key("net.b", [8], [_chunk(0, 64, 8, [0])]),
            _key("net.extra", [6], [_chunk(0, 88, 6, [0])]),
        ]
    )
    return base_root, target_root, base_manifest, target_manifest, bytes(base), bytes(target)


def test_shifted_chunks_and_extra_key_round_trip(tmp_path: Path) -> None:
    base_root, target_root, base_manifest, target_manifest, _, target = _fixture(tmp_path)
    plan = codec.build_delta_plan(
        base_manifest=base_manifest,
        target_manifest=target_manifest,
        target_shard="__0_0.distcp",
        target_size_bytes=len(target),
    )
    delta = tmp_path / "delta.gz"
    expected_target = codec.FileDescriptor(len(target), _sha(target))

    observed_delta, observed_target = codec.encode_delta(
        base_model_root=base_root,
        target_model_root=target_root,
        plan=plan,
        expected_target=expected_target,
        output=str(delta),
        compression_level=1,
        block_size=4096,
    )
    decoded = tmp_path / "decoded.distcp"
    round_trip = codec.decode_delta(
        base_model_root=base_root,
        plan=plan,
        delta_path=delta,
        expected_delta=observed_delta,
        expected_target=expected_target,
        output_path=decoded,
        block_size=4096,
    )

    assert observed_target == expected_target
    assert round_trip == expected_target
    assert decoded.read_bytes() == target
    assert decoded.stat().st_mode & 0o777 == 0o444
    assert plan.matched_bytes == 16
    assert plan.unmatched_chunk_bytes == 6
    assert plan.gap_bytes == len(target) - 22


def test_overlap_fails_closed(tmp_path: Path) -> None:
    _, _, base_manifest, target_manifest, _, target = _fixture(tmp_path)
    target_manifest["keys"][2]["chunks"][0]["target_storage"]["offset"] = 68

    with pytest.raises(codec.DeltaCodecError, match="ranges overlap"):
        codec.build_delta_plan(
            base_manifest=base_manifest,
            target_manifest=target_manifest,
            target_shard="__0_0.distcp",
            target_size_bytes=len(target),
        )


def test_unsafe_storage_path_fails_closed(tmp_path: Path) -> None:
    _, _, base_manifest, target_manifest, _, target = _fixture(tmp_path)
    target_manifest["keys"][0]["chunks"][0]["target_storage"]["relative_path"] = "../escape"

    with pytest.raises(codec.DeltaCodecError, match="relative_path is unsafe"):
        codec.build_delta_plan(
            base_manifest=base_manifest,
            target_manifest=target_manifest,
            target_shard="__0_0.distcp",
            target_size_bytes=len(target),
        )


def test_wrong_target_hash_removes_delta_output(tmp_path: Path) -> None:
    base_root, target_root, base_manifest, target_manifest, _, target = _fixture(tmp_path)
    plan = codec.build_delta_plan(
        base_manifest=base_manifest,
        target_manifest=target_manifest,
        target_shard="__0_0.distcp",
        target_size_bytes=len(target),
    )
    delta = tmp_path / "bad-target.gz"

    with pytest.raises(codec.DeltaCodecError, match="target shard descriptor mismatch"):
        codec.encode_delta(
            base_model_root=base_root,
            target_model_root=target_root,
            plan=plan,
            expected_target=codec.FileDescriptor(len(target), "0" * 64),
            output=str(delta),
            compression_level=1,
            block_size=4096,
        )

    assert not delta.exists()


def test_corrupt_delta_is_rejected_without_output(tmp_path: Path) -> None:
    base_root, target_root, base_manifest, target_manifest, _, target = _fixture(tmp_path)
    plan = codec.build_delta_plan(
        base_manifest=base_manifest,
        target_manifest=target_manifest,
        target_shard="__0_0.distcp",
        target_size_bytes=len(target),
    )
    delta = tmp_path / "delta.gz"
    expected_target = codec.FileDescriptor(len(target), _sha(target))
    observed_delta, _ = codec.encode_delta(
        base_model_root=base_root,
        target_model_root=target_root,
        plan=plan,
        expected_target=expected_target,
        output=str(delta),
        compression_level=1,
        block_size=4096,
    )
    delta.chmod(0o600)
    payload = bytearray(delta.read_bytes())
    payload[len(payload) // 2] ^= 1
    delta.write_bytes(payload)
    delta.chmod(0o444)
    output = tmp_path / "must-not-exist.distcp"

    with pytest.raises(codec.DeltaCodecError, match="SHA-256 mismatch"):
        codec.decode_delta(
            base_model_root=base_root,
            plan=plan,
            delta_path=delta,
            expected_delta=observed_delta,
            expected_target=expected_target,
            output_path=output,
            block_size=4096,
        )

    assert not output.exists()


def test_cli_writes_immutable_encode_receipt(tmp_path: Path) -> None:
    base_root, target_root, base_manifest, target_manifest, _, target = _fixture(tmp_path)
    base_path = tmp_path / "base-manifest.json"
    target_path = tmp_path / "target-manifest.json"
    base_payload = (json.dumps(base_manifest, sort_keys=True) + "\n").encode()
    target_payload = (json.dumps(target_manifest, sort_keys=True) + "\n").encode()
    base_path.write_bytes(base_payload)
    target_path.write_bytes(target_payload)
    delta = tmp_path / "cli-delta.gz"
    receipt = tmp_path / "encode-receipt.json"

    result = codec.main(
        [
            "encode",
            "--base-key-manifest",
            str(base_path),
            "--expected-base-key-manifest-sha256",
            _sha(base_payload),
            "--target-key-manifest",
            str(target_path),
            "--expected-target-key-manifest-sha256",
            _sha(target_payload),
            "--base-model-root",
            str(base_root),
            "--target-model-root",
            str(target_root),
            "--target-shard",
            "__0_0.distcp",
            "--expected-target-size",
            str(len(target)),
            "--expected-target-sha256",
            _sha(target),
            "--output",
            str(delta),
            "--receipt-output",
            str(receipt),
        ]
    )

    assert result == 0
    value = json.loads(receipt.read_text())
    assert value["status"] == "passed"
    assert value["delta"]["sha256"] == _sha(delta.read_bytes())
    assert receipt.stat().st_mode & 0o777 == 0o444
