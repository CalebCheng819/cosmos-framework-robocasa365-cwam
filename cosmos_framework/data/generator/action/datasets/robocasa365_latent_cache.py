# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Strict fixed-record Wan latent cache for RoboCasa365.

The cache deliberately uses ordinary ``os.pread`` calls rather than mmap.  The
H-cluster GPFS is a poor fit for mmap-heavy access, while fixed-size records in
large immutable shards give workers cheap random reads without creating one
file per training window.

Only caches with a checksum-linked ``_SUCCESS.json`` and a passed bitwise
equivalence gate are accepted.  There is no online-video fallback in this
reader. BF16 records remain BF16 on the CPU and across host-to-device transfer;
the model converts them to the cache's logical float32 dtype on the GPU.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch

CACHE_SCHEMA_VERSION = 1
CACHE_MANIFEST_NAME = "manifest.json"
CACHE_SUCCESS_NAME = "_SUCCESS.json"
CACHE_STAGE = "post_padding_crop"
IDENTITY_DIGEST_ALGORITHM = "sha256-length-prefixed-canonical-json-v1"
OFFICIAL_WAN22_VAE_SHA256 = "20eb789667fa5e60e7516bf509512f6cb61f01b0aa0695eadaea930c13892b36"
WAN_CACHE_VAE_BEHAVIOR_CONTRACT: dict[str, Any] = {
    "contract_scope": "behavior_relevant_active_robocasa365_recipe",
    "implementation": "Wan2pt2VAEInterface",
    "dtype": "bfloat16",
    "chunk_duration": 93,
    "keep_decoder_cache": False,
    "latent_channels": 48,
    "spatial_compression_factor": 16,
    "temporal_compression_factor": 4,
    "causal": True,
    "use_streaming_encode": False,
    "encode_chunk_frames": {"256": 68, "480": 24, "720": 8, "768": 8},
    "encode_exact_durations": [17, 61, 73],
}

_STORAGE_DTYPES: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}
_DTYPE_BYTES = {
    "bfloat16": 2,
    "float32": 4,
}


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of *path* without mmap."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize JSON deterministically for stable cache fingerprints."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def ordered_identity_sha256(identities: Iterable[Mapping[str, Any]]) -> str:
    """Hash an ordered identity stream without ambiguous JSON concatenation."""

    digest = hashlib.sha256()
    for identity in identities:
        payload = canonical_json_bytes(dict(identity))
        digest.update(len(payload).to_bytes(8, byteorder="little", signed=False))
        digest.update(payload)
    return digest.hexdigest()


def active_preprocessing_code_sha256() -> dict[str, str]:
    """Hash every active implementation that defines cached latent semantics."""

    package_root = Path(__file__).parents[4]
    paths = {
        "robocasa365_latent_cache": Path(__file__),
        "robocasa365_segment_dataset": Path(__file__).with_name("robocasa365_segment_dataset.py"),
        "action_transforms": Path(__file__).parents[1] / "transforms.py",
        "wan2pt2_vae_4x16x16": package_root / "model/generator/tokenizers/wan2pt2_vae_4x16x16.py",
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"RoboCasa365 latent cache is missing {label}: {path}.")
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Could not read RoboCasa365 latent cache {label} at {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"RoboCasa365 latent cache {label} must be a JSON object: {path}.")
    return value


def _safe_relative_path(root: Path, relative: object, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"RoboCasa365 latent cache {label} must be a non-empty relative path.")
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"RoboCasa365 latent cache {label} must stay under the cache root: {relative!r}.")
    resolved = (root / candidate).resolve()
    resolved_root = root.resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(f"RoboCasa365 latent cache {label} escapes the cache root: {relative!r}.") from error
    return resolved


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"RoboCasa365 latent cache {label} must be a 64-character SHA-256 digest.")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"RoboCasa365 latent cache {label} is not hexadecimal.") from error
    return value.lower()


def _strict_int(value: object, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"RoboCasa365 latent cache {label} must be an integer >= {minimum}, got {value!r}.")
    return value


def validate_published_cache_files(
    root: str | Path,
    *,
    verify_shard_checksums: bool = True,
    verify_mapping_contents: bool = True,
) -> dict[str, Any]:
    """Validate a published cache and return its declared file inventory.

    This is the shared, training-independent parser for GPFS-to-NVMe staging.
    It validates ``_SUCCESS -> manifest``, every mapping (checksum, dtype,
    length, range, unique count), and every fixed-record shard (strict integer
    ID/range, size, and optionally content checksum). Mapping checksums, dtype,
    length, and ordinal range are always checked; ``verify_mapping_contents``
    additionally recomputes each unique-record count for staging/preflight.
    No model or dataset expected values are required.

    Returns:
        A dictionary containing ``cache_id``, ``manifest_sha256``, the parsed
        ``manifest``, and ``files``. Each file entry has ``role``, a
        cache-root-relative ``path``, ``size_bytes``, ``sha256``, and
        ``checksum_verified``.
    """

    cache_root = Path(root)
    success_path = cache_root / CACHE_SUCCESS_NAME
    success = _read_json_object(success_path, CACHE_SUCCESS_NAME)
    if _strict_int(success.get("schema_version"), f"{CACHE_SUCCESS_NAME}.schema_version", minimum=1) != 1:
        raise ValueError("Unsupported RoboCasa365 latent cache success schema_version.")
    if success.get("status") != "complete":
        raise RuntimeError(
            f"RoboCasa365 latent cache is not complete: status={success.get('status')!r} at {success_path}."
        )
    manifest_path = _safe_relative_path(
        cache_root,
        success.get("manifest_path"),
        f"{CACHE_SUCCESS_NAME}.manifest_path",
    )
    if manifest_path.name != CACHE_MANIFEST_NAME:
        raise ValueError(
            f"RoboCasa365 latent cache manifest must be named {CACHE_MANIFEST_NAME!r}, got {manifest_path.name!r}."
        )
    expected_manifest_sha = _require_sha256(
        success.get("manifest_sha256"),
        f"{CACHE_SUCCESS_NAME}.manifest_sha256",
    )
    actual_manifest_sha = sha256_file(manifest_path)
    if actual_manifest_sha != expected_manifest_sha:
        raise RuntimeError(
            "RoboCasa365 latent cache manifest checksum mismatch: "
            f"expected={expected_manifest_sha}, actual={actual_manifest_sha}."
        )
    manifest = _read_json_object(manifest_path, CACHE_MANIFEST_NAME)
    if _strict_int(manifest.get("schema_version"), "manifest.schema_version", minimum=1) != 1:
        raise ValueError("Unsupported RoboCasa365 latent cache manifest schema_version.")
    cache_id = manifest.get("cache_id")
    if not isinstance(cache_id, str) or not cache_id:
        raise ValueError("RoboCasa365 latent cache manifest.cache_id must be a non-empty string.")
    if success.get("cache_id") != cache_id:
        raise RuntimeError(
            "RoboCasa365 latent cache cache_id mismatch: "
            f"success={success.get('cache_id')!r}, manifest={cache_id!r}."
        )

    latent = manifest.get("latent")
    if not isinstance(latent, dict):
        raise ValueError("RoboCasa365 latent cache manifest.latent must be an object.")
    raw_shape = latent.get("shape")
    if (
        not isinstance(raw_shape, list)
        or len(raw_shape) != 4
        or any(type(dim) is not int or dim <= 0 for dim in raw_shape)
    ):
        raise ValueError("RoboCasa365 latent cache latent.shape must contain four positive integers.")
    storage_dtype = latent.get("storage_dtype")
    if storage_dtype not in _STORAGE_DTYPES:
        raise ValueError(f"Unsupported RoboCasa365 latent cache storage_dtype={storage_dtype!r}.")
    expected_record_nbytes = int(np.prod(raw_shape, dtype=np.int64)) * _DTYPE_BYTES[str(storage_dtype)]
    record_nbytes = _strict_int(latent.get("record_nbytes"), "latent.record_nbytes", minimum=1)
    record_stride_bytes = _strict_int(latent.get("record_stride_bytes"), "latent.record_stride_bytes", minimum=1)
    if record_nbytes != expected_record_nbytes or record_stride_bytes < record_nbytes:
        raise ValueError(
            "RoboCasa365 latent cache fixed-record layout is inconsistent: "
            f"record_nbytes={record_nbytes}, expected={expected_record_nbytes}, stride={record_stride_bytes}."
        )

    equivalence = manifest.get("equivalence")
    if (
        not isinstance(equivalence, dict)
        or equivalence.get("status") != "passed"
        or equivalence.get("torch_equal") is not True
        or _strict_int(equivalence.get("num_samples"), "equivalence.num_samples", minimum=1) < 1
    ):
        raise RuntimeError("RoboCasa365 latent cache does not contain a passed bitwise equivalence gate.")

    files: list[dict[str, Any]] = [
        {
            "role": "success",
            "path": CACHE_SUCCESS_NAME,
            "size_bytes": success_path.stat().st_size,
            "sha256": sha256_file(success_path),
            "checksum_verified": True,
        },
        {
            "role": "manifest",
            "path": CACHE_MANIFEST_NAME,
            "size_bytes": manifest_path.stat().st_size,
            "sha256": actual_manifest_sha,
            "checksum_verified": True,
        },
    ]
    declared_paths = {CACHE_SUCCESS_NAME, CACHE_MANIFEST_NAME}

    raw_shards = manifest.get("shards")
    if not isinstance(raw_shards, list) or not raw_shards:
        raise ValueError("RoboCasa365 latent cache manifest.shards must be a non-empty list.")
    shard_ranges: list[tuple[int, int, int]] = []
    seen_shard_ids: set[int] = set()
    for index, raw_shard in enumerate(raw_shards):
        if not isinstance(raw_shard, dict):
            raise ValueError(f"RoboCasa365 latent cache shards[{index}] must be an object.")
        shard_id = _strict_int(raw_shard.get("shard_id"), f"shards[{index}].shard_id")
        if shard_id in seen_shard_ids:
            raise ValueError(f"Duplicate RoboCasa365 latent cache shard_id={shard_id}.")
        first = _strict_int(raw_shard.get("first_ordinal"), f"shards[{index}].first_ordinal")
        count = _strict_int(raw_shard.get("num_records"), f"shards[{index}].num_records", minimum=1)
        size = _strict_int(raw_shard.get("size_bytes"), f"shards[{index}].size_bytes", minimum=1)
        expected_size = count * record_stride_bytes
        if size != expected_size:
            raise ValueError(
                f"RoboCasa365 latent cache shard {shard_id} size metadata mismatch: "
                f"manifest={size}, expected={expected_size}."
            )
        relative_path = raw_shard.get("path")
        path = _safe_relative_path(cache_root, relative_path, f"shards[{index}].path")
        relative = str(relative_path)
        if relative in declared_paths:
            raise ValueError(f"Duplicate declared RoboCasa365 latent cache path: {relative!r}.")
        declared_paths.add(relative)
        if not path.is_file():
            raise FileNotFoundError(f"RoboCasa365 latent cache shard is missing: {path}.")
        actual_size = path.stat().st_size
        if actual_size != size:
            raise RuntimeError(
                f"RoboCasa365 latent cache shard {shard_id} size mismatch: manifest={size}, actual={actual_size}."
            )
        expected_sha = _require_sha256(raw_shard.get("sha256"), f"shards[{index}].sha256")
        if verify_shard_checksums:
            actual_sha = sha256_file(path)
            if actual_sha != expected_sha:
                raise RuntimeError(
                    f"RoboCasa365 latent cache shard {shard_id} checksum mismatch: "
                    f"expected={expected_sha}, actual={actual_sha}."
                )
        files.append(
            {
                "role": "shard",
                "path": relative,
                "size_bytes": actual_size,
                "sha256": expected_sha,
                "checksum_verified": bool(verify_shard_checksums),
            }
        )
        seen_shard_ids.add(shard_id)
        shard_ranges.append((first, first + count, shard_id))

    shard_ranges.sort()
    expected_first = 0
    for first, stop, shard_id in shard_ranges:
        if first != expected_first:
            raise ValueError(
                "RoboCasa365 latent cache shards must cover one contiguous ordinal range: "
                f"expected first={expected_first}, got {first} for shard {shard_id}."
            )
        expected_first = stop
    num_unique_records = expected_first

    mapping_gate = manifest.get("mapping_gate")
    if not isinstance(mapping_gate, dict):
        raise ValueError("RoboCasa365 latent cache manifest.mapping_gate must be an object.")
    if (
        mapping_gate.get("status") != "passed"
        or mapping_gate.get("full_coverage") is not True
        or mapping_gate.get("algorithm") != "canonical-first-occurrence-v1"
        or mapping_gate.get("identity_digest_algorithm") != IDENTITY_DIGEST_ALGORITHM
    ):
        raise RuntimeError(f"RoboCasa365 latent cache mapping full-coverage gate did not pass: {mapping_gate!r}.")
    gate_unique_records = _strict_int(
        mapping_gate.get("unique_records"),
        "mapping_gate.unique_records",
        minimum=1,
    )
    if gate_unique_records != num_unique_records:
        raise RuntimeError(
            "RoboCasa365 latent cache mapping gate unique-record count mismatch: "
            f"gate={gate_unique_records}, shards={num_unique_records}."
        )
    identity_record_bytes = _strict_int(
        mapping_gate.get("ordinal_identity_record_bytes"),
        "mapping_gate.ordinal_identity_record_bytes",
        minimum=1,
    )
    if identity_record_bytes != 32:
        raise ValueError(
            f"RoboCasa365 latent cache ordinal identity records must be 32 bytes, got {identity_record_bytes}."
        )
    identity_relative_path = mapping_gate.get("ordinal_identity_path")
    identity_path = _safe_relative_path(
        cache_root,
        identity_relative_path,
        "mapping_gate.ordinal_identity_path",
    )
    identity_relative = str(identity_relative_path)
    if identity_relative in declared_paths:
        raise ValueError(f"Duplicate declared RoboCasa365 latent cache path: {identity_relative!r}.")
    declared_paths.add(identity_relative)
    expected_identity_size = num_unique_records * identity_record_bytes
    gate_identity_size = _strict_int(
        mapping_gate.get("ordinal_identity_size_bytes"),
        "mapping_gate.ordinal_identity_size_bytes",
        minimum=1,
    )
    if gate_identity_size != expected_identity_size or not identity_path.is_file():
        raise RuntimeError(
            "RoboCasa365 latent cache ordinal identity inventory is missing or has invalid metadata: "
            f"path={identity_path}, size={gate_identity_size}, expected={expected_identity_size}."
        )
    actual_identity_size = identity_path.stat().st_size
    if actual_identity_size != expected_identity_size:
        raise RuntimeError(
            "RoboCasa365 latent cache ordinal identity inventory size mismatch: "
            f"actual={actual_identity_size}, expected={expected_identity_size}."
        )
    expected_identity_sha = _require_sha256(
        mapping_gate.get("ordinal_identity_sha256"),
        "mapping_gate.ordinal_identity_sha256",
    )
    actual_identity_sha = sha256_file(identity_path)
    if actual_identity_sha != expected_identity_sha:
        raise RuntimeError(
            "RoboCasa365 latent cache ordinal identity inventory checksum mismatch: "
            f"expected={expected_identity_sha}, actual={actual_identity_sha}."
        )
    files.append(
        {
            "role": "ordinal_identity_inventory",
            "path": identity_relative,
            "size_bytes": actual_identity_size,
            "sha256": actual_identity_sha,
            "checksum_verified": True,
        }
    )

    raw_splits = manifest.get("splits")
    if not isinstance(raw_splits, dict) or not raw_splits:
        raise ValueError("RoboCasa365 latent cache manifest.splits must be a non-empty object.")
    total_logical_samples = 0
    for split, raw_split in sorted(raw_splits.items()):
        if not isinstance(split, str) or not split or not isinstance(raw_split, dict):
            raise ValueError(f"Invalid RoboCasa365 latent cache split entry {split!r}.")
        logical_samples = _strict_int(raw_split.get("logical_samples"), f"splits.{split}.logical_samples")
        total_logical_samples += logical_samples
        unique_records = _strict_int(raw_split.get("unique_records"), f"splits.{split}.unique_records")
        _require_sha256(
            raw_split.get("source_split_manifest_sha256"),
            f"splits.{split}.source_split_manifest_sha256",
        )
        _require_sha256(
            raw_split.get("ordered_visual_identity_sha256"),
            f"splits.{split}.ordered_visual_identity_sha256",
        )
        relative_path = raw_split.get("mapping_path")
        mapping_path = _safe_relative_path(cache_root, relative_path, f"splits.{split}.mapping_path")
        relative = str(relative_path)
        if relative in declared_paths:
            raise ValueError(f"Duplicate declared RoboCasa365 latent cache path: {relative!r}.")
        declared_paths.add(relative)
        if not mapping_path.is_file():
            raise FileNotFoundError(f"RoboCasa365 latent cache mapping is missing: {mapping_path}.")
        expected_sha = _require_sha256(raw_split.get("mapping_sha256"), f"splits.{split}.mapping_sha256")
        actual_sha = sha256_file(mapping_path)
        if actual_sha != expected_sha:
            raise RuntimeError(
                f"RoboCasa365 latent cache split {split!r} mapping checksum mismatch: "
                f"expected={expected_sha}, actual={actual_sha}."
            )
        mapping = np.load(mapping_path, allow_pickle=False)
        if mapping.ndim != 1 or mapping.dtype not in (np.dtype("uint32"), np.dtype("uint64")):
            raise ValueError(
                f"RoboCasa365 latent cache mapping must be uint32/uint64 [N], got {mapping.shape} {mapping.dtype}."
            )
        if mapping.shape[0] != logical_samples:
            raise RuntimeError(
                f"RoboCasa365 latent cache split {split!r} mapping length mismatch: "
                f"manifest={logical_samples}, mapping={mapping.shape[0]}."
            )
        if mapping.size and int(mapping.max()) >= num_unique_records:
            raise RuntimeError(
                f"RoboCasa365 latent cache split {split!r} mapping ordinal is out of range: "
                f"max={int(mapping.max())}, records={num_unique_records}."
            )
        if unique_records > num_unique_records:
            raise RuntimeError(
                f"RoboCasa365 latent cache split {split!r} declares too many unique records: "
                f"split={unique_records}, global={num_unique_records}."
            )
        if verify_mapping_contents:
            actual_unique = int(np.unique(mapping).shape[0])
            if actual_unique != unique_records:
                raise RuntimeError(
                    f"RoboCasa365 latent cache split {split!r} unique-record count mismatch: "
                    f"manifest={unique_records}, mapping={actual_unique}."
                )
        files.append(
            {
                "role": f"mapping:{split}",
                "path": relative,
                "size_bytes": mapping_path.stat().st_size,
                "sha256": actual_sha,
                "checksum_verified": True,
            }
        )
    gate_logical_samples = _strict_int(
        mapping_gate.get("logical_samples"),
        "mapping_gate.logical_samples",
        minimum=1,
    )
    if gate_logical_samples != total_logical_samples:
        raise RuntimeError(
            "RoboCasa365 latent cache mapping gate logical-sample count mismatch: "
            f"gate={gate_logical_samples}, splits={total_logical_samples}."
        )

    return {
        "cache_id": cache_id,
        "manifest_sha256": actual_manifest_sha,
        "manifest": manifest,
        "files": files,
    }


@dataclass(frozen=True)
class LatentRecordLayout:
    """Physical and logical layout of one fixed-size cache record."""

    shape: tuple[int, int, int, int]
    storage_dtype_name: str
    logical_dtype_name: str
    record_nbytes: int
    record_stride_bytes: int
    video_num_frames: int
    image_size: tuple[int, int, int, int]

    @property
    def storage_dtype(self) -> torch.dtype:
        return _STORAGE_DTYPES[self.storage_dtype_name]

    @property
    def numel(self) -> int:
        return int(np.prod(self.shape, dtype=np.int64))


@dataclass(frozen=True)
class LatentShard:
    """Validated ordinal range backed by one immutable shard file."""

    shard_id: int
    path: Path
    first_ordinal: int
    num_records: int
    size_bytes: int
    sha256: str

    @property
    def stop_ordinal(self) -> int:
        return self.first_ordinal + self.num_records


class RoboCasa365LatentCache:
    """Read a strict, checksum-linked RoboCasa365 Wan latent cache.

    Args:
        root: Published cache root containing ``manifest.json`` and
            ``_SUCCESS.json``.
        split: Split whose flat-index mapping should be loaded.
        expected_dataset_id: Dataset ID from the active RoboCasa365 manifest.
        expected_dataset_manifest_sha256: SHA-256 of the active
            ``dataset_manifest.json``.
        expected_num_samples: Exact flat dataset length for *split*.
        expected_camera_mode: Active ordered camera-layout mode.
        expected_ordered_camera_fields: Active ordered source camera fields.
        expected_fps: Active video sampling frequency.
        expected_dataset_image_size: Per-camera square resize performed by the
            RoboCasa365 dataset before camera concatenation.
        expected_image_size: Exact output of the active ``VideoResize``
            contract: ``[target_h, target_w, content_h, content_w]``.
        expected_vae_weights_sha256: SHA-256 of the Wan weights used by the
            active training model.
        expected_preprocessing_code_sha256: Hashes of the active dataset,
            resize, cache reader, and Wan encode implementations.
        expected_split_manifest_sha256: SHA-256 of the active split parquet;
            together with loader code hashes this binds ordering/identity
            semantics without an O(number of windows) startup pass.
        expected_ordered_visual_identity_sha256: Optional precomputed digest
            from a converted view that publishes one. The cache builder always
            writes and prepublish-verifies this digest.
        verify_shard_checksums: Hash every shard during construction.  This is
            intended for preflight/staging verification; normal workers still
            validate the checksum-linked manifest, mapping checksum, shard
            sizes, and every short read.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        split: str,
        expected_dataset_id: str,
        expected_dataset_manifest_sha256: str,
        expected_num_samples: int,
        expected_camera_mode: str,
        expected_ordered_camera_fields: tuple[str, ...],
        expected_fps: float,
        expected_dataset_image_size: int,
        expected_image_size: tuple[int, int, int, int],
        expected_vae_weights_sha256: str,
        expected_preprocessing_code_sha256: Mapping[str, str],
        expected_split_manifest_sha256: str,
        expected_ordered_visual_identity_sha256: str | None = None,
        expected_video_resize_resolution: str | int | None = None,
        verify_shard_checksums: bool = False,
    ) -> None:
        self.root = Path(root)
        self.split = str(split)
        self._fds: dict[Path, int] = {}
        self._fd_pid = os.getpid()

        published = validate_published_cache_files(
            self.root,
            verify_shard_checksums=bool(verify_shard_checksums),
            # Runtime workers trust the prepublish full-coverage gate plus
            # checksum-linked mappings; the staging/preflight API defaults to
            # recomputing unique counts once.
            verify_mapping_contents=False,
        )
        manifest = published["manifest"]
        cache_id = published["cache_id"]
        actual_manifest_sha = published["manifest_sha256"]

        dataset = manifest.get("dataset")
        if not isinstance(dataset, dict):
            raise ValueError("RoboCasa365 latent cache manifest.dataset must be an object.")
        if dataset.get("dataset_id") != expected_dataset_id:
            raise RuntimeError(
                "RoboCasa365 latent cache dataset_id mismatch: "
                f"cache={dataset.get('dataset_id')!r}, active={expected_dataset_id!r}."
            )
        cached_dataset_sha = _require_sha256(
            dataset.get("manifest_sha256"),
            "manifest.dataset.manifest_sha256",
        )
        active_dataset_sha = _require_sha256(
            expected_dataset_manifest_sha256,
            "expected_dataset_manifest_sha256",
        )
        if cached_dataset_sha != active_dataset_sha:
            raise RuntimeError(
                "RoboCasa365 latent cache dataset manifest checksum mismatch: "
                f"cache={cached_dataset_sha}, active={active_dataset_sha}."
            )

        equivalence = manifest.get("equivalence")
        if not isinstance(equivalence, dict):
            raise ValueError("RoboCasa365 latent cache manifest.equivalence must be an object.")
        if equivalence.get("status") != "passed" or equivalence.get("torch_equal") is not True:
            raise RuntimeError(
                "RoboCasa365 latent cache requires a passed bitwise equivalence gate; "
                f"got status={equivalence.get('status')!r}, torch_equal={equivalence.get('torch_equal')!r}."
            )
        if int(equivalence.get("num_samples", 0)) < 1:
            raise RuntimeError("RoboCasa365 latent cache equivalence gate must cover at least one sample.")

        self.cache_id = cache_id
        self.manifest = manifest
        self.manifest_sha256 = actual_manifest_sha
        self.preprocessing = self._validate_preprocessing(
            manifest.get("preprocessing"),
            expected_camera_mode=expected_camera_mode,
            expected_ordered_camera_fields=expected_ordered_camera_fields,
            expected_fps=expected_fps,
            expected_dataset_image_size=expected_dataset_image_size,
            expected_image_size=expected_image_size,
            expected_video_resize_resolution=expected_video_resize_resolution,
            expected_code_sha256=expected_preprocessing_code_sha256,
        )
        self.vae = self._validate_vae(
            manifest.get("vae"),
            expected_weights_sha256=expected_vae_weights_sha256,
        )
        self.layout = self._parse_layout(manifest.get("latent"))
        self._validate_derived_layout()
        self.shards = self._parse_shards(
            manifest.get("shards"),
            # The shared published-files validator above has already applied
            # the caller's checksum policy; do not hash 100+ GiB twice.
            verify_checksums=False,
        )
        self._shard_first_ordinals = [shard.first_ordinal for shard in self.shards]
        self.num_unique_records = self.shards[-1].stop_ordinal
        self.mapping = self._load_mapping(
            manifest.get("splits"),
            expected_num_samples=int(expected_num_samples),
            expected_split_manifest_sha256=expected_split_manifest_sha256,
            expected_ordered_visual_identity_sha256=expected_ordered_visual_identity_sha256,
        )

    def _validate_preprocessing(
        self,
        value: object,
        *,
        expected_camera_mode: str,
        expected_ordered_camera_fields: tuple[str, ...],
        expected_fps: float,
        expected_dataset_image_size: int,
        expected_image_size: tuple[int, int, int, int],
        expected_video_resize_resolution: str | int | None,
        expected_code_sha256: Mapping[str, str],
    ) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise ValueError("RoboCasa365 latent cache manifest.preprocessing must be an object.")
        if int(value.get("schema_version", -1)) != 1:
            raise ValueError("RoboCasa365 latent cache preprocessing.schema_version must be 1.")
        if value.get("camera_mode") != expected_camera_mode:
            raise RuntimeError(
                "RoboCasa365 latent cache camera_mode mismatch: "
                f"cache={value.get('camera_mode')!r}, active={expected_camera_mode!r}."
            )
        camera_fields = value.get("ordered_camera_fields")
        if camera_fields != list(expected_ordered_camera_fields):
            raise RuntimeError(
                "RoboCasa365 latent cache ordered camera fields mismatch: "
                f"cache={camera_fields!r}, active={list(expected_ordered_camera_fields)!r}."
            )
        try:
            cached_fps = float(value.get("fps"))
        except (TypeError, ValueError) as error:
            raise ValueError("RoboCasa365 latent cache preprocessing.fps must be numeric.") from error
        if not math.isfinite(cached_fps) or not math.isclose(cached_fps, float(expected_fps), rel_tol=0, abs_tol=1e-9):
            raise RuntimeError(
                f"RoboCasa365 latent cache fps mismatch: cache={cached_fps}, active={expected_fps}."
            )
        if int(value.get("dataset_image_size", -1)) != int(expected_dataset_image_size):
            raise RuntimeError(
                "RoboCasa365 latent cache dataset image_size mismatch: "
                f"cache={value.get('dataset_image_size')!r}, active={expected_dataset_image_size}."
            )

        resize = value.get("video_resize")
        if not isinstance(resize, dict):
            raise ValueError("RoboCasa365 latent cache preprocessing.video_resize must be an object.")
        normalized_resolution = (
            None if expected_video_resize_resolution is None else str(expected_video_resize_resolution)
        )
        required_resize = {
            "resolution": normalized_resolution,
            "resolution_selection": (
                "auto_get_vision_data_resolution" if normalized_resolution is None else "explicit_resolution_tier"
            ),
            "keep_aspect_ratio": True,
            "pad_keys": ["video"],
            "resize_interpolation": "bicubic",
            "antialias": True,
            "padding_origin": "top_left",
            "padding_mode": "reflect_else_edge",
            "image_size": list(expected_image_size),
        }
        if resize != required_resize:
            raise RuntimeError(
                "RoboCasa365 latent cache VideoResize contract mismatch: "
                f"cache={resize!r}, active={required_resize!r}."
            )
        crop = value.get("post_encode_crop")
        required_crop = {
            "stage": CACHE_STAGE,
            "spatial_rounding": "floor",
            "minimum_latent_size": 1,
        }
        if crop != required_crop:
            raise RuntimeError(
                "RoboCasa365 latent cache post-encode crop contract mismatch: "
                f"cache={crop!r}, active={required_crop!r}."
            )

        raw_code_sha = value.get("code_sha256")
        if not isinstance(raw_code_sha, dict):
            raise ValueError("RoboCasa365 latent cache preprocessing.code_sha256 must be an object.")
        cached_code_sha = {
            str(name): _require_sha256(digest, f"preprocessing.code_sha256.{name}")
            for name, digest in raw_code_sha.items()
        }
        active_code_sha = {
            str(name): _require_sha256(digest, f"expected_preprocessing_code_sha256.{name}")
            for name, digest in expected_code_sha256.items()
        }
        if cached_code_sha != active_code_sha:
            raise RuntimeError(
                "RoboCasa365 latent cache preprocessing implementation hashes do not match the active code: "
                f"cache={cached_code_sha!r}, active={active_code_sha!r}."
            )
        return value

    def _validate_vae(self, value: object, *, expected_weights_sha256: str) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise ValueError("RoboCasa365 latent cache manifest.vae must be an object.")
        weights_sha = _require_sha256(value.get("weights_sha256"), "manifest.vae.weights_sha256")
        active_weights_sha = _require_sha256(expected_weights_sha256, "expected_vae_weights_sha256")
        if weights_sha != active_weights_sha:
            raise RuntimeError(
                "RoboCasa365 latent cache Wan VAE weights checksum mismatch: "
                f"cache={weights_sha}, active={active_weights_sha}."
            )
        if int(value.get("weights_size_bytes", -1)) <= 0:
            raise ValueError("RoboCasa365 latent cache vae.weights_size_bytes must be positive.")
        config = value.get("behavior_relevant_config")
        if config != WAN_CACHE_VAE_BEHAVIOR_CONTRACT:
            raise RuntimeError(
                "RoboCasa365 latent cache Wan VAE configuration mismatch: "
                f"cache={config!r}, required={WAN_CACHE_VAE_BEHAVIOR_CONTRACT!r}."
            )
        return value

    def _validate_derived_layout(self) -> None:
        config = self.vae["behavior_relevant_config"]
        frames = self.layout.video_num_frames
        temporal_factor = int(config["temporal_compression_factor"])
        image_size = self.layout.image_size
        expected_shape = (
            int(config["latent_channels"]),
            (frames - 1) // temporal_factor + 1,
            max(image_size[2] // int(config["spatial_compression_factor"]), 1),
            max(image_size[3] // int(config["spatial_compression_factor"]), 1),
        )
        if self.layout.shape != expected_shape:
            raise RuntimeError(
                "RoboCasa365 latent cache shape does not match the VAE and post-crop contract: "
                f"cache={self.layout.shape}, derived={expected_shape}."
            )
        resize_image_size = tuple(self.preprocessing["video_resize"]["image_size"])
        if self.layout.image_size != resize_image_size:
            raise RuntimeError(
                "RoboCasa365 latent cache latent.image_size disagrees with preprocessing.video_resize: "
                f"latent={self.layout.image_size}, resize={resize_image_size}."
            )

    def _parse_layout(self, value: object) -> LatentRecordLayout:
        if not isinstance(value, dict):
            raise ValueError("RoboCasa365 latent cache manifest.latent must be an object.")
        raw_shape = value.get("shape")
        if (
            not isinstance(raw_shape, list)
            or len(raw_shape) != 4
            or any(not isinstance(dim, int) or dim <= 0 for dim in raw_shape)
        ):
            raise ValueError("RoboCasa365 latent cache latent.shape must contain four positive integers.")
        shape = tuple(raw_shape)
        storage_dtype_name = value.get("storage_dtype")
        if storage_dtype_name not in _STORAGE_DTYPES:
            raise ValueError(
                "RoboCasa365 latent cache storage_dtype must be one of "
                f"{sorted(_STORAGE_DTYPES)}, got {storage_dtype_name!r}."
            )
        logical_dtype_name = value.get("logical_dtype")
        if logical_dtype_name != "float32":
            raise ValueError(
                f"RoboCasa365 latent cache logical_dtype must be 'float32', got {logical_dtype_name!r}."
            )
        if value.get("stage") != CACHE_STAGE:
            raise ValueError(
                f"RoboCasa365 latent cache stage must be {CACHE_STAGE!r}, got {value.get('stage')!r}."
            )
        if value.get("byte_order") != "little":
            raise ValueError("RoboCasa365 latent cache byte_order must be 'little'.")

        numel = int(np.prod(shape, dtype=np.int64))
        expected_nbytes = numel * _DTYPE_BYTES[str(storage_dtype_name)]
        record_nbytes = int(value.get("record_nbytes", -1))
        record_stride_bytes = int(value.get("record_stride_bytes", -1))
        if record_nbytes != expected_nbytes:
            raise ValueError(
                "RoboCasa365 latent cache record_nbytes mismatch: "
                f"manifest={record_nbytes}, expected={expected_nbytes} for shape={shape} "
                f"dtype={storage_dtype_name}."
            )
        if record_stride_bytes < record_nbytes:
            raise ValueError(
                "RoboCasa365 latent cache record_stride_bytes must be at least record_nbytes: "
                f"stride={record_stride_bytes}, record={record_nbytes}."
            )

        video_num_frames = int(value.get("video_num_frames", -1))
        raw_image_size = value.get("image_size")
        if video_num_frames <= 0:
            raise ValueError("RoboCasa365 latent cache video_num_frames must be positive.")
        if (
            not isinstance(raw_image_size, list)
            or len(raw_image_size) != 4
            or any(not isinstance(dim, int) or dim <= 0 for dim in raw_image_size)
        ):
            raise ValueError("RoboCasa365 latent cache image_size must contain four positive integers.")

        return LatentRecordLayout(
            shape=shape,  # type: ignore[arg-type]
            storage_dtype_name=str(storage_dtype_name),
            logical_dtype_name=str(logical_dtype_name),
            record_nbytes=record_nbytes,
            record_stride_bytes=record_stride_bytes,
            video_num_frames=video_num_frames,
            image_size=tuple(raw_image_size),  # type: ignore[arg-type]
        )

    def _parse_shards(self, value: object, *, verify_checksums: bool) -> list[LatentShard]:
        if not isinstance(value, list) or not value:
            raise ValueError("RoboCasa365 latent cache manifest.shards must be a non-empty list.")
        shards: list[LatentShard] = []
        seen_ids: set[int] = set()
        for raw in value:
            if not isinstance(raw, dict):
                raise ValueError("Every RoboCasa365 latent cache shard entry must be an object.")
            shard_id = int(raw.get("shard_id", -1))
            first_ordinal = int(raw.get("first_ordinal", -1))
            num_records = int(raw.get("num_records", -1))
            size_bytes = int(raw.get("size_bytes", -1))
            if shard_id < 0 or shard_id in seen_ids:
                raise ValueError(f"Invalid or duplicate RoboCasa365 latent cache shard_id={shard_id}.")
            if first_ordinal < 0 or num_records <= 0:
                raise ValueError(
                    f"Invalid RoboCasa365 latent cache shard range: first={first_ordinal}, count={num_records}."
                )
            expected_size = num_records * self.layout.record_stride_bytes
            if size_bytes != expected_size:
                raise ValueError(
                    f"RoboCasa365 latent cache shard {shard_id} size metadata mismatch: "
                    f"manifest={size_bytes}, expected={expected_size}."
                )
            path = _safe_relative_path(self.root, raw.get("path"), f"shards[{shard_id}].path")
            if not path.is_file():
                raise FileNotFoundError(f"RoboCasa365 latent cache shard is missing: {path}.")
            actual_size = path.stat().st_size
            if actual_size != size_bytes:
                raise RuntimeError(
                    f"RoboCasa365 latent cache shard {shard_id} size mismatch: "
                    f"manifest={size_bytes}, actual={actual_size}."
                )
            expected_sha = _require_sha256(raw.get("sha256"), f"shards[{shard_id}].sha256")
            if verify_checksums:
                actual_sha = sha256_file(path)
                if actual_sha != expected_sha:
                    raise RuntimeError(
                        f"RoboCasa365 latent cache shard {shard_id} checksum mismatch: "
                        f"expected={expected_sha}, actual={actual_sha}."
                    )
            seen_ids.add(shard_id)
            shards.append(
                LatentShard(
                    shard_id=shard_id,
                    path=path,
                    first_ordinal=first_ordinal,
                    num_records=num_records,
                    size_bytes=size_bytes,
                    sha256=expected_sha,
                )
            )

        shards.sort(key=lambda shard: shard.first_ordinal)
        expected_first = 0
        for shard in shards:
            if shard.first_ordinal != expected_first:
                raise ValueError(
                    "RoboCasa365 latent cache shards must cover one contiguous ordinal range: "
                    f"expected first={expected_first}, got {shard.first_ordinal} for shard {shard.shard_id}."
                )
            expected_first = shard.stop_ordinal
        return shards

    def _load_mapping(
        self,
        raw_splits: object,
        *,
        expected_num_samples: int,
        expected_split_manifest_sha256: str,
        expected_ordered_visual_identity_sha256: str | None,
    ) -> np.ndarray:
        if not isinstance(raw_splits, dict) or self.split not in raw_splits:
            raise KeyError(f"RoboCasa365 latent cache has no mapping for split={self.split!r}.")
        split_info = raw_splits[self.split]
        if not isinstance(split_info, dict):
            raise ValueError(f"RoboCasa365 latent cache split {self.split!r} entry must be an object.")
        logical_samples = int(split_info.get("logical_samples", -1))
        if logical_samples != expected_num_samples:
            raise RuntimeError(
                f"RoboCasa365 latent cache split {self.split!r} sample count mismatch: "
                f"cache={logical_samples}, dataset={expected_num_samples}."
            )
        cached_split_sha = _require_sha256(
            split_info.get("source_split_manifest_sha256"),
            f"splits.{self.split}.source_split_manifest_sha256",
        )
        active_split_sha = _require_sha256(
            expected_split_manifest_sha256,
            "expected_split_manifest_sha256",
        )
        if cached_split_sha != active_split_sha:
            raise RuntimeError(
                f"RoboCasa365 latent cache split {self.split!r} source parquet mismatch: "
                f"cache={cached_split_sha}, active={active_split_sha}."
            )
        cached_identity_sha = _require_sha256(
            split_info.get("ordered_visual_identity_sha256"),
            f"splits.{self.split}.ordered_visual_identity_sha256",
        )
        if expected_ordered_visual_identity_sha256 is not None:
            active_identity_sha = _require_sha256(
                expected_ordered_visual_identity_sha256,
                "expected_ordered_visual_identity_sha256",
            )
            if cached_identity_sha != active_identity_sha:
                raise RuntimeError(
                    f"RoboCasa365 latent cache split {self.split!r} ordered visual identity mismatch: "
                    f"cache={cached_identity_sha}, active={active_identity_sha}."
                )
        mapping_path = _safe_relative_path(
            self.root,
            split_info.get("mapping_path"),
            f"splits.{self.split}.mapping_path",
        )
        expected_sha = _require_sha256(
            split_info.get("mapping_sha256"),
            f"splits.{self.split}.mapping_sha256",
        )
        actual_sha = sha256_file(mapping_path)
        if actual_sha != expected_sha:
            raise RuntimeError(
                f"RoboCasa365 latent cache split {self.split!r} mapping checksum mismatch: "
                f"expected={expected_sha}, actual={actual_sha}."
            )
        mapping = np.load(mapping_path, allow_pickle=False)
        if mapping.ndim != 1 or mapping.dtype not in (np.dtype("uint32"), np.dtype("uint64")):
            raise ValueError(
                f"RoboCasa365 latent cache mapping must be uint32/uint64 [N], got {mapping.shape} {mapping.dtype}."
            )
        if mapping.shape[0] != expected_num_samples:
            raise RuntimeError(
                f"RoboCasa365 latent cache mapping length mismatch: "
                f"mapping={mapping.shape[0]}, dataset={expected_num_samples}."
            )
        if mapping.size and int(mapping.max()) >= self.num_unique_records:
            raise RuntimeError(
                "RoboCasa365 latent cache mapping references an out-of-range ordinal: "
                f"max={int(mapping.max())}, records={self.num_unique_records}."
            )
        return mapping

    def __len__(self) -> int:
        return int(self.mapping.shape[0])

    @property
    def latent_shape(self) -> tuple[int, int, int, int]:
        return self.layout.shape

    @property
    def video_num_frames(self) -> int:
        return self.layout.video_num_frames

    @property
    def image_size(self) -> tuple[int, int, int, int]:
        return self.layout.image_size

    def ordinal_for_sample(self, sample_index: int) -> int:
        sample_index = int(sample_index)
        if sample_index < 0:
            sample_index += len(self)
        if sample_index < 0 or sample_index >= len(self):
            raise IndexError(
                f"RoboCasa365 latent cache sample index {sample_index} out of range for split length {len(self)}."
            )
        return int(self.mapping[sample_index])

    def read_sample(self, sample_index: int) -> torch.Tensor:
        """Read one split sample as a contiguous stored-dtype ``[C,T,H,W]``."""

        return self.read_ordinal(self.ordinal_for_sample(sample_index))

    def read_ordinal(self, ordinal: int) -> torch.Tensor:
        """Read one unique cache ordinal using a single fixed-size ``os.pread``."""

        ordinal = int(ordinal)
        if ordinal < 0 or ordinal >= self.num_unique_records:
            raise IndexError(
                f"RoboCasa365 latent cache ordinal {ordinal} out of range for {self.num_unique_records} records."
            )
        shard_index = bisect.bisect_right(self._shard_first_ordinals, ordinal) - 1
        shard = self.shards[shard_index]
        row = ordinal - shard.first_ordinal
        offset = row * self.layout.record_stride_bytes
        fd = self._get_fd(shard.path)
        payload = os.pread(fd, self.layout.record_nbytes, offset)
        if len(payload) != self.layout.record_nbytes:
            raise RuntimeError(
                f"Short read from RoboCasa365 latent cache shard {shard.shard_id}: "
                f"ordinal={ordinal}, offset={offset}, expected={self.layout.record_nbytes}, got={len(payload)}."
            )

        # bytearray gives torch a writable buffer. Preserve BF16 storage through
        # the CPU and H2D paths; get_data_and_condition converts it to logical
        # float32 on the GPU. Clone so the returned tensor owns its buffer.
        stored = torch.frombuffer(
            bytearray(payload),
            dtype=self.layout.storage_dtype,
            count=self.layout.numel,
        ).reshape(self.layout.shape)
        result = stored.clone().contiguous()
        if not torch.isfinite(result).all():
            raise RuntimeError(
                f"RoboCasa365 latent cache record contains non-finite values: ordinal={ordinal}, "
                f"shard={shard.shard_id}."
            )
        return result

    def _get_fd(self, path: Path) -> int:
        pid = os.getpid()
        if pid != self._fd_pid:
            # A reader may be constructed before DataLoader workers fork.
            # Never inherit and share the parent's descriptor cache.
            self.close()
            self._fd_pid = pid
        fd = self._fds.get(path)
        if fd is None:
            fd = os.open(path, os.O_RDONLY)
            self._fds[path] = fd
        return fd

    def close(self) -> None:
        fds = getattr(self, "_fds", {})
        for fd in fds.values():
            try:
                os.close(fd)
            except OSError:
                pass
        fds.clear()

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_fds"] = {}
        state["_fd_pid"] = os.getpid()
        return state

    def __del__(self) -> None:
        self.close()
