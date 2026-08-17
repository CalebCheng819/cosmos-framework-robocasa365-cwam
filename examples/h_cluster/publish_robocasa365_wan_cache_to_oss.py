# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Publish one completed full-padding RoboCasa365 Wan cache to guarded HDD2 OSS."""

from __future__ import annotations

import argparse
import configparser
import datetime as dt
import hashlib
import json
import os
import re
import stat
import subprocess
import urllib.request
from pathlib import Path
from typing import Any, Iterable

from cosmos_framework.data.generator.action.datasets.robocasa365_latent_cache import (
    CACHE_STAGE_FULL_PADDING,
    sha256_file,
    validate_published_cache_files,
)

_REMOTE_PREFIX = "eailab-hdd2:eailab/chengjuntao/"
_EXPECTED_ENDPOINT = "http://hdd2.h.pjlab.org.cn:8060"
_CAPACITY_URLS = {
    "user": "http://xsky-query.sre.svc.h.pjlab.org.cn/hdd2_user/dexmanip",
    "eailab": "http://xsky-query.sre.svc.h.pjlab.org.cn/hdd2_bucket/eailab",
    "fkp_migrate": "http://xsky-query.sre.svc.h.pjlab.org.cn/hdd2_bucket/fkp-migrate",
}
_MAX_EXPECTED_USED_GB = 20_000.0
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8") + b"\n"


def _write_json_atomic(path: Path, value: object, *, mode: int = 0o444) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(_json_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _parse_gb(value: object, label: str) -> float:
    if not isinstance(value, str) or not value.endswith("GB"):
        raise RuntimeError(f"Capacity API {label} must be a decimal GB string, got {value!r}.")
    parsed = float(value[:-2])
    if parsed < 0:
        raise RuntimeError(f"Capacity API {label} cannot be negative, got {value!r}.")
    return parsed


def _capacity_gate(upload_bytes: int) -> dict[str, Any]:
    payloads: dict[str, dict[str, Any]] = {}
    for name, url in _CAPACITY_URLS.items():
        with urllib.request.urlopen(url, timeout=15) as response:  # noqa: S310 - fixed internal endpoints
            value = json.load(response)
        if not isinstance(value, dict):
            raise RuntimeError(f"Capacity API {name} did not return an object.")
        payloads[name] = value

    user_used_gb = _parse_gb(payloads["user"].get("used"), "user.used")
    bucket_used_gb = _parse_gb(payloads["eailab"].get("used"), "eailab.used") + _parse_gb(
        payloads["fkp_migrate"].get("used"), "fkp_migrate.used"
    )
    current_used_gb = max(user_used_gb, bucket_used_gb)
    upload_gb = upload_bytes / 1_000_000_000
    expected_used_gb = current_used_gb + upload_gb
    result = {
        "status": "passed" if expected_used_gb <= _MAX_EXPECTED_USED_GB else "blocked",
        "threshold_gb": _MAX_EXPECTED_USED_GB,
        "user_used_gb": user_used_gb,
        "bucket_used_gb": bucket_used_gb,
        "current_used_gb": current_used_gb,
        "upload_bytes": upload_bytes,
        "upload_gb": upload_gb,
        "expected_used_gb": expected_used_gb,
        "remaining_to_threshold_gb": _MAX_EXPECTED_USED_GB - expected_used_gb,
        "raw": payloads,
    }
    if result["status"] != "passed":
        raise RuntimeError(
            "HDD2 expected use exceeds the mandatory 20 TB reporting threshold: "
            f"current={current_used_gb:.2f} GB, upload={upload_gb:.3f} GB, "
            f"expected={expected_used_gb:.3f} GB."
        )
    return result


def _validate_remote(remote: str) -> None:
    if not remote.startswith(_REMOTE_PREFIX):
        raise ValueError(f"OSS target must stay below {_REMOTE_PREFIX!r}, got {remote!r}.")
    relative = remote[len(_REMOTE_PREFIX) :].strip("/")
    if not relative or ".." in Path(relative).parts or "//" in relative:
        raise ValueError(f"OSS target has an unsafe relative prefix: {remote!r}.")


def _validate_rclone_config(path: Path) -> None:
    path = path.resolve(strict=True)
    metadata = path.stat()
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.geteuid() or metadata.st_mode & 0o077:
        raise RuntimeError("Decrypted rclone config must be an owner-only regular file.")
    parser = configparser.RawConfigParser()
    parser.read(path)
    if not parser.has_section("eailab-hdd2"):
        raise RuntimeError("Decrypted rclone config has no eailab-hdd2 remote.")
    section = parser["eailab-hdd2"]
    required = {
        "type": "s3",
        "provider": "Ceph",
        "endpoint": _EXPECTED_ENDPOINT,
        "acl": "private",
    }
    actual = {key: section.get(key) for key in required}
    if actual != required:
        raise RuntimeError(f"eailab-hdd2 non-secret config mismatch: expected={required!r}, actual={actual!r}.")
    if not section.get("access_key_id") or not section.get("secret_access_key"):
        raise RuntimeError("eailab-hdd2 credentials are absent from the decrypted config.")


def _run_rclone(
    rclone: Path,
    config: Path,
    arguments: Iterable[str],
    *,
    capture: bool = False,
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["RCLONE_CONFIG"] = str(config)
    command = [str(rclone), *arguments]
    return subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
        env=environment,
    )


def _remote_files(rclone: Path, config: Path, remote: str) -> list[str]:
    result = _run_rclone(
        rclone,
        config,
        ("lsf", remote, "--recursive", "--files-only"),
        capture=True,
    )
    return sorted(line for line in result.stdout.splitlines() if line)


def _remote_cat(rclone: Path, config: Path, remote_path: str) -> bytes:
    environment = os.environ.copy()
    environment["RCLONE_CONFIG"] = str(config)
    result = subprocess.run(
        [str(rclone), "cat", remote_path],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
    )
    return result.stdout


def _remote_sha256(rclone: Path, config: Path, remote_path: str) -> str:
    environment = os.environ.copy()
    environment["RCLONE_CONFIG"] = str(config)
    process = subprocess.Popen(
        [str(rclone), "cat", remote_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
    )
    assert process.stdout is not None and process.stderr is not None
    digest = hashlib.sha256()
    for block in iter(lambda: process.stdout.read(16 * 1024 * 1024), b""):
        digest.update(block)
    stderr = process.stderr.read().decode("utf-8", errors="replace")
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"rclone cat failed for {remote_path!r}: {stderr[-2000:]}")
    return digest.hexdigest()


def _build_inventory(cache_root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    shard_metadata = {
        str(entry["path"]): (int(entry["size_bytes"]), str(entry["sha256"]))
        for entry in manifest["shards"]
    }
    files: list[dict[str, Any]] = []
    total_bytes = 0
    for path in sorted(cache_root.rglob("*"), key=lambda item: item.relative_to(cache_root).as_posix()):
        if path.is_symlink():
            raise RuntimeError(f"OSS publication refuses symlinks: {path}.")
        if path.is_dir():
            continue
        if not path.is_file():
            raise RuntimeError(f"OSS publication found a non-regular entry: {path}.")
        relative = path.relative_to(cache_root).as_posix()
        if ".partial" in path.name or ".tmp-" in path.name:
            raise RuntimeError(f"OSS publication refuses transient cache file: {path}.")
        size = path.stat().st_size
        if relative in shard_metadata:
            expected_size, digest = shard_metadata[relative]
            if size != expected_size or not _SHA256_RE.fullmatch(digest):
                raise RuntimeError(f"Manifest shard metadata mismatch for {relative!r}.")
            digest_source = "validated_cache_manifest"
        else:
            digest = sha256_file(path)
            digest_source = "publisher_full_read"
        files.append(
            {
                "path": relative,
                "size_bytes": size,
                "sha256": digest,
                "sha256_source": digest_source,
            }
        )
        total_bytes += size
    return {
        "schema_version": 1,
        "algorithm": "relative-path-size-sha256-v1",
        "file_count": len(files),
        "total_bytes": total_bytes,
        "files": files,
    }


def _sample_readback_paths(manifest: dict[str, Any]) -> list[str]:
    shards = [str(entry["path"]) for entry in manifest["shards"]]
    selected = {shards[0], shards[len(shards) // 2], shards[-1]}
    selected.update({"manifest.json", "build_plan.json", "_SUCCESS.json"})
    return sorted(selected)


def publish(args: argparse.Namespace) -> dict[str, Any]:
    cache_root = args.cache_root.resolve(strict=True)
    work_root = args.work_root.resolve(strict=True)
    rclone = args.rclone.resolve(strict=True)
    rclone_config = args.rclone_config.resolve(strict=True)
    if not cache_root.is_dir() or not work_root.is_dir() or not os.access(work_root, os.W_OK):
        raise RuntimeError("Cache/work roots must be existing accessible directories.")
    if not rclone.is_file() or not os.access(rclone, os.X_OK):
        raise RuntimeError(f"Pinned rclone is not executable: {rclone}.")
    if sha256_file(rclone) != args.expected_rclone_sha256:
        raise RuntimeError("Pinned rclone SHA-256 mismatch.")
    _validate_remote(args.remote)
    _validate_rclone_config(rclone_config)

    published = validate_published_cache_files(cache_root, verify_shard_checksums=False)
    manifest = published["manifest"]
    latent = manifest.get("latent", {})
    if (
        latent.get("stage") != CACHE_STAGE_FULL_PADDING
        or latent.get("shape") != [48, 5, 30, 52]
        or latent.get("image_size") != [480, 832, 256, 768]
        or latent.get("storage_dtype") != "bfloat16"
    ):
        raise RuntimeError(f"Cache is not the pinned full-padding BF16 contract: latent={latent!r}.")

    inventory = _build_inventory(cache_root, manifest)
    inventory.update(
        {
            "cache_id": published["cache_id"],
            "manifest_sha256": published["manifest_sha256"],
            "remote": args.remote,
            "run_id": args.run_id,
            "source_commit": args.source_commit,
        }
    )
    inventory_path = work_root / "OSS_INVENTORY.json"
    if inventory_path.exists():
        existing = json.loads(inventory_path.read_text())
        if existing != inventory:
            raise RuntimeError("Existing OSS inventory differs from the current immutable cache.")
    else:
        _write_json_atomic(inventory_path, inventory)
    inventory_sha256 = sha256_file(inventory_path)

    reservation = {
        "schema_version": 1,
        "kind": "robocasa365_wan_full_padding_oss_reservation",
        "run_id": args.run_id,
        "cache_id": published["cache_id"],
        "manifest_sha256": published["manifest_sha256"],
        "inventory_sha256": inventory_sha256,
        "remote": args.remote,
        "source_commit": args.source_commit,
    }
    reservation_path = work_root / "OSS_UPLOAD_RESERVATION.json"
    if reservation_path.exists():
        if json.loads(reservation_path.read_text()) != reservation:
            raise RuntimeError("Existing local OSS reservation differs from the current request.")
    else:
        _write_json_atomic(reservation_path, reservation)

    remote_reservation = f"{args.remote.rstrip('/')}/OSS_UPLOAD_RESERVATION.json"
    existing_files = _remote_files(rclone, rclone_config, args.remote)
    if existing_files:
        if "OSS_UPLOAD_RESERVATION.json" not in existing_files:
            raise RuntimeError("OSS target is non-empty but has no matching publication reservation.")
        if json.loads(_remote_cat(rclone, rclone_config, remote_reservation)) != reservation:
            raise RuntimeError("Remote OSS reservation does not match this immutable cache request.")
        if "OSS_PUBLISHED.json" in existing_files:
            marker = json.loads(_remote_cat(rclone, rclone_config, f"{args.remote.rstrip('/')}/OSS_PUBLISHED.json"))
            if marker.get("status") == "complete" and marker.get("inventory_sha256") == inventory_sha256:
                return marker
            raise RuntimeError("OSS target already has a non-matching terminal publication marker.")
    else:
        _run_rclone(
            rclone,
            rclone_config,
            ("copyto", str(reservation_path), remote_reservation, "--transfers", "1", "--checkers", "1"),
        )

    capacity = _capacity_gate(int(inventory["total_bytes"]))
    _write_json_atomic(work_root / "OSS_CAPACITY_GATE.json", capacity)

    common_transfer = (
        "--transfers",
        "24",
        "--checkers",
        "64",
        "--s3-upload-concurrency",
        "4",
        "--s3-chunk-size",
        "64M",
        "--retries",
        "8",
        "--low-level-retries",
        "20",
        "--immutable",
        "--fast-list",
    )
    _run_rclone(
        rclone,
        rclone_config,
        (
            "copy",
            str(cache_root),
            args.remote,
            "--exclude",
            "_SUCCESS.json",
            *common_transfer,
        ),
    )
    _run_rclone(
        rclone,
        rclone_config,
        (
            "check",
            str(cache_root),
            args.remote,
            "--one-way",
            "--size-only",
            "--exclude",
            "_SUCCESS.json",
            "--checkers",
            "64",
        ),
    )
    _run_rclone(
        rclone,
        rclone_config,
        ("copyto", str(inventory_path), f"{args.remote.rstrip('/')}/OSS_INVENTORY.json", "--immutable"),
    )
    success_path = cache_root / "_SUCCESS.json"
    _run_rclone(
        rclone,
        rclone_config,
        ("copyto", str(success_path), f"{args.remote.rstrip('/')}/_SUCCESS.json", "--immutable"),
    )

    inventory_by_path = {entry["path"]: entry for entry in inventory["files"]}
    readback: list[dict[str, Any]] = []
    for relative in _sample_readback_paths(manifest):
        expected = inventory_by_path[relative]["sha256"]
        actual = _remote_sha256(rclone, rclone_config, f"{args.remote.rstrip('/')}/{relative}")
        if actual != expected:
            raise RuntimeError(f"OSS SHA-256 readback mismatch for {relative!r}: {actual} != {expected}.")
        readback.append({"path": relative, "sha256": actual, "status": "passed"})

    marker = {
        "schema_version": 1,
        "kind": "robocasa365_wan_full_padding_oss_publication",
        "status": "complete",
        "completed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "run_id": args.run_id,
        "cache_id": published["cache_id"],
        "source_commit": args.source_commit,
        "remote": args.remote,
        "manifest_sha256": published["manifest_sha256"],
        "inventory_sha256": inventory_sha256,
        "source_file_count": inventory["file_count"],
        "source_total_bytes": inventory["total_bytes"],
        "latent_contract": latent,
        "capacity_gate": capacity,
        "size_check": {"status": "passed", "one_way": True, "size_only": True},
        "sha256_readback": readback,
    }
    marker_path = work_root / "OSS_PUBLISHED.json"
    _write_json_atomic(marker_path, marker)
    remote_marker = f"{args.remote.rstrip('/')}/OSS_PUBLISHED.json"
    _run_rclone(rclone, rclone_config, ("copyto", str(marker_path), remote_marker, "--immutable"))
    if hashlib.sha256(_remote_cat(rclone, rclone_config, remote_marker)).hexdigest() != sha256_file(marker_path):
        raise RuntimeError("Terminal OSS publication marker failed independent SHA-256 readback.")
    return marker


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--remote", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--rclone", type=Path, required=True)
    parser.add_argument("--rclone-config", type=Path, required=True)
    parser.add_argument("--expected-rclone-sha256", required=True)
    parser.add_argument("--receipt-path", type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9._-]+", args.run_id):
        parser.error("--run-id contains unsupported characters.")
    if not re.fullmatch(r"[0-9a-f]{40}", args.source_commit):
        parser.error("--source-commit must be a lowercase 40-character Git SHA.")
    if not _SHA256_RE.fullmatch(args.expected_rclone_sha256):
        parser.error("--expected-rclone-sha256 must be a lowercase SHA-256.")
    return args


def main() -> None:
    args = _parse_args()
    marker = publish(args)
    receipt = {
        "schema_version": 1,
        "kind": "robocasa365_wan_full_padding_terminal_receipt",
        "status": "complete",
        "run_id": args.run_id,
        "source_commit": args.source_commit,
        "remote": args.remote,
        "marker": marker,
    }
    if args.receipt_path.exists():
        if json.loads(args.receipt_path.read_text()) != receipt:
            raise RuntimeError("Existing terminal receipt differs from completed OSS publication.")
    else:
        _write_json_atomic(args.receipt_path, receipt)
    print(
        "OSS_PUBLICATION_COMPLETE "
        f"remote={args.remote} bytes={marker['source_total_bytes']} "
        f"files={marker['source_file_count']} cache_id={marker['cache_id']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
