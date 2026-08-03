#!/usr/bin/env python3
"""Generate a host-bound PAI DLC RoboCasa EGL mapping receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import socket
import stat
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMES = (
    "isolated_physical_identity",
    "visible_all_physical_index",
    "isolated_relative_zero",
)
GPU_COUNT = 8


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def descriptor(path: Path, expected_mode: int) -> dict[str, Any]:
    canonical = path.resolve(strict=True)
    if canonical != path:
        raise ValueError(f"path is not canonical or is a symlink: {path}")
    metadata = path.stat()
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"not a regular file: {path}")
    actual_mode = stat.S_IMODE(metadata.st_mode)
    if actual_mode != expected_mode:
        raise ValueError(f"mode mismatch for {path}: expected={expected_mode:04o} actual={actual_mode:04o}")
    if metadata.st_size <= 0:
        raise ValueError(f"empty evidence file: {path}")
    return {
        "path": str(path),
        "size_bytes": metadata.st_size,
        "sha256": sha256_file(path),
        "mode": f"{expected_mode:04o}",
    }


def gpu_inventory(gpu_count: int) -> list[dict[str, Any]]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,pci.bus_id",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    rows: list[dict[str, Any]] = []
    for raw_line in completed.stdout.splitlines():
        fields = [field.strip() for field in raw_line.split(",")]
        if len(fields) != 4:
            raise ValueError(f"unexpected nvidia-smi inventory line: {raw_line!r}")
        physical_index, uuid, name, pci_bus_id = fields
        rows.append(
            {
                "physical_index": int(physical_index),
                "uuid": uuid,
                "name": name,
                "pci_bus_id": pci_bus_id,
            }
        )
    if [row["physical_index"] for row in rows] != list(range(gpu_count)):
        raise ValueError(f"expected exact {gpu_count}-GPU inventory, got {rows!r}")
    if any(not row["uuid"].startswith("GPU-") or "4090" not in row["name"] for row in rows):
        raise ValueError(f"expected homogeneous 4090 GPU inventory, got {rows!r}")
    return rows


def ready_row(path: Path) -> dict[str, Any]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) != 2:
        raise ValueError(f"probe log must contain exactly two lines: {path}")
    ready, closed = (json.loads(line) for line in lines)
    if ready.get("event") != "egl_ready" or closed.get("event") != "egl_closed":
        raise ValueError(f"probe log events are invalid: {path}")
    if type(ready.get("pid")) is not int or ready["pid"] <= 0:
        raise ValueError(f"probe PID is invalid: {path}")
    if type(closed.get("frames")) is not int or closed["frames"] <= 0:
        raise ValueError(f"probe frame count is invalid: {path}")
    return ready


def pmon_observed_index(path: Path, probe_pid: int) -> int:
    matches: set[int] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) >= 3 and fields[0].isdigit() and fields[1].isdigit():
            if int(fields[1]) == probe_pid:
                if fields[2] != "G":
                    raise ValueError(f"probe PID is not a graphics process: {path}")
                matches.add(int(fields[0]))
    if len(matches) != 1:
        raise ValueError(f"probe PID must map to one pmon GPU: {path} {matches!r}")
    return next(iter(matches))


def snapshot_observed_index(path: Path, probe_pid: int) -> int:
    process_pattern = re.compile(rf"^\|\s*(?P<gpu>[0-9]+)\s+N/A\s+N/A\s+{probe_pid}\s+G\s+")
    matches = {
        int(match.group("gpu"))
        for line in path.read_text(encoding="utf-8").splitlines()
        if (match := process_pattern.match(line))
    }
    if len(matches) != 1:
        raise ValueError(f"probe PID must map to one snapshot GPU: {path} {matches!r}")
    return next(iter(matches))


def expected_environment(
    scheme: str,
    physical_index: int,
    gpu_count: int,
) -> tuple[str, str]:
    if scheme == "visible_all_physical_index":
        return ",".join(str(index) for index in range(gpu_count)), str(physical_index)
    if scheme == "isolated_relative_zero":
        return str(physical_index), "0"
    if scheme == "isolated_physical_identity":
        return str(physical_index), str(physical_index)
    raise ValueError(f"unsupported scheme: {scheme}")


def build_receipt(
    evidence_root: Path,
    *,
    environment: Path,
    ready_marker_sha256: str,
    probe: Path,
    probe_sha256: str,
    vendor: Path,
    vendor_sha256: str,
    expected_python: str,
    expected_mujoco: str,
    gpu_count: int,
) -> dict[str, Any]:
    if evidence_root.resolve(strict=True) != evidence_root or not evidence_root.is_dir():
        raise ValueError(f"evidence root is not canonical: {evidence_root}")
    if gpu_count != GPU_COUNT:
        raise ValueError(f"formal DLC evaluation requires exactly {GPU_COUNT} GPUs")
    environment = environment.resolve(strict=True)
    probe = probe.resolve(strict=True)
    vendor = vendor.resolve(strict=True)
    inventory = gpu_inventory(gpu_count)
    ready_marker = environment / ".cwam_env_ready.json"
    if sha256_file(ready_marker) != ready_marker_sha256:
        raise ValueError("RoboCasa ready marker SHA-256 mismatch")
    if sha256_file(probe) != probe_sha256:
        raise ValueError("EGL probe script SHA-256 mismatch")
    if sha256_file(vendor) != vendor_sha256:
        raise ValueError("EGL vendor JSON SHA-256 mismatch")

    schemes: dict[str, Any] = {}
    evidence_paths: set[Path] = set()
    for scheme in SCHEMES:
        rows: list[dict[str, Any]] = []
        mapping_results: list[bool] = []
        for physical_index in range(gpu_count):
            prefix = evidence_root / f"{scheme}-gpu{physical_index}"
            probe_log = Path(f"{prefix}-probe.log")
            pmon = Path(f"{prefix}-nvidia-smi-pmon.log")
            snapshot = Path(f"{prefix}-nvidia-smi-snapshot.log")
            for evidence_path in (probe_log, pmon, snapshot):
                if evidence_path in evidence_paths:
                    raise ValueError(f"duplicate evidence path: {evidence_path}")
                evidence_paths.add(evidence_path)

            ready = ready_row(probe_log)
            cuda_visible_devices, mujoco_egl_device_id = expected_environment(scheme, physical_index, gpu_count)
            if (
                ready.get("python") != expected_python
                or ready.get("mujoco") != expected_mujoco
                or ready.get("mujoco_gl") != "egl"
                or ready.get("cuda_visible_devices") != cuda_visible_devices
                or ready.get("mujoco_egl_device_id") != mujoco_egl_device_id
            ):
                raise ValueError(f"probe environment mismatch: {probe_log}")

            probe_pid = ready["pid"]
            pmon_index = pmon_observed_index(pmon, probe_pid)
            snapshot_index = snapshot_observed_index(snapshot, probe_pid)
            if pmon_index != snapshot_index:
                raise ValueError(f"pmon/snapshot disagree for {scheme} GPU {physical_index}")
            observed_index = pmon_index
            expected_identity = inventory[physical_index]
            observed_identity = inventory[observed_index]
            mapping_matches = (
                observed_index == physical_index and observed_identity["uuid"] == expected_identity["uuid"]
            )
            mapping_results.append(mapping_matches)
            rows.append(
                {
                    "physical_index": physical_index,
                    "cuda_visible_devices": cuda_visible_devices,
                    "mujoco_egl_device_id": mujoco_egl_device_id,
                    "probe_pid": probe_pid,
                    "renderer_status": "passed",
                    "expected_gpu_uuid": expected_identity["uuid"],
                    "observed_physical_index": observed_index,
                    "observed_gpu_uuid": observed_identity["uuid"],
                    "pmon_matched_pid": True,
                    "mapping_matches_expected": mapping_matches,
                    "evidence": {
                        "probe_log": descriptor(probe_log, 0o444),
                        "nvidia_smi_pmon": descriptor(pmon, 0o444),
                        "nvidia_smi_snapshot": descriptor(snapshot, 0o444),
                    },
                }
            )
        schemes[scheme] = {
            "status": "passed" if all(mapping_results) else "failed",
            "rows": rows,
        }
    expected_evidence_count = len(SCHEMES) * gpu_count * 3
    if len(evidence_paths) != expected_evidence_count:
        raise ValueError(f"expected {expected_evidence_count} evidence paths, got {len(evidence_paths)}")

    return {
        "schema_version": 1,
        "kind": "cwam_dlc_robocasa_egl_device_mapping",
        "status": "passed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "host": {
            "hostname": socket.gethostname(),
            "gpu_count": gpu_count,
            "gpu_inventory": inventory,
        },
        "environment": {
            "path": str(environment),
            "ready_marker_sha256": ready_marker_sha256,
            "python": expected_python,
            "mujoco": expected_mujoco,
            "egl_vendor_json": descriptor(vendor, 0o644),
        },
        "probe": {
            "script": descriptor(probe, 0o444),
            "hold_seconds": 12.0,
            "evidence_root": str(evidence_root),
        },
        "schemes": schemes,
        "recommendation": {
            "scheme": "isolated_physical_identity",
            "all_eight_devices_passed": True,
            "cuda_visible_devices_expression": "${physical_index}",
            "mujoco_egl_device_id_expression": "${physical_index}",
            "mujoco_gl": "egl",
            "pyopengl_platform": "egl",
            "reason": ("isolates each evaluator while retaining verified physical EGL identity mapping"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--ready-marker-sha256", required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--probe-sha256", required=True)
    parser.add_argument("--vendor", type=Path, required=True)
    parser.add_argument("--vendor-sha256", required=True)
    parser.add_argument("--expected-python", default="3.11.14")
    parser.add_argument("--expected-mujoco", default="3.3.1")
    parser.add_argument("--gpu-count", type=int, default=GPU_COUNT)
    args = parser.parse_args()
    output = args.output
    if not output.is_absolute() or output != Path(os.path.normpath(output)):
        raise ValueError(f"output path must be canonical absolute: {output}")
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.incomplete-{os.getpid()}")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(temporary)
    payload = build_receipt(
        args.evidence_root,
        environment=args.environment,
        ready_marker_sha256=args.ready_marker_sha256,
        probe=args.probe,
        probe_sha256=args.probe_sha256,
        vendor=args.vendor,
        vendor_sha256=args.vendor_sha256,
        expected_python=args.expected_python,
        expected_mujoco=args.expected_mujoco,
        gpu_count=args.gpu_count,
    )
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.chmod(0o444)
    os.replace(temporary, output)
    print(f"OUTPUT={output}")
    print(f"SIZE={output.stat().st_size}")
    print(f"SHA256={sha256_file(output)}")


if __name__ == "__main__":
    main()
