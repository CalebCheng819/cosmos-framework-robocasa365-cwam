#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Fail-closed validation for the pinned H-cluster CUDA 12.8 environment."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

EXPECTED_SOURCE_HEAD = "09f23119ea92c707207bba55565e7a09d16896a2"
EXPECTED_PYPROJECT_SHA256 = "f9a476ddb1b66366ad046d93c5a9286b9be1d62caa1f38b2f8a33dfbbbce2d34"
EXPECTED_UV_LOCK_SHA256 = "6df54ed0928bb3698a161fa9219451b00b03c36d35c221e517c2d23a17d69e7d"
EXPECTED_TORCH = "2.10.0+cu128"
EXPECTED_TORCH_CUDA = "12.8"
EXPECTED_AV = "16.1.0"
EXPECTED_TORCHCODEC = "0.10.0+cu128"
EXPECTED_UV_CONCURRENCY = {"downloads": 8, "builds": 4, "installs": 8}
EXPECTED_CUDART_ALIAS_TARGET = "cuda_runtime"
EXPECTED_CUDART_SONAME = "libcudart.so.12"
EXPECTED_CUDART_MARKER = {
    "alias_target": EXPECTED_CUDART_ALIAS_TARGET,
    "runtime_soname": EXPECTED_CUDART_SONAME,
    "transformer_engine_import": True,
}


def validate_marker(
    marker: Mapping[str, Any],
    *,
    runtime_torch: str,
    runtime_torch_cuda: str | None,
    runtime_pydantic: str,
) -> dict[str, Any]:
    expected_scalars = {
        "status": "ready",
        "torch": EXPECTED_TORCH,
        "torch_cuda": EXPECTED_TORCH_CUDA,
        "av": EXPECTED_AV,
        "source_head": EXPECTED_SOURCE_HEAD,
        "pyproject_sha256": EXPECTED_PYPROJECT_SHA256,
        "uv_lock_sha256": EXPECTED_UV_LOCK_SHA256,
        "uv_index_strategy": "unsafe-first-match",
    }
    for key, expected in expected_scalars.items():
        actual = marker.get(key)
        if actual != expected:
            raise ValueError(f"environment marker {key!r} mismatch: expected={expected!r}, actual={actual!r}")

    if marker.get("uv_concurrency") != EXPECTED_UV_CONCURRENCY:
        raise ValueError(
            "environment marker uv_concurrency mismatch: "
            f"expected={EXPECTED_UV_CONCURRENCY!r}, actual={marker.get('uv_concurrency')!r}"
        )
    if marker.get("cudart_compatibility") != EXPECTED_CUDART_MARKER:
        raise ValueError(
            "environment marker cudart_compatibility mismatch: "
            f"expected={EXPECTED_CUDART_MARKER!r}, "
            f"actual={marker.get('cudart_compatibility')!r}"
        )
    if runtime_torch != marker["torch"] or runtime_torch_cuda != marker["torch_cuda"]:
        raise ValueError(
            "runtime torch does not match marker: "
            f"runtime=({runtime_torch!r}, {runtime_torch_cuda!r}), "
            f"marker=({marker['torch']!r}, {marker['torch_cuda']!r})"
        )

    packages = marker.get("packages")
    if not isinstance(packages, Mapping):
        raise ValueError("environment marker packages must be a mapping")
    marker_pydantic = packages.get("pydantic")
    if marker_pydantic != runtime_pydantic:
        raise ValueError(
            f"runtime pydantic does not match marker: runtime={runtime_pydantic!r}, marker={marker_pydantic!r}"
        )
    marker_torchcodec = packages.get("torchcodec")
    if marker_torchcodec != EXPECTED_TORCHCODEC:
        raise ValueError(
            f"environment marker torchcodec mismatch: expected={EXPECTED_TORCHCODEC!r}, actual={marker_torchcodec!r}"
        )

    return {
        "environment_marker": dict(marker),
        "runtime_pydantic": runtime_pydantic,
        "runtime_torch": runtime_torch,
        "runtime_torch_cuda": runtime_torch_cuda,
    }


def validate_cudart_compatibility(environment: Path) -> dict[str, str]:
    """Validate the CUDA 12 compatibility alias required by TE 2.12.

    Transformer Engine 2.12 probes ``nvidia/cudart`` while the CUDA 12.8
    runtime wheel installs the same payload under ``nvidia/cuda_runtime``.
    The pinned environment therefore carries one relative alias.  Validate
    both the alias itself and its exact target before any model import.
    """

    if not environment.is_absolute():
        raise ValueError(f"environment path must be absolute: {environment}")
    site_packages = sorted(environment.glob("lib/python*/site-packages"))
    if len(site_packages) != 1 or not site_packages[0].is_dir():
        raise ValueError(f"expected one Python site-packages tree under {environment}, got {site_packages}")
    nvidia_root = site_packages[0] / "nvidia"
    alias = nvidia_root / "cudart"
    expected_target = nvidia_root / EXPECTED_CUDART_ALIAS_TARGET
    if not alias.is_symlink():
        raise ValueError(f"required cudart compatibility alias is missing or not a symlink: {alias}")
    observed_target = os.readlink(alias)
    if observed_target != EXPECTED_CUDART_ALIAS_TARGET:
        raise ValueError(
            "cudart compatibility alias target mismatch: "
            f"expected={EXPECTED_CUDART_ALIAS_TARGET!r}, actual={observed_target!r}"
        )
    try:
        resolved_alias = alias.resolve(strict=True)
        resolved_expected = expected_target.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"cudart compatibility alias is dangling or unreadable: {alias}") from error
    if resolved_alias != resolved_expected or not resolved_expected.is_dir():
        raise ValueError(
            f"cudart compatibility alias escapes its pinned target: "
            f"alias={resolved_alias}, expected={resolved_expected}"
        )
    runtime_library = resolved_expected / "lib" / EXPECTED_CUDART_SONAME
    if runtime_library.is_symlink() or not runtime_library.is_file():
        raise ValueError(f"pinned CUDA runtime library is missing or unsafe: {runtime_library}")
    return {
        "alias": str(alias),
        "alias_target": observed_target,
        "resolved_target": str(resolved_expected),
        "runtime_library": str(runtime_library),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    import pydantic
    import torch
    import yaml

    marker_path = args.environment / ".cwam_env_ready.json"
    marker = json.loads(marker_path.read_text())
    payload = validate_marker(
        marker,
        runtime_torch=torch.__version__,
        runtime_torch_cuda=torch.version.cuda,
        runtime_pydantic=pydantic.__version__,
    )
    payload["cudart_compatibility"] = validate_cudart_compatibility(args.environment)
    payload.update(
        {
            "environment": str(args.environment),
            "marker_path": str(marker_path),
            "runtime_cuda_available": torch.cuda.is_available(),
            "runtime_cuda_device_count": torch.cuda.device_count(),
            "runtime_pyyaml": yaml.__version__,
        }
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
