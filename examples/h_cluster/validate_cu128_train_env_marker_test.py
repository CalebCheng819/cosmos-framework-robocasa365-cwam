# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import copy
import os
import tempfile
import unittest
from pathlib import Path

from examples.h_cluster.validate_cu128_train_env_marker import (
    EXPECTED_AV,
    EXPECTED_CUDART_ALIAS_TARGET,
    EXPECTED_CUDART_MARKER,
    EXPECTED_CUDART_SONAME,
    EXPECTED_PYPROJECT_SHA256,
    EXPECTED_SOURCE_HEAD,
    EXPECTED_TORCH,
    EXPECTED_TORCH_CUDA,
    EXPECTED_TORCHCODEC,
    EXPECTED_UV_CONCURRENCY,
    EXPECTED_UV_LOCK_SHA256,
    validate_cudart_compatibility,
    validate_marker,
)


def _valid_marker() -> dict[str, object]:
    return {
        "status": "ready",
        "torch": EXPECTED_TORCH,
        "torch_cuda": EXPECTED_TORCH_CUDA,
        "av": EXPECTED_AV,
        "source_head": EXPECTED_SOURCE_HEAD,
        "pyproject_sha256": EXPECTED_PYPROJECT_SHA256,
        "uv_lock_sha256": EXPECTED_UV_LOCK_SHA256,
        "uv_index_strategy": "unsafe-first-match",
        "uv_concurrency": dict(EXPECTED_UV_CONCURRENCY),
        "cudart_compatibility": dict(EXPECTED_CUDART_MARKER),
        "packages": {"pydantic": "2.12.5", "torchcodec": EXPECTED_TORCHCODEC},
    }


class ValidateCu128TrainEnvMarkerTest(unittest.TestCase):
    def test_accepts_exact_contract(self) -> None:
        marker = _valid_marker()
        payload = validate_marker(
            marker,
            runtime_torch=EXPECTED_TORCH,
            runtime_torch_cuda=EXPECTED_TORCH_CUDA,
            runtime_pydantic="2.12.5",
        )
        self.assertEqual(payload["runtime_torch"], EXPECTED_TORCH)

    def test_rejects_stale_source(self) -> None:
        marker = copy.deepcopy(_valid_marker())
        marker["source_head"] = "stale"
        with self.assertRaisesRegex(ValueError, "source_head"):
            validate_marker(
                marker,
                runtime_torch=EXPECTED_TORCH,
                runtime_torch_cuda=EXPECTED_TORCH_CUDA,
                runtime_pydantic="2.12.5",
            )

    def test_rejects_missing_pydantic_provenance(self) -> None:
        marker = copy.deepcopy(_valid_marker())
        marker["packages"] = {}
        with self.assertRaisesRegex(ValueError, "pydantic"):
            validate_marker(
                marker,
                runtime_torch=EXPECTED_TORCH,
                runtime_torch_cuda=EXPECTED_TORCH_CUDA,
                runtime_pydantic="2.12.5",
            )

    def test_rejects_missing_torchcodec_provenance(self) -> None:
        marker = copy.deepcopy(_valid_marker())
        marker["packages"].pop("torchcodec")
        with self.assertRaisesRegex(ValueError, "torchcodec"):
            validate_marker(
                marker,
                runtime_torch=EXPECTED_TORCH,
                runtime_torch_cuda=EXPECTED_TORCH_CUDA,
                runtime_pydantic="2.12.5",
            )

    def test_rejects_runtime_torch_mismatch(self) -> None:
        marker = _valid_marker()
        with self.assertRaisesRegex(ValueError, "runtime torch"):
            validate_marker(
                marker,
                runtime_torch="2.10.0",
                runtime_torch_cuda=EXPECTED_TORCH_CUDA,
                runtime_pydantic="2.12.5",
            )

    def test_rejects_missing_cudart_marker(self) -> None:
        marker = _valid_marker()
        marker.pop("cudart_compatibility")
        with self.assertRaisesRegex(ValueError, "cudart_compatibility"):
            validate_marker(
                marker,
                runtime_torch=EXPECTED_TORCH,
                runtime_torch_cuda=EXPECTED_TORCH_CUDA,
                runtime_pydantic="2.12.5",
            )


class ValidateCudartCompatibilityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.environment = Path(self.temporary.name) / "environment"
        self.nvidia_root = self.environment / "lib/python3.13/site-packages/nvidia"
        runtime_library = self.nvidia_root / EXPECTED_CUDART_ALIAS_TARGET / "lib" / EXPECTED_CUDART_SONAME
        runtime_library.parent.mkdir(parents=True)
        runtime_library.write_bytes(b"cudart")
        os.symlink(EXPECTED_CUDART_ALIAS_TARGET, self.nvidia_root / "cudart")

    def test_accepts_exact_relative_alias(self) -> None:
        result = validate_cudart_compatibility(self.environment)
        self.assertEqual(result["alias_target"], EXPECTED_CUDART_ALIAS_TARGET)
        self.assertEqual(Path(result["runtime_library"]).read_bytes(), b"cudart")

    def test_rejects_missing_or_wrong_alias(self) -> None:
        alias = self.nvidia_root / "cudart"
        alias.unlink()
        with self.assertRaisesRegex(ValueError, "missing or not a symlink"):
            validate_cudart_compatibility(self.environment)

        os.symlink(".", alias)
        with self.assertRaisesRegex(ValueError, "target mismatch"):
            validate_cudart_compatibility(self.environment)

    def test_rejects_dangling_alias(self) -> None:
        alias = self.nvidia_root / "cudart"
        alias.unlink()
        os.symlink(EXPECTED_CUDART_ALIAS_TARGET, alias)
        runtime_root = self.nvidia_root / EXPECTED_CUDART_ALIAS_TARGET
        for candidate in sorted(runtime_root.rglob("*"), reverse=True):
            candidate.unlink() if candidate.is_file() else candidate.rmdir()
        runtime_root.rmdir()
        with self.assertRaisesRegex(ValueError, "dangling or unreadable"):
            validate_cudart_compatibility(self.environment)


if __name__ == "__main__":
    unittest.main()
