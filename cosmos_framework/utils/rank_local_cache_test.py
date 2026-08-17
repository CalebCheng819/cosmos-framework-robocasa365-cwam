# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

from pathlib import Path

import pytest

from cosmos_framework.utils.rank_local_cache import configure_rank_local_compile_cache


def test_configure_rank_local_compile_cache_is_opt_in(tmp_path: Path) -> None:
    env = {"LOCAL_RANK": "3"}

    assert configure_rank_local_compile_cache(env) == {}
    assert env == {"LOCAL_RANK": "3"}


def test_configure_rank_local_compile_cache_uses_local_rank(tmp_path: Path) -> None:
    env = {"CWAM_RANK_CACHE_ROOT": str(tmp_path), "LOCAL_RANK": "3"}

    configured = configure_rank_local_compile_cache(env)

    expected_root = tmp_path / "rank_3"
    assert configured == {
        "TRITON_CACHE_DIR": str(expected_root / "triton"),
        "TORCHINDUCTOR_CACHE_DIR": str(expected_root / "torchinductor"),
        "CUDA_CACHE_PATH": str(expected_root / "cuda"),
    }
    for variable, path in configured.items():
        assert env[variable] == path
        assert Path(path).is_dir()


def test_configure_rank_local_compile_cache_rejects_invalid_rank(tmp_path: Path) -> None:
    env = {"CWAM_RANK_CACHE_ROOT": str(tmp_path), "LOCAL_RANK": "rank-zero"}

    with pytest.raises(ValueError, match="LOCAL_RANK"):
        configure_rank_local_compile_cache(env)
