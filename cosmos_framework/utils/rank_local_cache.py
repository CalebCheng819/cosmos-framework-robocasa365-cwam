# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Configure process-local compiler caches before CUDA initialization."""

from __future__ import annotations

import os
from pathlib import Path
from typing import MutableMapping


def configure_rank_local_compile_cache(
    environ: MutableMapping[str, str] | None = None,
) -> dict[str, str]:
    """Create and export rank-local compiler caches when a root is configured."""

    env = os.environ if environ is None else environ
    root = env.get("CWAM_RANK_CACHE_ROOT", "").strip()
    if not root:
        return {}

    local_rank = env.get("LOCAL_RANK", "0")
    if not local_rank.isdigit():
        raise ValueError(f"LOCAL_RANK must be a non-negative integer, got {local_rank!r}")

    rank_root = Path(root).expanduser() / f"rank_{local_rank}"
    cache_paths = {
        "TRITON_CACHE_DIR": rank_root / "triton",
        "TORCHINDUCTOR_CACHE_DIR": rank_root / "torchinductor",
        "CUDA_CACHE_PATH": rank_root / "cuda",
    }
    for variable, cache_path in cache_paths.items():
        cache_path.mkdir(parents=True, exist_ok=True)
        env[variable] = str(cache_path)

    return {variable: str(path) for variable, path in cache_paths.items()}
