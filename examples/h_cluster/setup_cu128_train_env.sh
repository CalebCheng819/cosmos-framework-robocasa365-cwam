#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Build the pinned Cosmos3 CUDA-12.8 training environment on personal GPFS.
# This script intentionally does not install the mutable project itself; jobs
# import their immutable source snapshot through PYTHONPATH.

set -Eeuo pipefail
umask 022

: "${COSMOS_BASE:=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chengjuntao}"
: "${COSMOS_REPO:=$COSMOS_BASE/code/cosmos-framework-robocasa365-cwam}"
: "${COSMOS_ENV:=$COSMOS_BASE/envs/cosmos-framework-09f23119-cu128-train}"
: "${COSMOS_TOOLS:=$COSMOS_BASE/.cosmos3_dcp_tools}"
: "${COSMOS_UV:=$COSMOS_TOOLS/uv-bootstrap/bin/uv}"
: "${COSMOS_CACHE:=$COSMOS_TOOLS/uv-cache}"
: "${COSMOS_PYROOT:=$COSMOS_TOOLS/python}"
: "${EXPECTED_HEAD:=09f23119ea92c707207bba55565e7a09d16896a2}"
: "${EXPECTED_PYPROJECT_SHA256:=f9a476ddb1b66366ad046d93c5a9286b9be1d62caa1f38b2f8a33dfbbbce2d34}"
: "${EXPECTED_UV_LOCK_SHA256:=6df54ed0928bb3698a161fa9219451b00b03c36d35c221e517c2d23a17d69e7d}"

mkdir -p "$COSMOS_BASE/envs" "$COSMOS_TOOLS/logs" "$COSMOS_CACHE" "$COSMOS_PYROOT"
COSMOS_LOG="${COSMOS_LOG:-$COSMOS_TOOLS/logs/env-sync-09f23119-cu128-train-$(date -u +%Y%m%dT%H%M%SZ).log}"
exec > >(tee -a "$COSMOS_LOG") 2>&1
echo "LOG=$COSMOS_LOG"
date -Is

exec 9>"$COSMOS_TOOLS/env-sync-09f23119-cu128-train.lock"
flock -n 9 || {
    echo "ERROR: another cu128 environment sync holds the lock" >&2
    exit 1
}

# H workspaces require the platform-provided proxy bootstrap for external
# package indexes. Do not print the resulting proxy environment values.
source <(curl -fsSL http://deploy.i.h.pjlab.org.cn/infra/scripts/setup_proxy.sh) >/dev/null

"$COSMOS_UV" --version
test "$(git -c safe.directory="$COSMOS_REPO" -C "$COSMOS_REPO" rev-parse HEAD)" = "$EXPECTED_HEAD"
test "$(sha256sum "$COSMOS_REPO/pyproject.toml" | awk '{print $1}')" = "$EXPECTED_PYPROJECT_SHA256"
test "$(sha256sum "$COSMOS_REPO/uv.lock" | awk '{print $1}')" = "$EXPECTED_UV_LOCK_SHA256"

export UV_CACHE_DIR="$COSMOS_CACHE"
export UV_PYTHON_INSTALL_DIR="$COSMOS_PYROOT"
export UV_HTTP_TIMEOUT=600
: "${UV_CONCURRENT_DOWNLOADS:=8}"
: "${UV_CONCURRENT_BUILDS:=4}"
: "${UV_CONCURRENT_INSTALLS:=8}"
export UV_CONCURRENT_DOWNLOADS UV_CONCURRENT_BUILDS UV_CONCURRENT_INSTALLS
"$COSMOS_UV" python install 3.13.14 --install-dir "$COSMOS_PYROOT"
COSMOS_PYTHON="$("$COSMOS_UV" python find 3.13.14 --managed-python --no-project --resolve-links)"
"$COSMOS_PYTHON" -VV

export UV_PROJECT_ENVIRONMENT="$COSMOS_ENV"
export UV_PYTHON="$COSMOS_PYTHON"
export UV_PYTHON_DOWNLOADS=never
export UV_LINK_MODE=hardlink
# H exposes a trusted internal package index before its PyPI mirror. Prefer
# compatible versions from the first index, but allow the mirror when the
# internal index has only an incompatible build requirement (for example,
# evdev 1.9.3 requires setuptools>=77 while the first index has 75.3.2).
export UV_INDEX_STRATEGY=unsafe-first-match
cd "$COSMOS_REPO"
"$COSMOS_UV" sync \
    --frozen \
    --extra=train \
    --group=cu128-train \
    --no-dev \
    --no-install-project
"$COSMOS_UV" sync \
    --frozen \
    --extra=train \
    --group=cu128-train \
    --no-dev \
    --no-install-project \
    --offline \
    --check

# Transformer Engine 2.12 probes ``nvidia/cudart`` while the CUDA 12.8
# runtime wheel installs its payload as ``nvidia/cuda_runtime``.  Keep one
# deterministic relative alias and validate it before creating the marker.
COSMOS_SITE_PACKAGES="$(
    "$COSMOS_ENV/bin/python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))'
)"
COSMOS_NVIDIA_ROOT="$COSMOS_SITE_PACKAGES/nvidia"
COSMOS_CUDART_ALIAS="$COSMOS_NVIDIA_ROOT/cudart"
if [[ -e "$COSMOS_CUDART_ALIAS" || -L "$COSMOS_CUDART_ALIAS" ]]; then
    [[ -L "$COSMOS_CUDART_ALIAS" && "$(readlink "$COSMOS_CUDART_ALIAS")" == cuda_runtime ]] ||
        {
            echo "ERROR: refusing an unexpected cudart compatibility path: $COSMOS_CUDART_ALIAS" >&2
            exit 1
        }
else
    ln -s cuda_runtime "$COSMOS_CUDART_ALIAS"
fi
test -s "$COSMOS_CUDART_ALIAS/lib/libcudart.so.12"

export COSMOS_ENV COSMOS_REPO EXPECTED_HEAD EXPECTED_PYPROJECT_SHA256 EXPECTED_UV_LOCK_SHA256
LD_LIBRARY_PATH= PYTHONNOUSERSITE=1 PYTHONPATH="$COSMOS_REPO" "$COSMOS_ENV/bin/python" - <<'PY'
import importlib.metadata as md
import json
import os
import sys
from pathlib import Path

import av
import cosmos_framework
import hydra
import omegaconf
import pyarrow
import pydantic
import torch
import transformer_engine
import transformer_engine.pytorch

assert sys.version_info[:2] == (3, 13), sys.version
assert torch.__version__ == "2.10.0+cu128", torch.__version__
assert torch.version.cuda == "12.8", torch.version.cuda

package_names = [
    "torch",
    "torchvision",
    "torchcodec",
    "torchao",
    "triton",
    "flash-attn",
    "flash-attn-3-nv",
    "natten",
    "transformer-engine",
    "lerobot",
    "pyarrow",
    "pydantic",
]
packages = {name: md.version(name) for name in package_names}
payload = {
    "schema_version": 1,
    "status": "ready",
    "python": sys.version,
    "executable": sys.executable,
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda_available_in_workspace": torch.cuda.is_available(),
    "pyarrow": pyarrow.__version__,
    "av": av.__version__,
    "packages": packages,
    "source_head": os.environ["EXPECTED_HEAD"],
    "pyproject_sha256": os.environ["EXPECTED_PYPROJECT_SHA256"],
    "uv_lock_sha256": os.environ["EXPECTED_UV_LOCK_SHA256"],
    "uv_index_strategy": os.environ["UV_INDEX_STRATEGY"],
    "uv_concurrency": {
        "downloads": int(os.environ["UV_CONCURRENT_DOWNLOADS"]),
        "builds": int(os.environ["UV_CONCURRENT_BUILDS"]),
        "installs": int(os.environ["UV_CONCURRENT_INSTALLS"]),
    },
    "cudart_compatibility": {
        "alias_target": "cuda_runtime",
        "runtime_soname": "libcudart.so.12",
        "transformer_engine_import": True,
    },
}
target = Path(os.environ["COSMOS_ENV"]) / ".cwam_env_ready.json"
temporary = target.with_suffix(".json.tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
temporary.replace(target)
print(json.dumps(payload, indent=2, sort_keys=True))
print("ENV_READY", target)
PY

du -sh "$COSMOS_ENV" "$COSMOS_CACHE"
date -Is
