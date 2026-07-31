#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Re-run the pinned Cosmos3 CUDA-12.8 environment sync on Alibaba PAI DSW.
#
# This is intentionally a retry of the immutable uv.lock resolution.  It
# reuses the existing DSW uv cache, does not change package versions, and
# publishes a DSW-native ready marker only after imports and CUDA checks pass.

set -Eeuo pipefail
umask 022

readonly ITERATION_ID="ITER-2026-07-31-CWAM-DSW974-NATIVE-ENV-RETRY"
readonly EXPECTED_HEAD="09f23119ea92c707207bba55565e7a09d16896a2"
readonly EXPECTED_PYPROJECT_SHA256="f9a476ddb1b66366ad046d93c5a9286b9be1d62caa1f38b2f8a33dfbbbce2d34"
readonly EXPECTED_UV_LOCK_SHA256="6df54ed0928bb3698a161fa9219451b00b03c36d35c221e517c2d23a17d69e7d"
readonly EXPECTED_PYTHON_VERSION="3.13.14"
readonly EXPECTED_TORCH_VERSION="2.10.0+cu128"
readonly EXPECTED_TORCH_CUDA_VERSION="12.8"
readonly EXPECTED_UV_VERSION="0.11.28"

: "${CWAM_DSW_ROOT:=/mnt/workspace/cwam_cosmos3_robocasa365_eval}"
: "${COSMOS_REPO:=$CWAM_DSW_ROOT/inputs/repos/cosmos-framework-env-spec-09f23119}"
: "${COSMOS_ENV:=$CWAM_DSW_ROOT/inputs/envs/cosmos-framework-09f23119-cu128-train}"
: "${COSMOS_UV:=$CWAM_DSW_ROOT/inputs/envs/uv-tool/bin/uv}"
: "${COSMOS_CACHE:=$CWAM_DSW_ROOT/staging/uv-cache}"
: "${COSMOS_PYROOT:=$CWAM_DSW_ROOT/inputs/envs/python}"
: "${COSMOS_LOG_DIR:=$CWAM_DSW_ROOT/logs}"
: "${COSMOS_LOCK:=$CWAM_DSW_ROOT/staging/cosmos-env-retry-09f23119-cu128-train.lock}"

readonly READY_MARKER="$COSMOS_ENV/.cwam_env_ready.json"

sha256_file() {
    sha256sum -- "$1" | awk '{print $1}'
}

require_sha256() {
    local label="$1"
    local path="$2"
    local expected="$3"
    local actual
    actual="$(sha256_file "$path")"
    if [[ "$actual" != "$expected" ]]; then
        echo "ERROR: $label SHA-256 mismatch: expected=$expected actual=$actual path=$path" >&2
        return 1
    fi
}

record_failure() {
    local status=$?
    local line="${BASH_LINENO[0]:-unknown}"
    {
        printf 'iteration_id=%s\n' "$ITERATION_ID"
        printf 'failed_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        printf 'exit_status=%s\n' "$status"
        printf 'line=%s\n' "$line"
        printf 'log=%s\n' "$COSMOS_LOG"
    } >"$FAILURE_MARKER"
    chmod 0444 "$FAILURE_MARKER"
    echo "ERROR: Cosmos DSW environment retry failed: status=$status line=$line log=$COSMOS_LOG" >&2
    exit "$status"
}

for path in "$CWAM_DSW_ROOT" "$COSMOS_REPO" "$COSMOS_ENV" "$COSMOS_CACHE" "$COSMOS_PYROOT"; do
    [[ "$path" == /* ]] || {
        echo "ERROR: path must be absolute: $path" >&2
        exit 1
    }
done

[[ -d "$CWAM_DSW_ROOT" ]] || {
    echo "ERROR: missing DSW root: $CWAM_DSW_ROOT" >&2
    exit 1
}
[[ -d "$COSMOS_REPO" ]] || {
    echo "ERROR: missing pinned Cosmos environment source: $COSMOS_REPO" >&2
    exit 1
}
[[ -x "$COSMOS_UV" ]] || {
    echo "ERROR: missing executable uv: $COSMOS_UV" >&2
    exit 1
}
[[ -f "$COSMOS_REPO/pyproject.toml" && -f "$COSMOS_REPO/uv.lock" ]] || {
    echo "ERROR: pinned pyproject.toml or uv.lock is missing" >&2
    exit 1
}

mkdir -p "$COSMOS_ENV" "$COSMOS_CACHE" "$COSMOS_PYROOT" "$COSMOS_LOG_DIR" "$(dirname "$COSMOS_LOCK")"
COSMOS_LOG="${COSMOS_LOG:-$COSMOS_LOG_DIR/cosmos-env-retry-09f23119-cu128-train-$(date -u +%Y%m%dT%H%M%SZ).log}"
readonly COSMOS_LOG
readonly FAILURE_MARKER="${COSMOS_LOG}.failure"
readonly READY_MARKER_TMP="${READY_MARKER}.tmp.$$"
exec > >(tee -a "$COSMOS_LOG") 2>&1
trap record_failure ERR

echo "ITERATION_ID=$ITERATION_ID"
echo "STARTED_AT_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "LOG=$COSMOS_LOG"

exec 9>"$COSMOS_LOCK"
flock -n 9 || {
    echo "ERROR: another Cosmos DSW environment retry holds $COSMOS_LOCK" >&2
    exit 1
}

[[ ! -e "$READY_MARKER" && ! -L "$READY_MARKER" ]] || {
    echo "ERROR: preserving existing Cosmos ready marker: $READY_MARKER" >&2
    exit 1
}
[[ ! -e "$READY_MARKER_TMP" && ! -L "$READY_MARKER_TMP" ]] || {
    echo "ERROR: preserving existing Cosmos temporary marker: $READY_MARKER_TMP" >&2
    exit 1
}

test "$(git -c safe.directory="$COSMOS_REPO" -C "$COSMOS_REPO" rev-parse HEAD)" = "$EXPECTED_HEAD"
require_sha256 pyproject "$COSMOS_REPO/pyproject.toml" "$EXPECTED_PYPROJECT_SHA256"
require_sha256 uv_lock "$COSMOS_REPO/uv.lock" "$EXPECTED_UV_LOCK_SHA256"

UV_VERSION="$("$COSMOS_UV" --version | awk '{print $2}')"
readonly UV_VERSION
[[ "$UV_VERSION" == "$EXPECTED_UV_VERSION" ]] || {
    echo "ERROR: uv version mismatch: expected=$EXPECTED_UV_VERSION actual=$UV_VERSION" >&2
    exit 1
}
echo "uv $UV_VERSION"
export UV_CACHE_DIR="$COSMOS_CACHE"
export UV_PYTHON_INSTALL_DIR="$COSMOS_PYROOT"
export UV_HTTP_TIMEOUT=1200
export UV_CONCURRENT_DOWNLOADS=8
export UV_CONCURRENT_BUILDS=4
export UV_CONCURRENT_INSTALLS=8
export UV_PROJECT_ENVIRONMENT="$COSMOS_ENV"
export UV_PYTHON_DOWNLOADS=never
export UV_LINK_MODE=hardlink
export UV_INDEX_STRATEGY=unsafe-first-match
export UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple

"$COSMOS_UV" python install "$EXPECTED_PYTHON_VERSION" --install-dir "$COSMOS_PYROOT"
COSMOS_PYTHON="$(
    "$COSMOS_UV" python find "$EXPECTED_PYTHON_VERSION" \
        --managed-python \
        --no-project \
        --resolve-links
)"
readonly COSMOS_PYTHON
"$COSMOS_PYTHON" -VV
export UV_PYTHON="$COSMOS_PYTHON"

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

# Transformer Engine 2.12 probes nvidia/cudart while the CUDA runtime wheel
# stores the same payload under nvidia/cuda_runtime.
COSMOS_SITE_PACKAGES="$(
    "$COSMOS_ENV/bin/python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))'
)"
readonly COSMOS_SITE_PACKAGES
readonly COSMOS_NVIDIA_ROOT="$COSMOS_SITE_PACKAGES/nvidia"
readonly COSMOS_CUDART_ALIAS="$COSMOS_NVIDIA_ROOT/cudart"
if [[ -e "$COSMOS_CUDART_ALIAS" || -L "$COSMOS_CUDART_ALIAS" ]]; then
    [[ -L "$COSMOS_CUDART_ALIAS" && "$(readlink "$COSMOS_CUDART_ALIAS")" == cuda_runtime ]] || {
        echo "ERROR: refusing unexpected cudart compatibility path: $COSMOS_CUDART_ALIAS" >&2
        exit 1
    }
else
    ln -s cuda_runtime "$COSMOS_CUDART_ALIAS"
fi
test -s "$COSMOS_CUDART_ALIAS/lib/libcudart.so.12"

export COSMOS_ENV COSMOS_REPO READY_MARKER READY_MARKER_TMP
export EXPECTED_HEAD EXPECTED_PYPROJECT_SHA256 EXPECTED_UV_LOCK_SHA256
export EXPECTED_PYTHON_VERSION EXPECTED_TORCH_VERSION EXPECTED_TORCH_CUDA_VERSION
export EXPECTED_UV_VERSION
LD_LIBRARY_PATH= PYTHONNOUSERSITE=1 PYTHONPATH="$COSMOS_REPO" "$COSMOS_ENV/bin/python" - <<'PY'
import importlib.metadata as md
import json
import os
import platform
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

assert platform.python_version() == os.environ["EXPECTED_PYTHON_VERSION"], sys.version
assert torch.__version__ == os.environ["EXPECTED_TORCH_VERSION"], torch.__version__
assert torch.version.cuda == os.environ["EXPECTED_TORCH_CUDA_VERSION"], torch.version.cuda
assert torch.cuda.is_available(), "torch.cuda.is_available() is false"
assert torch.cuda.device_count() == 4, torch.cuda.device_count()

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
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "packages": packages,
    "source_head": os.environ["EXPECTED_HEAD"],
    "pyproject_sha256": os.environ["EXPECTED_PYPROJECT_SHA256"],
    "uv_lock_sha256": os.environ["EXPECTED_UV_LOCK_SHA256"],
    "uv": os.environ["EXPECTED_UV_VERSION"],
    "default_index": os.environ["UV_DEFAULT_INDEX"],
    "cudart_alias": "libcudart.so.12",
}
expected_marker_keys = {
    "schema_version",
    "status",
    "python",
    "torch",
    "torch_cuda",
    "cuda_available",
    "packages",
    "source_head",
    "pyproject_sha256",
    "uv_lock_sha256",
    "uv",
    "default_index",
    "cudart_alias",
}
assert set(payload) == expected_marker_keys, sorted(set(payload) ^ expected_marker_keys)
target = Path(os.environ["READY_MARKER"])
temporary = Path(os.environ["READY_MARKER_TMP"])
with temporary.open("x") as stream:
    stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
temporary.chmod(0o444)
temporary.replace(target)
print(json.dumps(payload, indent=2, sort_keys=True))
print("ENV_READY", target)
PY

chmod 0444 "$READY_MARKER"
echo "READY_MARKER=$READY_MARKER"
echo "READY_MARKER_SHA256=$(sha256_file "$READY_MARKER")"
stat -c 'READY_MARKER_MODE=%a READY_MARKER_BYTES=%s READY_MARKER_MTIME=%y' "$READY_MARKER"
du -sh "$COSMOS_ENV" "$COSMOS_CACHE"
echo "COMPLETED_AT_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
