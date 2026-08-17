#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Restore the managed CPython executable referenced by the existing pinned
# Cosmos environment. Package resolution is not changed.

set -Eeuo pipefail

: "${PERSONAL_ROOT:=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chengjuntao}"
: "${COSMOS_ENV:=$PERSONAL_ROOT/envs/cosmos-framework-09f23119-cu128-train}"
: "${PYTHON_ROOT:=$PERSONAL_ROOT/.cosmos3_dcp_tools/python}"
: "${PYTHON_VERSION:=3.13.14}"

UV_BIN="$COSMOS_ENV/bin/uv"
PYTHON_BIN="$COSMOS_ENV/bin/python"

[[ -x "$UV_BIN" ]] || {
    echo "ERROR: executable uv is missing from $COSMOS_ENV" >&2
    exit 1
}

if [[ ! -x "$PYTHON_BIN" ]]; then
    mkdir -p "$PYTHON_ROOT"
    source <(curl -fsSL http://deploy.i.h.pjlab.org.cn/infra/scripts/setup_proxy.sh) >/dev/null
    UV_PYTHON_INSTALL_DIR="$PYTHON_ROOT" "$UV_BIN" python install "$PYTHON_VERSION"
fi

[[ -x "$PYTHON_BIN" ]] || {
    echo "ERROR: managed Python repair did not restore $PYTHON_BIN" >&2
    exit 1
}

"$PYTHON_BIN" -VV
PYTHONNOUSERSITE=1 "$PYTHON_BIN" - <<'PY'
import sys

import pytest
import torch

assert sys.version_info[:3] == (3, 13, 14), sys.version
assert torch.__version__ == "2.10.0+cu128", torch.__version__
assert torch.version.cuda == "12.8", torch.version.cuda
assert torch.cuda.is_available(), "CUDA is not available inside the B2 worker"
assert torch.cuda.device_count() == 8, torch.cuda.device_count()
print(f"RUNTIME_READY python={sys.version.split()[0]} torch={torch.__version__} gpus={torch.cuda.device_count()}")
PY
