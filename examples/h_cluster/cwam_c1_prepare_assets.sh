#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

set -Eeuo pipefail

SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${COSMOS_SOURCE:?Set COSMOS_SOURCE to the immutable C1 source snapshot.}"
export EXPERIMENT_LABEL=C1
export CONTROL_ROOT="${CONTROL_ROOT:-/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chengjuntao/experiments/cwam-c1-20260817/control}"
export SOURCE_ROOT="${SOURCE_ROOT:-$COSMOS_SOURCE}"
export B2_INPUT_ROOT="${C1_INPUT_ROOT:-/nvme/cwam-c1/inputs}"
export SECRET_ROOT="${SECRET_ROOT:-/dev/shm/cwam-c1-rclone}"
export AUX_RUNTIME_REMOTE="${AUX_RUNTIME_REMOTE:-eailab-hdd2:eailab/chengjuntao/migrate-20260805_141655/.robocasa_eval_tools}"
export AUX_RUNTIME_OVERLAY_REMOTE="${AUX_RUNTIME_OVERLAY_REMOTE:-eailab-hdd2:eailab/chengjuntao/migrate-20260805_142643/.robocasa_eval_tools}"
export AUX_RUNTIME_ROOT="${AUX_RUNTIME_ROOT:-/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chengjuntao/.robocasa_eval_tools}"
export AUX_RUNTIME_REQUIRED_FILE="${AUX_RUNTIME_REQUIRED_FILE:-$AUX_RUNTIME_ROOT/python/cpython-3.11.14-linux-x86_64-gnu/bin/python3.11}"
export AUX_RUNTIME_REQUIRED_DATA_FILE="${AUX_RUNTIME_REQUIRED_DATA_FILE:-$AUX_RUNTIME_ROOT/python/cpython-3.11.14-linux-x86_64-gnu/lib/python3.11/encodings/__init__.py}"

bash "$SOURCE_DIR/cwam_b2_prepare_assets.sh"

ROBOCASA_PY="/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chengjuntao/envs/robocasa-1.0.1-py311-mj331-rs232ce7-shared/bin/python"
[[ -x "$ROBOCASA_PY" ]] || {
    echo "ERROR: RoboCasa environment remains unavailable after runtime restore: $ROBOCASA_PY" >&2
    exit 1
}
"$ROBOCASA_PY" - <<'PY'
import mujoco
import robocasa
import robosuite

print(
    "C1_ROBOCASA_RUNTIME_READY "
    f"mujoco={mujoco.__version__} "
    f"robocasa={getattr(robocasa, '__version__', 'unknown')} "
    f"robosuite={getattr(robosuite, '__version__', 'unknown')}",
    flush=True,
)
PY
