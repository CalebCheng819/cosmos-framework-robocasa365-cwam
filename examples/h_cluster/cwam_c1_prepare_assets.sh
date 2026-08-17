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

exec bash "$SOURCE_DIR/cwam_b2_prepare_assets.sh"
