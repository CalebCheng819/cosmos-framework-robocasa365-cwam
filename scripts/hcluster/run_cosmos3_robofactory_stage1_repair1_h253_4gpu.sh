#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export EXPECTED_NODE_TAG="node/gpu-l-lg-cmc-h-h200-0253.host.h.pjlab.org.cn"
export EXPECTED_GPU_COUNT=4
export REPAIR_MODE=repair1

exec /usr/bin/bash "${SCRIPT_DIR}/run_cosmos3_robofactory_stage1_h299_8gpu.sh"
