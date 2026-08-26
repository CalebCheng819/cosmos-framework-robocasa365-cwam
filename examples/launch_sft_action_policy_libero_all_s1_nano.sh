#!/usr/bin/env bash
# Launch the CWAM S1 full-context policy on the four local LIBERO suites.

set -Eeuo pipefail

TOML_FILE="examples/toml/sft_config/action_policy_libero_all_s1_nano.toml"
: "${BASE_CHECKPOINT_PATH:=examples/checkpoints/Cosmos3-Nano}"
: "${LIBERO_SUITE_DIR_SUFFIX:=_no_noops_lerobot}"
export LIBERO_ROOT="${LIBERO_ROOT:-}"
export LIBERO_SUITE_DIR_SUFFIX

EXTRA_DATASET_CHECK='for _s in libero_spatial libero_object libero_goal libero_10; do _d="$LIBERO_ROOT/${_s}${LIBERO_SUITE_DIR_SUFFIX}"; [[ -f "$_d/meta/info.json" ]] || { echo "ERROR: missing LIBERO suite: $_d" >&2; exit 1; }; done'
TAIL_OVERRIDES=(
    "job.name=${RUN_NAME:-action_policy_libero_all_s1_nano}"
    ${EXTRA_TAIL_OVERRIDES:-}
)

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
