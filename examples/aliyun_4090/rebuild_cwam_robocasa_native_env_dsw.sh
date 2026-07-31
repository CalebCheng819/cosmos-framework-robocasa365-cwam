#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Build a fresh, relocatable RoboCasa evaluator environment on DSW.
#
# The audited requirements contain torchcodec==0.5.  uv's CUDA backend selects
# the PEP 440 local variant 0.5+cu128, so this script explicitly reinstalls the
# verified official PyPI 0.5 wheel before enforcing the exact 163-distribution
# inventory.  It never mutates or publishes over the failed canonical venv.

set -Eeuo pipefail
umask 022

readonly ITERATION_ID="ITER-2026-07-31-CWAM-DSW974-ROBOCASA-FRESH-REBUILD"
readonly EXPECTED_UV_VERSION="0.11.28"
readonly EXPECTED_PYTHON_VERSION="3.11.14"
readonly EXPECTED_TORCH_VERSION="2.7.1+cu128"
readonly EXPECTED_TORCH_CUDA_VERSION="12.8"
readonly EXPECTED_MUJOCO_VERSION="3.3.1"
readonly EXPECTED_ROBOCASA_VERSION="1.0.1"
readonly EXPECTED_ROBOSUITE_VERSION="1.5.2"
readonly EXPECTED_DISTRIBUTION_COUNT="163"
readonly EXPECTED_ROBOCASA_COMMIT="29f7ce8814c1547f5af762a0997fbd4b64848dd7"
readonly EXPECTED_ROBOSUITE_COMMIT="232ce7d4a6ed89c949a9aba024a05c8c32fdd08b"
readonly EXPECTED_DEMIAN_COMMIT="2600ece5bf0eeacf010a95f084dcda2de6d30d59"
readonly EXPECTED_ORIGIN_MANIFEST_SHA256="3a8c7fa6f857b69eb40f2bfd46b1d41d1f71674e38f8d224fe36f5f6fdd556b3"
readonly EXPECTED_ORIGIN_REQUIREMENTS_SHA256="5ac364c3af9a2e0f7873e01868e8e0010ed0c59dc2ea59ea7a86f3152d5d6181"
readonly EXPECTED_RELOCATED_REQUIREMENTS_SHA256="a1954ec8638f8f156eb8980a77585a4d51f28f78292fb6476498f021ca9c2f89"
readonly EXPECTED_FAILED_ENV_RECEIPT_SHA256="b80fa773de25318c4a147f790d63ebcf4fc7837d9636406975aaab5ff704c1bf"
readonly EXPECTED_PLAIN_WHEEL_SHA256="00252c2a138c655851cc7aa6e7ad6455c6513c9b2f93cec74563cba83196be8a"
readonly EXPECTED_PLAIN_WHEEL_SIZE="1374385"
readonly EXPECTED_WHEEL_RECEIPT_SHA256="1b16cc5419b7eb8558d5c0bd6dc4099493df629d50756ad3116b3c1da47f5ce4"
readonly PACKAGE_SOURCE_NOTE="exact versions rebuilt with Alibaba PyPI mirror and PyTorch cu128 index; torchcodec==0.5 overridden by verified official PyPI cp311 wheel; torchcodec_relay_receipt_sha256=1b16cc5419b7eb8558d5c0bd6dc4099493df629d50756ad3116b3c1da47f5ce4"

: "${CWAM_DSW_ROOT:=/mnt/workspace/cwam_cosmos3_robocasa365_eval}"
: "${ROBO_STAGING_ENV:?Set ROBO_STAGING_ENV to a unique new CPFS staging path}"
: "${ROBO_BUILD_RECEIPT:?Set ROBO_BUILD_RECEIPT to a unique control receipt path}"
: "${ROBO_BUILD_LOG:=$CWAM_DSW_ROOT/logs/robocasa-fresh-rebuild-$(date -u +%Y%m%dT%H%M%SZ).log}"
: "${ROBO_PLAIN_WHEEL:=/tmp/cwam_cosmos3_custom_wheels-20260731T080647Z/torchcodec-0.5-cp311-cp311-manylinux_2_28_x86_64.whl}"

readonly CANONICAL_FAILED_ENV="$CWAM_DSW_ROOT/inputs/envs/robocasa-1.0.1-py311-mj331-rs232ce7-shared"
readonly UV="$CWAM_DSW_ROOT/inputs/envs/uv-tool/bin/uv"
readonly PYTHON="$CWAM_DSW_ROOT/inputs/envs/python/cpython-3.11.14-linux-x86_64-gnu/bin/python3.11"
readonly UV_CACHE="$CWAM_DSW_ROOT/staging/uv-cache"
readonly ORIGIN_MANIFEST="$CWAM_DSW_ROOT/inputs/envs/manifests/robocasa-1.0.1-py311-mj331-rs232ce7-shared-python-environment-manifest.json"
readonly ORIGIN_REQUIREMENTS="$CWAM_DSW_ROOT/inputs/envs/manifests/robocasa-1.0.1-py311-mj331-rs232ce7-requirements.txt"
readonly RELOCATED_REQUIREMENTS="$CWAM_DSW_ROOT/inputs/envs/manifests/robocasa-1.0.1-py311-mj331-rs232ce7-requirements.dsw-relocated.txt"
readonly FAILED_ENV_RECEIPT="$CWAM_DSW_ROOT/control/receipts/robocasa-failed-env-20260731T082526Z-inventory.json"
readonly WHEEL_RECEIPT="$CWAM_DSW_ROOT/control/receipts/robocasa_torchcodec_plain_wheel_relay_receipt_20260731T084357Z.json"
readonly ROBOCASA_REPO="$CWAM_DSW_ROOT/inputs/repos/robocasa-29f7ce8814c1"
readonly ROBOSUITE_REPO="$CWAM_DSW_ROOT/inputs/repos/robosuite-232ce7d4a6ed"
readonly DEMIAN_REPO="$CWAM_DSW_ROOT/inputs/demian-2600ece5bf0e"
readonly EVALUATOR="$DEMIAN_REPO/demian_wam/eval_utils/run_robocasa_eval.py"
readonly READY_MARKER="$ROBO_STAGING_ENV/.cwam_env_ready.json"
readonly FAILURE_RECEIPT="${ROBO_BUILD_LOG}.failure"

sha256_file() {
    sha256sum -- "$1" | awk '{print $1}'
}

require_sha256() {
    local label="$1"
    local path="$2"
    local expected="$3"
    local actual
    actual="$(sha256_file "$path")"
    [[ "$actual" == "$expected" ]] || {
        echo "ERROR: $label SHA mismatch: expected=$expected actual=$actual path=$path" >&2
        return 1
    }
}

record_failure() {
    local status=$?
    local line="${BASH_LINENO[0]:-unknown}"
    {
        printf 'iteration_id=%s\n' "$ITERATION_ID"
        printf 'failed_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        printf 'exit_status=%s\n' "$status"
        printf 'line=%s\n' "$line"
        printf 'staging_environment=%s\n' "$ROBO_STAGING_ENV"
        printf 'log=%s\n' "$ROBO_BUILD_LOG"
    } >"$FAILURE_RECEIPT"
    chmod 0444 "$FAILURE_RECEIPT"
    echo "ERROR: fresh RoboCasa build failed: status=$status line=$line log=$ROBO_BUILD_LOG" >&2
    exit "$status"
}

for path in \
    "$CWAM_DSW_ROOT" \
    "$ROBO_STAGING_ENV" \
    "$ROBO_BUILD_RECEIPT" \
    "$ROBO_BUILD_LOG" \
    "$ROBO_PLAIN_WHEEL"; do
    [[ "$path" == /* ]] || {
        echo "ERROR: path must be absolute: $path" >&2
        exit 1
    }
done
[[ "$ROBO_STAGING_ENV" == "$CWAM_DSW_ROOT/staging/env-builds/"* ]] || {
    echo "ERROR: staging environment must be under $CWAM_DSW_ROOT/staging/env-builds" >&2
    exit 1
}
[[ "$ROBO_BUILD_RECEIPT" == "$CWAM_DSW_ROOT/control/receipts/"* ]] || {
    echo "ERROR: build receipt must be under $CWAM_DSW_ROOT/control/receipts" >&2
    exit 1
}
[[ "$ROBO_BUILD_LOG" == "$CWAM_DSW_ROOT/logs/"* ]] || {
    echo "ERROR: build log must be under $CWAM_DSW_ROOT/logs" >&2
    exit 1
}

mkdir -p \
    "$(dirname "$ROBO_STAGING_ENV")" \
    "$(dirname "$ROBO_BUILD_RECEIPT")" \
    "$(dirname "$ROBO_BUILD_LOG")"
for path in "$ROBO_STAGING_ENV" "$ROBO_BUILD_RECEIPT" "$ROBO_BUILD_LOG" "$FAILURE_RECEIPT"; do
    [[ ! -e "$path" && ! -L "$path" ]] || {
        echo "ERROR: refusing existing output path: $path" >&2
        exit 1
    }
done
[[ -d "$CANONICAL_FAILED_ENV" ]] || {
    echo "ERROR: canonical failed environment is absent" >&2
    exit 1
}
[[ ! -e "$CANONICAL_FAILED_ENV/.cwam_env_ready.json" &&
    ! -L "$CANONICAL_FAILED_ENV/.cwam_env_ready.json" ]] || {
    echo "ERROR: canonical failed environment unexpectedly has a ready marker" >&2
    exit 1
}

for path in \
    "$UV" \
    "$PYTHON" \
    "$ORIGIN_MANIFEST" \
    "$ORIGIN_REQUIREMENTS" \
    "$RELOCATED_REQUIREMENTS" \
    "$FAILED_ENV_RECEIPT" \
    "$WHEEL_RECEIPT" \
    "$ROBO_PLAIN_WHEEL" \
    "$EVALUATOR"; do
    [[ -f "$path" ]] || {
        echo "ERROR: required file is absent: $path" >&2
        exit 1
    }
done
for repo in "$ROBOCASA_REPO" "$ROBOSUITE_REPO" "$DEMIAN_REPO"; do
    [[ -d "$repo" && ! -L "$repo" ]] || {
        echo "ERROR: required real repository directory is absent: $repo" >&2
        exit 1
    }
done

require_sha256 origin_manifest "$ORIGIN_MANIFEST" "$EXPECTED_ORIGIN_MANIFEST_SHA256"
require_sha256 origin_requirements "$ORIGIN_REQUIREMENTS" "$EXPECTED_ORIGIN_REQUIREMENTS_SHA256"
require_sha256 relocated_requirements "$RELOCATED_REQUIREMENTS" "$EXPECTED_RELOCATED_REQUIREMENTS_SHA256"
require_sha256 failed_env_receipt "$FAILED_ENV_RECEIPT" "$EXPECTED_FAILED_ENV_RECEIPT_SHA256"
require_sha256 plain_wheel "$ROBO_PLAIN_WHEEL" "$EXPECTED_PLAIN_WHEEL_SHA256"
require_sha256 wheel_receipt "$WHEEL_RECEIPT" "$EXPECTED_WHEEL_RECEIPT_SHA256"
[[ "$(stat -c %s "$ROBO_PLAIN_WHEEL")" == "$EXPECTED_PLAIN_WHEEL_SIZE" ]]
[[ "$(stat -c %a "$ROBO_PLAIN_WHEEL")" == 444 ]]
[[ "$(stat -c %a "$FAILED_ENV_RECEIPT")" == 444 ]]
[[ "$(stat -c %a "$WHEEL_RECEIPT")" == 444 ]]
[[ "$("$UV" --version | awk '{print $2}')" == "$EXPECTED_UV_VERSION" ]]
[[ "$("$PYTHON" -c 'import platform; print(platform.python_version())')" == "$EXPECTED_PYTHON_VERSION" ]]
[[ "$(git -c safe.directory="$ROBOCASA_REPO" -C "$ROBOCASA_REPO" rev-parse HEAD)" == "$EXPECTED_ROBOCASA_COMMIT" ]]
[[ "$(git -c safe.directory="$ROBOSUITE_REPO" -C "$ROBOSUITE_REPO" rev-parse HEAD)" == "$EXPECTED_ROBOSUITE_COMMIT" ]]
[[ "$(git -c safe.directory="$DEMIAN_REPO" -C "$DEMIAN_REPO" rev-parse HEAD)" == "$EXPECTED_DEMIAN_COMMIT" ]]
[[ -z "$(git -c safe.directory="$ROBOCASA_REPO" -C "$ROBOCASA_REPO" status --porcelain=v1 --untracked-files=all)" ]]
[[ -z "$(git -c safe.directory="$ROBOSUITE_REPO" -C "$ROBOSUITE_REPO" status --porcelain=v1 --untracked-files=all)" ]]
[[ -z "$(git -c safe.directory="$DEMIAN_REPO" -C "$DEMIAN_REPO" status --porcelain=v1 --untracked-files=all)" ]]

exec > >(tee -a "$ROBO_BUILD_LOG") 2>&1
trap record_failure ERR
echo "ITERATION_ID=$ITERATION_ID"
echo "STARTED_AT_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "STAGING_ENV=$ROBO_STAGING_ENV"
echo "BUILD_RECEIPT=$ROBO_BUILD_RECEIPT"
echo "PACKAGE_SOURCE_NOTE=$PACKAGE_SOURCE_NOTE"

export UV_CACHE_DIR="$UV_CACHE"
export UV_HTTP_TIMEOUT=1200
export UV_CONCURRENT_DOWNLOADS=16
export UV_CONCURRENT_BUILDS=4
export UV_CONCURRENT_INSTALLS=16
export UV_LINK_MODE=hardlink

"$UV" venv --relocatable --python "$PYTHON" --seed "$ROBO_STAGING_ENV"
"$UV" pip install \
    --python "$ROBO_STAGING_ENV/bin/python" \
    --index-strategy unsafe-best-match \
    --default-index https://mirrors.aliyun.com/pypi/simple \
    --torch-backend cu128 \
    -r "$RELOCATED_REQUIREMENTS"
"$UV" pip install \
    --python "$ROBO_STAGING_ENV/bin/python" \
    --no-deps \
    --reinstall-package torchcodec \
    "$ROBO_PLAIN_WHEEL"
"$UV" pip check --python "$ROBO_STAGING_ENV/bin/python"

readonly EVAL_PYTHONPATH="$DEMIAN_REPO/demian_wam:$DEMIAN_REPO/openpi/src:$DEMIAN_REPO/openpi/packages/openpi-client/src:$ROBOCASA_REPO:$ROBOSUITE_REPO"
PYTHONDONTWRITEBYTECODE=1 \
PYTHONNOUSERSITE=1 \
PYTHONPATH="$EVAL_PYTHONPATH" \
"$ROBO_STAGING_ENV/bin/python" "$EVALUATOR" --help >/dev/null

export ROBO_STAGING_ENV ROBO_BUILD_RECEIPT READY_MARKER
export ORIGIN_MANIFEST ORIGIN_REQUIREMENTS RELOCATED_REQUIREMENTS
export FAILED_ENV_RECEIPT WHEEL_RECEIPT ROBO_PLAIN_WHEEL
export EXPECTED_PYTHON_VERSION EXPECTED_TORCH_VERSION EXPECTED_TORCH_CUDA_VERSION
export EXPECTED_MUJOCO_VERSION EXPECTED_ROBOCASA_VERSION EXPECTED_ROBOSUITE_VERSION
export EXPECTED_DISTRIBUTION_COUNT EXPECTED_ROBOCASA_COMMIT EXPECTED_ROBOSUITE_COMMIT
export EXPECTED_ORIGIN_MANIFEST_SHA256 EXPECTED_ORIGIN_REQUIREMENTS_SHA256
export EXPECTED_RELOCATED_REQUIREMENTS_SHA256 EXPECTED_PLAIN_WHEEL_SHA256
export EXPECTED_WHEEL_RECEIPT_SHA256 PACKAGE_SOURCE_NOTE EVALUATOR EVAL_PYTHONPATH

LD_LIBRARY_PATH= \
PYTHONDONTWRITEBYTECODE=1 \
PYTHONNOUSERSITE=1 \
PYTHONPATH="$EVAL_PYTHONPATH" \
"$ROBO_STAGING_ENV/bin/python" - <<'PY'
import hashlib
import importlib.metadata as md
import importlib.util
import json
import os
from pathlib import Path
import platform
import stat
import sys

from packaging.utils import canonicalize_name


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def descriptor(path):
    path = Path(path).resolve(strict=True)
    file_stat = path.stat()
    return {
        "path": str(path),
        "sha256": sha256(path),
        "size_bytes": file_stat.st_size,
        "mode": f"{stat.S_IMODE(file_stat.st_mode):04o}",
    }


environment = Path(os.environ["ROBO_STAGING_ENV"]).resolve(strict=True)
assert Path(sys.prefix).resolve(strict=True) == environment
assert platform.python_version() == os.environ["EXPECTED_PYTHON_VERSION"], sys.version

origin_path = Path(os.environ["ORIGIN_MANIFEST"])
origin = json.loads(origin_path.read_text())
expected = {
    canonicalize_name(name): item["version"]
    for name, item in origin["distributions"].items()
}
actual = {
    canonicalize_name(item.metadata["Name"]): item.version
    for item in md.distributions()
    if item.metadata.get("Name")
}
assert len(actual) == int(os.environ["EXPECTED_DISTRIBUTION_COUNT"]), len(actual)
assert actual == expected, {
    "missing": sorted(set(expected) - set(actual)),
    "extra": sorted(set(actual) - set(expected)),
    "drift": {
        name: (expected[name], actual[name])
        for name in sorted(expected.keys() & actual.keys())
        if expected[name] != actual[name]
    },
}

import mujoco
import robocasa
import robosuite
import torch

assert torch.__version__ == os.environ["EXPECTED_TORCH_VERSION"]
assert torch.version.cuda == os.environ["EXPECTED_TORCH_CUDA_VERSION"]
assert torch.cuda.is_available()
assert torch.cuda.device_count() == 4
gpu_names = [torch.cuda.get_device_name(index) for index in range(4)]
assert gpu_names == ["NVIDIA GeForce RTX 4090"] * 4, gpu_names
assert mujoco.__version__ == os.environ["EXPECTED_MUJOCO_VERSION"]
assert md.version("robocasa") == os.environ["EXPECTED_ROBOCASA_VERSION"]
assert md.version("robosuite") == os.environ["EXPECTED_ROBOSUITE_VERSION"]
assert md.version("torchcodec") == "0.5"

evaluator = Path(os.environ["EVALUATOR"]).resolve(strict=True)
spec = importlib.util.spec_from_file_location("cwam_robocasa_evaluator_smoke", evaluator)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
assert "torchcodec" not in sys.modules
evaluator_smoke = {
    "path": str(evaluator),
    "module": spec.name,
    "openpi_client_image_tools": str(
        Path(sys.modules["openpi_client.image_tools"].__file__).resolve()
    ),
    "torchcodec_loaded": "torchcodec" in sys.modules,
}

torchcodec_import = {"status": "passed"}
try:
    import torchcodec
except Exception as error:
    torchcodec_import = {
        "status": "failed_non_gate",
        "error_type": type(error).__name__,
        "error_message": str(error),
    }

marker = {
    "schema_version": 1,
    "status": "ready",
    "python": platform.python_version(),
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "mujoco": mujoco.__version__,
    "robocasa": md.version("robocasa"),
    "robosuite": md.version("robosuite"),
    "distribution_count": len(actual),
    "distributions_sha256": origin["distributions_sha256"],
    "origin_environment_manifest_sha256": os.environ[
        "EXPECTED_ORIGIN_MANIFEST_SHA256"
    ],
    "origin_requirements_sha256": os.environ[
        "EXPECTED_ORIGIN_REQUIREMENTS_SHA256"
    ],
    "relocated_requirements_sha256": os.environ[
        "EXPECTED_RELOCATED_REQUIREMENTS_SHA256"
    ],
    "robocasa_commit": os.environ["EXPECTED_ROBOCASA_COMMIT"],
    "robosuite_commit": os.environ["EXPECTED_ROBOSUITE_COMMIT"],
    "package_source_note": os.environ["PACKAGE_SOURCE_NOTE"],
}
expected_marker_keys = {
    "cuda_available",
    "distribution_count",
    "distributions_sha256",
    "mujoco",
    "origin_environment_manifest_sha256",
    "origin_requirements_sha256",
    "package_source_note",
    "python",
    "relocated_requirements_sha256",
    "robocasa",
    "robocasa_commit",
    "robosuite",
    "robosuite_commit",
    "schema_version",
    "status",
    "torch",
    "torch_cuda",
}
assert set(marker) == expected_marker_keys
marker_path = Path(os.environ["READY_MARKER"])
marker_temporary = marker_path.with_name(f"{marker_path.name}.tmp.{os.getpid()}")
assert not marker_path.exists() and not marker_path.is_symlink()
assert not marker_temporary.exists() and not marker_temporary.is_symlink()
with marker_temporary.open("x") as stream:
    json.dump(marker, stream, indent=2, sort_keys=True, allow_nan=False)
    stream.write("\n")
marker_temporary.chmod(0o444)
os.replace(marker_temporary, marker_path)

receipt = {
    "schema_version": 1,
    "status": "passed",
    "kind": "cwam_dsw974_robocasa_fresh_native_rebuild",
    "environment": str(environment),
    "marker": descriptor(marker_path),
    "package_source_note": os.environ["PACKAGE_SOURCE_NOTE"],
    "distribution_count": len(actual),
    "distributions": dict(sorted(actual.items())),
    "gpu_names": gpu_names,
    "evaluator_openpi_import_smoke": evaluator_smoke,
    "torchcodec_import_diagnostic": torchcodec_import,
    "inputs": {
        "origin_manifest": descriptor(os.environ["ORIGIN_MANIFEST"]),
        "origin_requirements": descriptor(os.environ["ORIGIN_REQUIREMENTS"]),
        "relocated_requirements": descriptor(os.environ["RELOCATED_REQUIREMENTS"]),
        "failed_environment_receipt": descriptor(
            os.environ["FAILED_ENV_RECEIPT"]
        ),
        "torchcodec_plain_wheel": descriptor(os.environ["ROBO_PLAIN_WHEEL"]),
        "torchcodec_wheel_relay_receipt": descriptor(
            os.environ["WHEEL_RECEIPT"]
        ),
    },
}
receipt_path = Path(os.environ["ROBO_BUILD_RECEIPT"])
receipt_temporary = receipt_path.with_name(f"{receipt_path.name}.tmp.{os.getpid()}")
assert not receipt_path.exists() and not receipt_path.is_symlink()
assert not receipt_temporary.exists() and not receipt_temporary.is_symlink()
with receipt_temporary.open("x") as stream:
    json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
    stream.write("\n")
receipt_temporary.chmod(0o444)
os.replace(receipt_temporary, receipt_path)
print(json.dumps({"marker": descriptor(marker_path), "receipt": descriptor(receipt_path)}, sort_keys=True))
PY

trap - ERR
chmod 0444 "$ROBO_BUILD_LOG"
echo "READY_MARKER=$READY_MARKER"
echo "READY_MARKER_SHA256=$(sha256_file "$READY_MARKER")"
echo "BUILD_RECEIPT_SHA256=$(sha256_file "$ROBO_BUILD_RECEIPT")"
stat -c 'READY_MARKER_MODE=%a READY_MARKER_BYTES=%s READY_MARKER_MTIME=%y' "$READY_MARKER"
stat -c 'BUILD_RECEIPT_MODE=%a BUILD_RECEIPT_BYTES=%s BUILD_RECEIPT_MTIME=%y' "$ROBO_BUILD_RECEIPT"
du -sh "$ROBO_STAGING_ENV"
echo "COMPLETED_AT_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
