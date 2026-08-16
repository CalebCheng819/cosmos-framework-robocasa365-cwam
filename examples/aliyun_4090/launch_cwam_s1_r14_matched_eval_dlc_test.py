# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Static contract tests for the S1-versus-R14 DLC bootstrap."""

from __future__ import annotations

import stat
import subprocess
from pathlib import Path

import pytest

pytestmark = [pytest.mark.L0, pytest.mark.CPU]

HERE = Path(__file__).resolve().parent
BOOTSTRAP = HERE / "launch_cwam_s1_r14_matched_eval_dlc.sh"


def _script() -> str:
    return BOOTSTRAP.read_text(encoding="utf-8")


def test_bootstrap_pins_experiment_and_r14_baseline() -> None:
    script = _script()

    assert "CWAM-DLC-ROB365-S1-VS-R14-MATCHED2-8GPU-S42-R9-20260817" in script
    assert "CWAM-DLC-ROB365-COSMOS3-NANO-S1-EMA-10000-S42-MATCHED2-R9-20260817" in script
    assert "MATCHED2-R8-20260817" not in script
    assert "CWAM-DSW960-ROB365-COSMOS3-NANO-R14-EMA-340EP-S42-FRESH-RERUN2-20260809" in script
    assert 'readonly R14_BASELINE_ARGS="$R14_BASELINE_ROOT/args_shard0.json"' in script
    assert 'readonly R14_BASELINE_RESULTS="$R14_BASELINE_ROOT/results_shard0.json"' in script
    assert "export R14_BASELINE_ARGS" in script
    assert "export R14_BASELINE_RESULTS" in script


def test_bootstrap_requires_one_eight_gpu_worker_and_both_mounts() -> None:
    script = _script()

    assert "nvidia-smi --query-gpu=index" in script
    assert ')" == 8 ]]' in script
    assert "findmnt -n -T /mnt/workspace -o TARGET" in script
    assert "findmnt -n -T /oss-chengjuntao -o TARGET" in script
    assert "readonly REQUIRED_LOCAL_BYTES=161061273600" in script


def test_bootstrap_binds_persistent_runs_to_writable_oss() -> None:
    script = _script()

    assert '[[ -L "$PROJECT_ROOT/runs" ]]' in script
    assert 'realpath -e -- "$PROJECT_ROOT/runs"' in script
    assert '== "$OSS_PROJECT_ROOT/runs"' in script
    assert 'findmnt -n -T "$OSS_PROJECT_ROOT/runs" -o TARGET' in script
    assert '[[ -w "$OSS_PROJECT_ROOT/runs" ]]' in script


def test_local_capacity_probe_executes_with_gnu_df() -> None:
    assignment = next(
        line for line in _script().splitlines() if line.startswith("available_local_bytes=")
    )
    completed = subprocess.run(
        [
            "bash",
            "-c",
            f"set -Eeuo pipefail\n{assignment}\nprintf '%s\\n' \"$available_local_bytes\"",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.strip().isdigit()


def test_bootstrap_provisions_nvidia_egl_before_large_asset_staging() -> None:
    script = _script()

    assert 'readonly EGL_GLVD_FRONTEND_SOURCE="$INPUTS_ROOT/runtime/glvnd/libEGL.so.1.1.0"' in script
    assert "controlled GLVND EGL frontend size mismatch" in script
    assert 'install -m 0444 -- "$EGL_GLVD_FRONTEND_SOURCE" "$EGL_GLVD_FRONTEND"' in script
    assert '"$EGL_GLVD_RUNTIME_ROOT/libEGL.so.1"' in script
    assert '"$EGL_GLVD_RUNTIME_ROOT/libEGL.so"' in script
    assert 'export LD_LIBRARY_PATH="$EGL_GLVD_RUNTIME_ROOT${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"' in script
    assert 'readonly EGL_VENDOR_JSON=/usr/share/glvnd/egl_vendor.d/10_nvidia.json' in script
    assert 'getattr(frontend, "eglQueryString")' in script
    assert 'ctypes.CDLL("libEGL_nvidia.so.0", mode=ctypes.RTLD_GLOBAL)' in script
    assert 'NVIDIA_DRIVER_CAPABILITIES must include graphics or all' in script
    assert 'install -d -m 0755 -- "$egl_vendor_dir"' in script
    assert 'chmod 0644 -- "$temporary_vendor_json"' in script
    assert 'mv -T -- "$temporary_vendor_json" "$EGL_VENDOR_JSON"' in script
    assert script.index("prepare_nvidia_egl_runtime\n") < script.index(
        'copy_regular_file \\\n    "$SOURCE_ENV_ARCHIVE"'
    )


def test_bootstrap_runs_real_mujoco_egl_smoke_before_large_asset_staging() -> None:
    script = _script()

    assert 'readonly EGL_SMOKE_PROBE="$SOURCE_SNAPSHOT_ROOT/examples/aliyun_4090/probe_mujoco_egl_mapping.py"' in script
    assert "CUDA_VISIBLE_DEVICES=0" in script
    assert "MUJOCO_EGL_DEVICE_ID=0" in script
    assert "MUJOCO_GL=egl" in script
    assert "PYOPENGL_PLATFORM=egl" in script
    assert '"$ROBOCASA_ENV_ROOT/bin/python" "$EGL_SMOKE_PROBE" --hold-seconds 1' in script
    assert script.index("preflight_mujoco_egl_runtime\n") < script.index(
        'copy_regular_file \\\n    "$SOURCE_ENV_ARCHIVE"'
    )


def test_bootstrap_stages_large_assets_under_fixed_local_root() -> None:
    script = _script()

    assert "readonly LOCAL_ASSET_ROOT=/tmp/cwam_s1_r14_matched_assets" in script
    assert "readonly LOCAL_BUILD_ROOT=/tmp/.cwam_s1_r14_matched_assets.incomplete" in script
    assert "readonly OSS_PROJECT_ROOT=/oss-chengjuntao/" in script
    assert '"$SOURCE_ENV_ARCHIVE"' in script
    assert '"$SOURCE_WAN_VAE"' in script
    assert '"$SOURCE_QWEN_TOKENIZER"' in script
    assert '"$SOURCE_CHECKPOINT_ROOT"' in script
    assert 'mv -T -- "$LOCAL_BUILD_ROOT" "$LOCAL_ASSET_ROOT"' in script


def test_bootstrap_normalizes_staged_directory_modes(tmp_path: Path) -> None:
    build_root = tmp_path / "build"
    stage_root = build_root / "checkpoints" / ".checkpoint.incomplete-transfer-20260812"
    model_root = stage_root / "model"
    model_root.mkdir(parents=True)
    for directory in (build_root, stage_root, model_root):
        directory.chmod(0o700)
    normalization = next(
        line
        for line in _script().splitlines()
        if line.startswith('find "$LOCAL_BUILD_ROOT" -type d -exec chmod 0755')
    )

    subprocess.run(
        ["bash", "-c", normalization],
        check=True,
        env={"LOCAL_BUILD_ROOT": str(build_root)},
    )

    assert all(
        stat.S_IMODE(directory.stat().st_mode) == 0o755
        for directory in (build_root, stage_root, model_root)
    )


def test_bootstrap_publishes_s1_from_exact_eight_shard_closure() -> None:
    script = _script()

    for index in range(8):
        assert f"__{index}_0.distcp" in script
    assert "--stage-root \"$LOCAL_CHECKPOINT_STAGE\"" in script
    assert "--final-root \"$LOCAL_CHECKPOINT_ROOT\"" in script
    assert "--contract \"$CHECKPOINT_CONTRACT\"" in script
    assert "--allowed-checkpoint-parent \"$LOCAL_CHECKPOINT_PARENT\"" in script
    assert "--publish" in script


def test_bootstrap_uses_immutable_external_control_files() -> None:
    script = _script()

    assert '[[ "$(stat -c %a -- "$path")" == 444 ]]' in script
    assert 'export FORMAL_RUNNER_OVERRIDE' in script
    assert 'readonly EGL_MATRIX_OVERRIDE="$CONTROL_DEPLOY_ROOT/run_mujoco_egl_mapping_matrix_dlc.sh"' in script
    assert 'export EGL_MATRIX_OVERRIDE' in script
    assert 'export S1_R14_MATCHED_HELPER' in script
    assert 'export COSMOS_ENV_ARCHIVE_METADATA_HELPER' in script
    assert 'export EXPECTED_COSMOS_ENV_ARCHIVE_SHA256' not in script
    assert 'export EXPECTED_COSMOS_ENV_ARCHIVE_PUBLICATION_RECEIPT_SHA256' not in script
    assert 'export EVAL_CONTROL_COMMIT' in script
    assert 'export S1_R14_MATCHED_MODE=1' in script
    assert 'exec bash "$FORMAL_LAUNCHER"' in script
    assert 'FORMAL_RUNNER_OVERRIDE="$COSMOS_SOURCE/' not in script


def test_task_protocol_is_owned_by_formal_runner() -> None:
    script = _script()

    assert "CloseBlenderLid" not in script
    assert "PickPlaceCounterToStove" not in script
    assert "EVAL_TASK_ARGS" not in script
