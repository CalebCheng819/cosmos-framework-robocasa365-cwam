# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Static fail-closed contract tests for the RoboCasa365 eval runner."""

from __future__ import annotations

from pathlib import Path

import pytest

RUNNER = Path(__file__).with_name("run_cwam_cosmos3_robocasa365_eval.sh")

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def test_runner_pins_causal_eval_and_required_evidence() -> None:
    script = RUNNER.read_text(encoding="utf-8")

    for token in (
        "--no-save-video",
        "--server-runtime-receipt",
        "summary.csv",
        "task_success_rates.csv",
        "RAW_RESULT_FILES",
        "cwam_serving_checkpoint_manifest.py",
        "SERVING_CHECKPOINT_VERIFIED",
        "capture_robocasa365_eval_runtime.py",
        "EVAL_RUNTIME_RECEIPT_READY",
        "--cosmos-environment-manifest",
        "--robocasa-environment-manifest",
        "--eval-runtime-receipt",
        "chmod 0444",
        "ROBOCASA365_EVAL_EXECUTION_SUCCESS",
        "ROBOCASA365_EVAL_EFFECT_GATE_PASSED",
    ):
        assert token in script
    for forbidden in (
        "--gt-reasoning-file",
        "--vlm-captions-file",
        "--fg-captions-path",
        "--dynamic-instructions-file",
        "--eval-dataset",
    ):
        assert forbidden not in script
    assert "--require-effect-pass" not in script
    assert script.index("chmod 0444") < script.index(
        "Formal evaluation completed and was sealed, but the preregistered effect gate failed"
    )


def test_runner_uses_one_server_and_evaluator_per_gpu_shard() -> None:
    script = RUNNER.read_text(encoding="utf-8")

    assert 'CUDA_VISIBLE_DEVICES="$shard"' in script
    assert 'MUJOCO_EGL_DEVICE_ID="$shard"' in script
    assert "MUJOCO_EGL_DEVICE_ID=0" not in script
    assert "port=$((BASE_PORT + shard))" in script
    assert "--shard-index" in script
    assert "--task-shard" in script
    assert "--num-shards" in script
    assert "--launch-nonce" in script
    assert "--expected-launch-nonce" in script
    assert "--expected-server-pid" in script
    assert 'cd "$COSMOS_SOURCE"' in script
    assert "source_module_sha256" in Path(__file__).with_name("capture_robocasa365_server_receipt.py").read_text(
        encoding="utf-8"
    )


def test_runner_resolves_cosmos_local_checkpoint_model_subdirectory() -> None:
    script = RUNNER.read_text(encoding="utf-8")

    assert 'if [[ -d "$SERVING_CHECKPOINT_PATH/model" ]]; then' in script
    assert 'SERVING_CHECKPOINT_PATH="$(canonical_existing "$SERVING_CHECKPOINT_PATH/model")"' in script
    assert script.index('readonly COSMOS_PY="$COSMOS_ENV/bin/python"') < script.index(
        'PYTHONPATH="$COSMOS_SOURCE" "$COSMOS_PY" "$COSMOS_SOURCE/$CHECKPOINT_HELPER_REL"'
    )


def test_runner_captures_runtime_before_starting_policy_servers() -> None:
    script = RUNNER.read_text(encoding="utf-8")

    runtime_capture = script.index('"$COSMOS_PY" "$COSMOS_SOURCE/$RUNTIME_RECEIPT_HELPER_REL"')
    policy_server_loop = script.index("for ((shard = 0; shard < NUM_SHARDS; shard++)); do")
    assert runtime_capture < policy_server_loop
    assert 'readonly H_WORKER_LD_LIBRARY_PATH="${LD_LIBRARY_PATH-}"' in script
    assert 'export LD_LIBRARY_PATH="$H_NVIDIA_DRIVER_LIB:$H_NVIDIA_DRIVER_LIB64"' in script
    assert "LD_LIBRARY_PATH= \\" not in script
    assert '--num-gpus "$NUM_SHARDS"' in script
    assert '--experiment-id "$EVAL_EXPERIMENT_ID"' in script
    assert '--logical-run-id "$EVAL_LOGICAL_RUN_ID"' in script
    assert '--capture-nonce "$EVAL_RUNTIME_CAPTURE_NONCE"' in script
    assert "secrets.token_hex(32)" in script
    runtime_helper = Path(__file__).with_name("capture_robocasa365_eval_runtime.py").read_text(encoding="utf-8")
    assert "EGLGLContext(max_width=64, max_height=64, device_id=shard_index)" in runtime_helper
    assert '"CUDA_VISIBLE_DEVICES": str(shard_index)' in runtime_helper
    assert '"MUJOCO_EGL_DEVICE_ID": str(shard_index)' in runtime_helper
    assert "robocasa_egl_contexts_created_per_shard" in runtime_helper


def test_runner_hard_pins_episode_reset_demian_revision() -> None:
    script = RUNNER.read_text(encoding="utf-8")

    assert 'readonly EXPECTED_DEMIAN_COMMIT="2600ece5bf0eeacf010a95f084dcda2de6d30d59"' in script
    assert (
        "readonly EXPECTED_DEMIAN_EVALUATOR_SHA256="
        '"c38ed1f4049d079b84c185cb572667a47eaf984c2c22b492efc8985401119836"' in script
    )
    assert (
        "readonly EXPECTED_DEMIAN_TASK_REGISTRY_SHA256="
        '"e902311cf435986859ae01920548aba169e085ad6d26187cad9fb0d1f9c315de"' in script
    )
    assert ': "${EXPECTED_DEMIAN_COMMIT:?' not in script
    assert ': "${EXPECTED_DEMIAN_EVALUATOR_SHA256:?' not in script


def test_runner_rejects_reused_outputs_and_non_nvme_scratch() -> None:
    script = RUNNER.read_text(encoding="utf-8")

    assert "Refusing to reuse EVAL_OUTPUT_ROOT" in script
    assert "Refusing to reuse EVAL_NVME_ROOT" in script
    assert "EVAL_NVME_ROOT must be a strict child of /nvme" in script
    assert "/nvme is not backed by /dev/nvme1n1" in script
    assert "/nvme/.cwam-robocasa365-eval/$EVAL_LOGICAL_RUN_ID" in script


def test_formal_mode_can_select_the_complete_split() -> None:
    script = RUNNER.read_text(encoding="utf-8")

    assert ': "${EVAL_TASKS_CSV:=}"' in script
    assert ': "${EVAL_TASKS_CSV:=OpenDrawer}"' not in script
    assert "formal evaluation must use the complete canonical split" in script
    assert "formal evaluation requires NUM_SHARDS=8" in script


def test_signal_traps_cannot_report_success() -> None:
    script = RUNNER.read_text(encoding="utf-8")

    assert "trap 'cleanup 130' INT" in script
    assert "trap 'cleanup 143' TERM" in script
    assert "trap cleanup EXIT INT TERM" not in script
