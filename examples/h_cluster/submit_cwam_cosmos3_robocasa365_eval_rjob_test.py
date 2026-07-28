# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Static and early-failure tests for the strict RoboCasa365 H submitter."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

SUBMITTER = Path(__file__).with_name("submit_cwam_cosmos3_robocasa365_eval_rjob.sh")

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def _gate_environment() -> dict[str, str]:
    digest = "a" * 64
    return {
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "COSMOS_SOURCE": "/not/reached/cosmos",
        "COSMOS_ENV": "/not/reached/cosmos-env",
        "ROBOCASA_ENV": "/not/reached/robocasa-env",
        "DEMIAN_REPO": "/not/reached/demian",
        "ROBOCASA_REPO": "/not/reached/robocasa",
        "ROBOSUITE_REPO": "/not/reached/robosuite",
        "SERVING_CHECKPOINT_PATH": "/not/reached/checkpoint",
        "CHECKPOINT_MANIFEST": "/not/reached/checkpoint-manifest.json",
        "TRAINING_CONFIG": "/not/reached/config.py",
        "ACTION_STATS": "/not/reached/action-stats.json",
        "SOURCE_MANIFEST": "/not/reached/snapshot-manifest.json",
        "COSMOS_ENVIRONMENT_MANIFEST": "/not/reached/cosmos-environment.json",
        "ROBOCASA_ENVIRONMENT_MANIFEST": "/not/reached/robocasa-environment.json",
        "EVAL_EXPERIMENT_ID": "CWAM-E35-GATE",
        "EVAL_LOGICAL_RUN_ID": "CWAM-E35-GATE-S42",
        "EXPECTED_SOURCE_COMMIT": "1" * 40,
        "EXPECTED_SNAPSHOT_MANIFEST_SHA256": digest,
        "EXPECTED_SOURCE_MANIFEST_SHA256": digest,
        "EXPECTED_CHECKPOINT_MANIFEST_SHA256": digest,
        "EXPECTED_TRAINING_CONFIG_SHA256": digest,
        "EXPECTED_ACTION_STATS_SHA256": digest,
        "EXPECTED_COSMOS_ENVIRONMENT_MANIFEST_SHA256": digest,
        "EXPECTED_ROBOCASA_ENVIRONMENT_MANIFEST_SHA256": digest,
        "EXPECTED_ROBOCASA_COMMIT": "29f7ce8814c1547f5af762a0997fbd4b64848dd7",
        "EXPECTED_ROBOSUITE_COMMIT": "2" * 40,
        "MINIMUM_MEAN_SUCCESS_RATE": "0.0",
        "MINIMUM_SUCCESSFUL_TASKS": "0",
        "EVAL_STAGE": "one_episode_gate",
        "EVAL_SPLIT": "atomic",
        "EVAL_TASKS_CSV": "OpenDrawer",
        "EPISODES_PER_TASK": "1",
        "NUM_SHARDS": "1",
        "BASE_SEED": "42",
        "MAX_STEPS": "400",
        "OPEN_LOOP_HORIZON": "12",
        "POLICY_SAMPLER": "unipc",
        "POLICY_SEED": "42",
        "POLICY_GUIDANCE": "1.0",
        "POLICY_NUM_STEPS": "30",
        "BASE_PORT": "16000",
        "SERVER_READY_TIMEOUT_SECONDS": "1800",
        "RJOB_NAME": "cwam-e35-gate",
        "RJOB_PREDICT_ONLY": "1",
    }


def _run_early(env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["/usr/bin/bash", str(SUBMITTER)],
        cwd=SUBMITTER.parent,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )


def test_default_is_prediction_only_and_phase_resources_are_exact() -> None:
    script = SUBMITTER.read_text(encoding="utf-8")

    assert ': "${RJOB_PREDICT_ONLY:=1}"' in script
    assert "RJOB_CPU=16\n        RJOB_MEMORY_MIB=187500\n        RJOB_GPU=1" in script
    assert "RJOB_CPU=128\n        RJOB_MEMORY_MIB=1500000\n        RJOB_GPU=8" in script
    assert '[[ "$NUM_SHARDS" == 1 ]]' in script
    assert '[[ "$NUM_SHARDS" == 8 ]]' in script
    assert '[[ -z "$EVAL_TASKS_CSV" ]]' in script
    assert "SUBMIT_ARGS+=(--predict-only true)" in script


def test_scheduler_image_mount_nvme_and_lifecycle_contract_are_pinned() -> None:
    script = SUBMITTER.read_text(encoding="utf-8")

    for token in (
        "registry.h.pjlab.org.cn/ailab/vulkan@sha256:59204dea15a88b9b444b4f20c2e54b6b92d41582ca718f35e9194335cc8615f7",
        "gpfs://gpfs2/ailab-eailabagent-gpfs:/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs",
        "gpfs://gpfs2/gpfs2-shared-public:/mnt/shared-storage-gpfs2/gpfs2-shared-public",
        "--store-host-nvme",
        "--private-machine group",
        "--priority 9",
        "--task-type normal",
        "--restart-policy never",
        "--termination-grace-period-seconds 60",
        "--backoff_limit 1",
        "--replicas",
        "node/gpu-lg-cmc-h-h200-0685.host.h.pjlab.org.cn",
    ):
        assert token in script
    assert '--image "$PINNED_CONTAINER_IMAGE"' in script
    assert '--cpu "$RJOB_CPU"' in script
    assert '--memory "$RJOB_MEMORY_MIB"' in script
    assert '--gpu "$RJOB_GPU"' in script


def test_external_nonce_scheduler_readback_and_graphics_environment_are_forwarded() -> None:
    script = SUBMITTER.read_text(encoding="utf-8")

    for token in (
        'SUBMISSION_NONCE="$("$PYTHON3_BIN" -I -c',
        'export EVAL_RUNTIME_CAPTURE_NONCE="$SUBMISSION_NONCE"',
        'export EVAL_SUBMITTED_AT_UTC="$SUBMITTED_AT_UTC"',
        "EXPECTED_EVAL_SUBMISSION_PENDING_SHA256",
        "EVAL_SCHEDULER_READBACK_RECEIPT",
        "scheduler_started_at_utc",
        "WORKER_COMMAND_SHA256",
        "worker-env-contract.json",
        "__EGL_VENDOR_LIBRARY_FILENAMES",
        "__GLX_VENDOR_LIBRARY_NAME",
        "unset BASH_ENV ENV PYTHONHOME PYTHONOPTIMIZE PYTHONPATH",
    ):
        assert token in script
    assert "os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW" in script
    assert "replica_started" in script


def test_missing_inputs_and_phase_drift_fail_before_any_rjob_call() -> None:
    result = _run_early(
        {
            "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        }
    )
    assert result.returncode != 0
    assert "Required environment variable is not set: COSMOS_SOURCE" in result.stderr

    invalid_gate = _gate_environment()
    invalid_gate["NUM_SHARDS"] = "8"
    result = _run_early(invalid_gate)
    assert result.returncode != 0
    assert "one_episode_gate requires NUM_SHARDS=1" in result.stderr
    assert "Required absolute tool is unavailable" not in result.stderr

    invalid_identity = _gate_environment()
    invalid_identity["EVAL_LOGICAL_RUN_ID"] = "../escape"
    result = _run_early(invalid_identity)
    assert result.returncode != 0
    assert "path-safe characters with alphanumeric endpoints" in result.stderr


def test_real_mode_uses_dual_exclusive_reservations_and_preserves_tombstones() -> None:
    script = SUBMITTER.read_text(encoding="utf-8")

    assert "Refusing to submit against an existing experiment output root" in script
    assert "by-experiment/$EVAL_EXPERIMENT_ID" in script
    assert "by-logical-run/$EVAL_LOGICAL_RUN_ID" in script
    assert '/usr/bin/mkdir --mode=0700 -- "$EXPERIMENT_RESERVATION_DIR"' in script
    assert '/usr/bin/mkdir --mode=0700 -- "$RUN_RESERVATION_DIR"' in script
    assert "logical_run_reservation_collision_after_experiment_lock" in script
    assert "reservation retained" in script
    assert "submission-unknown.json" in script
    assert "os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW" in script
    assert "trap 'post_accept_failure \"$?\"' ERR" in script
    assert "trap 'post_accept_exit \"$?\"' EXIT" in script
    assert 'mark_submission_unknown "unexpected_post_accept_failure"' in script
    assert 'mark_submission_unknown "unexpected_post_accept_exit"' in script
    assert script.index("trap 'post_accept_failure \"$?\"' ERR") < script.index('"$CHMOD_BIN" 0444 "$SUBMISSION_LOG"')
    assert script.index("trap 'post_accept_failure \"$?\"' ERR") < script.index(
        '"$PYTHON3_BIN" - \\\n    "$SCHEDULER_EXPECTED_CONTRACT"'
    )
    assert "POST_ACCEPT_COMPLETE=1\ntrap - ERR EXIT" in script


def test_submitter_has_no_unscoped_caller_environment_dependency() -> None:
    script = SUBMITTER.read_text(encoding="utf-8")
    runner = SUBMITTER.with_name("run_cwam_cosmos3_robocasa365_eval.sh").read_text(encoding="utf-8")

    assert "readonly CALLER_INPUTS=(" in script
    assert "readonly WORKER_ENV_NAMES=(" in script
    assert 'for name in "${WORKER_ENV_NAMES[@]}"; do' in script
    assert "printf -v quoted '%q' \"${!name}\"" in script
    assert "BASH_ENV" in script
    assert "PYTHONHOME" in script
    assert "LD_PRELOAD" in script
    assert 'readonly RJOB_BIN="/usr/local/bin/rjob"' in script
    required = set(re.findall(r': "\$\{([A-Z0-9_]+):\?', runner))
    worker_block = re.search(
        r"readonly WORKER_ENV_NAMES=\(\n(.*?)\n\)",
        script,
        re.DOTALL,
    )
    assert worker_block is not None
    forwarded = set(re.findall(r"^\s+([A-Z0-9_]+)\s*$", worker_block.group(1), re.MULTILINE))
    scheduler_injected = {"KUBEBRAIN_RJOB_NAME", "KUBEBRAIN_RJOB_UID"}
    assert required <= forwarded | scheduler_injected


def test_all_embedded_python_programs_compile() -> None:
    script = SUBMITTER.read_text(encoding="utf-8")
    programs = re.findall(r"<<'PY'\n(.*?)\nPY(?:\n|\Z)", script, re.DOTALL)

    assert len(programs) >= 10
    for index, program in enumerate(programs, start=1):
        compile(program, f"{SUBMITTER.name}:heredoc-{index}", "exec")
