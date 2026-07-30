# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Static and early-failure tests for the strict RoboCasa365 H submitter."""

from __future__ import annotations

import json
import os
import re
import stat
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
        "EVAL_INTENT_MANIFEST": "/not/reached/eval-intent.json",
        "SOURCE_DCP_EXPORT_RECEIPT": "/not/reached/source-dcp-export-receipt.json",
        "EVAL_EXPERIMENT_ID": "CWAM-E35-GATE",
        "EVAL_LOGICAL_RUN_ID": "CWAM-E35-GATE-S42",
        "EVAL_EVALUATION_CLASS": "exploratory",
        "SOURCE_GIT_PUSHED": "false",
        "EXPECTED_SOURCE_COMMIT": "1" * 40,
        "EXPECTED_SNAPSHOT_MANIFEST_SHA256": digest,
        "EXPECTED_SOURCE_MANIFEST_SHA256": digest,
        "EXPECTED_CHECKPOINT_MANIFEST_SHA256": digest,
        "EXPECTED_TRAINING_CONFIG_SHA256": digest,
        "EXPECTED_ACTION_STATS_SHA256": digest,
        "EXPECTED_COSMOS_ENVIRONMENT_MANIFEST_SHA256": digest,
        "EXPECTED_ROBOCASA_ENVIRONMENT_MANIFEST_SHA256": digest,
        "EXPECTED_EVAL_INTENT_MANIFEST_SHA256": digest,
        "EXPECTED_SOURCE_DCP_EXPORT_RECEIPT_SHA256": digest,
        "EXPECTED_EGL_VENDOR_JSON_SHA256": digest,
        "EXPECTED_ROBOCASA_COMMIT": "29f7ce8814c1547f5af762a0997fbd4b64848dd7",
        "EXPECTED_ROBOSUITE_COMMIT": "2" * 40,
        "MINIMUM_MEAN_SUCCESS_RATE": "0.0",
        "MINIMUM_SUCCESSFUL_TASKS": "0",
        "EVAL_STAGE": "one_episode_gate",
        "EVAL_SPLIT": "atomic",
        "EVAL_TASKS_CSV": "OpenDrawer",
        "PRIOR_GATE_EVIDENCE": "",
        "EXPECTED_PRIOR_GATE_EVIDENCE_SHA256": "",
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


def _unknown_publisher_program() -> str:
    script = SUBMITTER.read_text(encoding="utf-8")
    marker = "\"$SCHEDULER_TEXT_READBACK\" <<'PY'\n"
    start = script.index(marker) + len(marker)
    end = script.index("\nPY\n}", start)
    return script[start:end]


def _output_initializer_program() -> str:
    script = SUBMITTER.read_text(encoding="utf-8")
    marker = "\"$PINNED_SUBMISSION_ROOT\" <<'PY'\n"
    start = script.index(marker) + len(marker)
    end = script.index("\nPY\n}", start)
    return script[start:end]


def _run_output_initializer(personal_root: Path) -> subprocess.CompletedProcess[str]:
    output_parent = personal_root / "outputs" / "cwam-robocasa365-eval"
    return subprocess.run(
        [
            "/usr/bin/python3",
            "-I",
            "-c",
            _output_initializer_program(),
            str(personal_root),
            str(output_parent),
            str(output_parent / "runs"),
            str(output_parent / "submission-reservations"),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )


def _render_submit_args(*, predict_only: bool, host_network: bool = False) -> list[str]:
    script = SUBMITTER.read_text(encoding="utf-8")
    start = script.index("SUBMIT_ARGS=(\n")
    end = script.index('\nif [[ "$RJOB_PREDICT_ONLY" == 1 ]]', start)
    array_program = script[start:end]
    values = {
        "RJOB_NAME": "unit-rjob",
        "PINNED_QUOTA_GROUP": "unit-group",
        "RJOB_REPLICAS": "1",
        "PINNED_CONTAINER_IMAGE": "unit-image",
        "RJOB_CPU": "16",
        "RJOB_GPU": "1",
        "RJOB_MEMORY_MIB": "187500",
        "RJOB_HOST_NETWORK": "true" if host_network else "false",
        "PINNED_EXCLUDED_NODE_TAG": "unit-negative-tag",
        "PERSONAL_GPFS_MOUNT": "unit-personal-mount",
        "PUBLIC_GPFS_MOUNT": "unit-public-mount",
    }
    assignments = "\n".join(f"{name}={value!r}" for name, value in values.items())
    harness = "\n".join(
        (
            "set -Eeuo pipefail",
            assignments,
            array_program,
            f"RJOB_PREDICT_ONLY={int(predict_only)}",
            'if [[ "$RJOB_PREDICT_ONLY" == 1 ]]; then SUBMIT_ARGS+=(--predict-only true); fi',
            "printf '%s\\n' \"${SUBMIT_ARGS[@]}\"",
        )
    )
    result = subprocess.run(
        ["/usr/bin/bash", "--noprofile", "--norc", "-c", harness],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.splitlines()


def _run_unknown_publisher(
    *,
    parent: Path,
    program: str | None = None,
) -> subprocess.CompletedProcess[str]:
    paths = {
        "pending": parent / "submission-pending.json",
        "accepted": parent / "scheduler-readback-receipt.json",
        "unknown": parent / "submission-unknown.json",
        "submit_log": parent / "rjob-submit.log",
        "scheduler_expected": parent / "rjob-expected-contract.json",
        "worker_env": parent / "worker-env-contract.json",
        "capture_log": parent / "rjob-scheduler-capture.log",
        "structured": parent / "rjob-scheduler-receipt.json",
        "readback": parent / "rjob-get.log",
    }
    return subprocess.run(
        [
            "/usr/bin/python3",
            "-c",
            program or _unknown_publisher_program(),
            str(paths["pending"]),
            str(paths["accepted"]),
            str(paths["unknown"]),
            str(parent),
            "injected_failure",
            "",
            str(paths["submit_log"]),
            str(paths["scheduler_expected"]),
            str(paths["worker_env"]),
            str(paths["capture_log"]),
            str(paths["structured"]),
            str(paths["readback"]),
        ],
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


def test_output_infrastructure_first_run_is_secure_and_existing_roots_are_reused(
    tmp_path: Path,
) -> None:
    personal_root = tmp_path / "personal"
    personal_root.mkdir(mode=0o755)
    first = _run_output_initializer(personal_root)
    assert first.returncode == 0, first.stderr

    managed = (
        personal_root / "outputs",
        personal_root / "outputs" / "cwam-robocasa365-eval",
        personal_root / "outputs" / "cwam-robocasa365-eval" / "runs",
        personal_root / "outputs" / "cwam-robocasa365-eval" / "submission-reservations",
        personal_root / "outputs" / "cwam-robocasa365-eval" / "submission-reservations" / "by-experiment",
        personal_root / "outputs" / "cwam-robocasa365-eval" / "submission-reservations" / "by-logical-run",
    )
    identities = {}
    for path in managed:
        metadata = path.lstat()
        identities[path] = (metadata.st_dev, metadata.st_ino)
        assert path.resolve(strict=True) == path
        assert stat.S_ISDIR(metadata.st_mode)
        assert not stat.S_ISLNK(metadata.st_mode)
        assert stat.S_IMODE(metadata.st_mode) == 0o755
        assert (metadata.st_uid, metadata.st_gid) == (os.geteuid(), os.getegid())

    second = _run_output_initializer(personal_root)
    assert second.returncode == 0, second.stderr
    assert {path: (path.lstat().st_dev, path.lstat().st_ino) for path in managed} == identities


@pytest.mark.parametrize("unsafe_kind", ["symlink", "regular_file"])
def test_output_infrastructure_rejects_symlink_and_wrong_type(
    tmp_path: Path,
    unsafe_kind: str,
) -> None:
    personal_root = tmp_path / "personal"
    personal_root.mkdir(mode=0o755)
    if unsafe_kind == "symlink":
        outside = tmp_path / "outside"
        outside.mkdir()
        (personal_root / "outputs").symlink_to(outside, target_is_directory=True)
    else:
        (personal_root / "outputs").write_text("not a directory", encoding="utf-8")

    result = _run_output_initializer(personal_root)

    assert result.returncode != 0
    assert "must be a non-symlink directory" in result.stderr
    assert not (tmp_path / "outside" / "cwam-robocasa365-eval").exists()


def test_output_infrastructure_rejects_wrong_mode_and_ownership(tmp_path: Path) -> None:
    personal_root = tmp_path / "personal"
    personal_root.mkdir(mode=0o755)
    assert _run_output_initializer(personal_root).returncode == 0
    runs_root = personal_root / "outputs" / "cwam-robocasa365-eval" / "runs"

    runs_root.chmod(0o775)
    wrong_mode = _run_output_initializer(personal_root)
    assert wrong_mode.returncode != 0
    assert "unsafe mode" in wrong_mode.stderr
    runs_root.chmod(0o755)

    namespace = {"__name__": "initializer_unit_test"}
    exec(compile(_output_initializer_program(), "<output-initializer>", "exec"), namespace)
    with pytest.raises(SystemExit, match="unsafe ownership"):
        namespace["initialize"](
            personal_root,
            personal_root / "outputs" / "cwam-robocasa365-eval",
            runs_root,
            personal_root / "outputs" / "cwam-robocasa365-eval" / "submission-reservations",
            expected_uid=os.geteuid() + 1,
            expected_gid=os.getegid(),
        )


def test_output_infrastructure_partial_failure_keeps_safe_roots_reentrant(
    tmp_path: Path,
) -> None:
    personal_root = tmp_path / "personal"
    personal_root.mkdir(mode=0o755)
    submission_root = personal_root / "outputs" / "cwam-robocasa365-eval" / "submission-reservations"
    submission_root.mkdir(mode=0o755, parents=True)
    for path in (
        personal_root / "outputs",
        personal_root / "outputs" / "cwam-robocasa365-eval",
        submission_root,
    ):
        path.chmod(0o755)
    collision = submission_root / "by-logical-run"
    collision.write_text("wrong type", encoding="utf-8")

    failed = _run_output_initializer(personal_root)
    assert failed.returncode != 0
    assert "must be a non-symlink directory" in failed.stderr
    assert (personal_root / "outputs" / "cwam-robocasa365-eval" / "runs").is_dir()
    assert (submission_root / "by-experiment").is_dir()
    assert collision.is_file()

    collision.unlink()
    retried = _run_output_initializer(personal_root)
    assert retried.returncode == 0, retried.stderr
    assert (submission_root / "by-logical-run").is_dir()


def test_predict_only_cannot_initialize_or_reserve_output_infrastructure() -> None:
    script = SUBMITTER.read_text(encoding="utf-8")
    actual_branch = script.index('if [[ "$RJOB_PREDICT_ONLY" == 0 ]]; then')
    initializer_call = script.index("    initialize_output_infrastructure\n", actual_branch)
    predict_else = script.index("\nelse\n", initializer_call)
    submit_array = script.index("SUBMIT_ARGS=(\n")

    assert script.count("    initialize_output_infrastructure\n") == 1
    assert actual_branch < initializer_call < predict_else < submit_array
    assert "/usr/bin/mkdir -p --" not in script[actual_branch:predict_else]
    assert _render_submit_args(predict_only=True)[-2:] == ["--predict-only", "true"]


def test_gate_omits_host_network_and_formal_uses_the_bare_flag_once() -> None:
    for predict_only in (False, True):
        gate_arguments = _render_submit_args(
            predict_only=predict_only,
            host_network=False,
        )
        assert "--host-network" not in gate_arguments
        assert gate_arguments.count("true") == int(predict_only)

        arguments = _render_submit_args(
            predict_only=predict_only,
            host_network=True,
        )
        indices = [index for index, value in enumerate(arguments) if value == "--host-network"]
        assert indices == [arguments.index("--host-network")]
        index = indices[0]
        assert arguments[index : index + 2] == ["--host-network", "--store-host-nvme"]
        assert arguments.count("--host-network") == 1
        assert arguments.count("true") == int(predict_only)


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
        "--host-network",
        "node/gpu-lg-cmc-h-h200-0685.host.h.pjlab.org.cn",
    ):
        assert token in script
    assert "--host-network true" not in script
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
    assert "refusing to publish unknown after accepted" in script
    assert "refusing to publish accepted after submission-unknown" in script
    assert "accepted and unknown submission terminal records both exist" in script
    assert "SUBMISSION_TERMINAL_COMPLETE=0" in script
    assert "trap 'submission_exit \"$?\"' EXIT" in script
    assert "trap 'submission_signal INT 130' INT" in script
    assert "trap 'submission_signal TERM 143' TERM" in script
    experiment_lock = script.index('/usr/bin/mkdir --mode=0700 -- "$EXPERIMENT_RESERVATION_DIR"')
    exit_trap = script.index("trap 'submission_exit \"$?\"' EXIT")
    run_lock = script.index('/usr/bin/mkdir --mode=0700 -- "$RUN_RESERVATION_DIR"')
    assert experiment_lock < exit_trap < run_lock
    assert script.index("trap 'submission_exit \"$?\"' EXIT") < script.index(
        '"$TIMEOUT_BIN" --foreground --kill-after=10s 180s'
    )
    assert "os.link(temporary, target, follow_symlinks=False)" in script
    assert "if published:\n        target.unlink(missing_ok=True)" in script
    assert "SUBMISSION_TERMINAL_COMPLETE=1\ntrap - EXIT HUP INT TERM" in script


def test_unknown_publisher_handles_failure_before_pending_exists(tmp_path: Path) -> None:
    parent = tmp_path / "reservation"
    parent.mkdir(mode=0o700)

    result = _run_unknown_publisher(parent=parent)

    assert result.returncode == 0, result.stderr
    unknown = parent / "submission-unknown.json"
    payload = json.loads(unknown.read_text(encoding="utf-8"))
    assert payload["submission_reservation"] is None
    assert payload["available_artifacts"] == {}
    assert stat.S_IMODE(unknown.stat().st_mode) == 0o444
    assert stat.S_IMODE(parent.stat().st_mode) == 0o555


def test_unknown_publisher_removes_partial_terminal_on_post_link_failure(tmp_path: Path) -> None:
    parent = tmp_path / "reservation"
    parent.mkdir(mode=0o700)
    pending = parent / "submission-pending.json"
    pending.write_text('{"status":"submitting"}\n', encoding="utf-8")
    program = _unknown_publisher_program().replace(
        "    published = True\n",
        "    published = True\n    raise RuntimeError('injected post-link failure')\n",
        1,
    )

    result = _run_unknown_publisher(parent=parent, program=program)

    assert result.returncode != 0
    assert "injected post-link failure" in result.stderr
    assert not (parent / "submission-unknown.json").exists()
    assert stat.S_IMODE(parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(pending.stat().st_mode) == 0o444
    assert not any(path.name.startswith(".submission-unknown.json.") for path in parent.iterdir())


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


def test_task_registry_bootstrap_import_is_cwd_independent() -> None:
    helper = SUBMITTER.with_name("cwam_robocasa365_eval_evidence.py").resolve()
    program = f"""
import importlib.util
import pathlib
import sys

helper = pathlib.Path({str(helper)!r})
sys.path.insert(0, str(helper.parent))
spec = importlib.util.spec_from_file_location("cwam_eval_submit_contract", helper)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert len(module.ATOMIC_TASKS) == 17
"""
    result = subprocess.run(
        ["/usr/bin/python3", "-c", program],
        cwd="/",
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr


def test_formal_prior_gate_is_verified_before_any_reservation_or_submission() -> None:
    script = SUBMITTER.read_text(encoding="utf-8")
    verification = "verify-prior-gate \\\n"
    reservation = '/usr/bin/mkdir --mode=0700 -- "$EXPERIMENT_RESERVATION_DIR"'
    submission = '"$TIMEOUT_BIN" --foreground --kill-after=10s 180s'

    assert verification in script
    assert '--current-checkpoint-manifest-sha256 "$EXPECTED_CHECKPOINT_MANIFEST_SHA256"' in script
    assert '--current-source-dcp-export-receipt-sha256 "$EXPECTED_SOURCE_DCP_EXPORT_RECEIPT_SHA256"' in script
    asset_manifest_arg = '--current-asset-manifest-sha256 "$PINNED_ASSET_MANIFEST_SHA256"'
    asset_receipt_arg = '--current-asset-publisher-receipt-sha256 "$PINNED_ASSET_RECEIPT_SHA256"'
    assert asset_manifest_arg in script
    assert asset_receipt_arg in script
    assert script.index("readonly PINNED_ASSET_MANIFEST_SHA256=") < script.index(asset_manifest_arg)
    assert script.index("readonly PINNED_ASSET_RECEIPT_SHA256=") < script.index(asset_receipt_arg)
    assert script.index(verification) < script.index(reservation)
    assert script.index(verification) < script.index(submission)


def test_all_live_model_inputs_and_repository_heads_are_verified_before_reservation() -> None:
    script = SUBMITTER.read_text(encoding="utf-8")
    reservation = script.index('/usr/bin/mkdir --mode=0700 -- "$EXPERIMENT_RESERVATION_DIR"')

    for token in (
        '"checkpoint_manifest|$CHECKPOINT_MANIFEST|$EXPECTED_CHECKPOINT_MANIFEST_SHA256"',
        '"training_config|$TRAINING_CONFIG|$EXPECTED_TRAINING_CONFIG_SHA256"',
        '"action_stats|$ACTION_STATS|$EXPECTED_ACTION_STATS_SHA256"',
        '"cosmos_environment_manifest|$COSMOS_ENVIRONMENT_MANIFEST|$EXPECTED_COSMOS_ENVIRONMENT_MANIFEST_SHA256"',
        '"robocasa_environment_manifest|$ROBOCASA_ENVIRONMENT_MANIFEST|$EXPECTED_ROBOCASA_ENVIRONMENT_MANIFEST_SHA256"',
        '"$CHECKPOINT_HELPER" verify',
        '"DeMiAn|$DEMIAN_REPO|$PINNED_DEMIAN_COMMIT"',
        '"RoboCasa|$ROBOCASA_REPO|$EXPECTED_ROBOCASA_COMMIT"',
        '"robosuite|$ROBOSUITE_REPO|$EXPECTED_ROBOSUITE_COMMIT"',
        "status --porcelain --untracked-files=all",
    ):
        assert token in script
        assert script.index(token) < reservation


def test_full_asset_repository_and_intent_preflights_precede_reservation() -> None:
    script = SUBMITTER.read_text(encoding="utf-8")

    asset_preflight = script.index('ASSET_PREFLIGHT_JSON="$(')
    full_asset_verify = script.index(
        '"$PYTHON3_BIN" "$ASSET_VERIFY_HELPER" verify \\\n        --asset-root "$ROBOCASA_REPO"',
        asset_preflight,
    )
    repository_preflight = script.index('TRACKED_REPOSITORY_PREFLIGHT_JSON="$(')
    tracked_file_verification = script.index(
        "reports[name] = module._verify_git_repository(",
        repository_preflight,
    )
    verify_all_tracked_files = script.index(
        "verify_tracked_files=True",
        tracked_file_verification,
    )
    combined_preflight = script.index('SUBMISSION_PREFLIGHT_JSON="$(')
    intent_arguments = script.index("INTENT_VERIFY_ARGS=(")
    intent_verification = script.index(
        '"$PYTHON3_BIN" "$INTENT_HELPER" "${INTENT_VERIFY_ARGS[@]}"',
        intent_arguments,
    )
    reservation = script.index('/usr/bin/mkdir --mode=0700 -- "$EXPERIMENT_RESERVATION_DIR"')
    submission = script.index('"$TIMEOUT_BIN" --foreground --kill-after=10s 180s')

    assert (
        asset_preflight
        < full_asset_verify
        < repository_preflight
        < tracked_file_verification
        < verify_all_tracked_files
        < combined_preflight
        < intent_arguments
        < intent_verification
        < reservation
        < submission
    )
    for token in (
        'if asset.get("status") != "passed":',
        '"asset_tree_full_hash_match": True',
        '"tracked_regular_files_match_head_blobs": True',
        '--manifest "$EVAL_INTENT_MANIFEST"',
        '--expected-manifest-sha256 "$EXPECTED_EVAL_INTENT_MANIFEST_SHA256"',
        '--source-dcp-export-receipt "$SOURCE_DCP_EXPORT_RECEIPT"',
        '--expected-source-dcp-export-receipt-sha256 "$EXPECTED_SOURCE_DCP_EXPORT_RECEIPT_SHA256"',
    ):
        assert token in script
        assert script.index(token) < reservation


def test_eval_intent_and_source_export_receipt_are_forwarded_to_worker() -> None:
    script = SUBMITTER.read_text(encoding="utf-8")
    worker_block = re.search(
        r"readonly WORKER_ENV_NAMES=\(\n(.*?)\n\)",
        script,
        re.DOTALL,
    )

    assert worker_block is not None
    forwarded = set(re.findall(r"^\s+([A-Z0-9_]+)\s*$", worker_block.group(1), re.MULTILINE))
    assert {
        "EVAL_INTENT_MANIFEST",
        "EXPECTED_EVAL_INTENT_MANIFEST_SHA256",
        "SOURCE_DCP_EXPORT_RECEIPT",
        "EXPECTED_SOURCE_DCP_EXPORT_RECEIPT_SHA256",
    } <= forwarded
    assert 'for name in "${WORKER_ENV_NAMES[@]}"; do' in script
    assert 'WORKER_SCRIPT+="export $name=$quoted"' in script
    assert script.index("readonly WORKER_ENV_NAMES=(") < script.index('WORKER_SCRIPT+="export $name=$quoted"')
