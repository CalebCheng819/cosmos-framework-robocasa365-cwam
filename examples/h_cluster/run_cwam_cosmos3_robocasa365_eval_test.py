# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Static fail-closed contract tests for the RoboCasa365 eval runner."""

from __future__ import annotations

import re
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
        "PINNED_WAN22_VAE_PATH",
        "PINNED_WAN22_VAE_SHA256",
        "WAN2PT2_VAE_VERIFIED",
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
    assert '--submitted-at-utc "$EVAL_SUBMITTED_AT_UTC"' in script
    assert '--cosmos-source-manifest "$SOURCE_MANIFEST"' in script
    assert '--expected-cosmos-source-manifest-sha256 "$EXPECTED_SOURCE_MANIFEST_SHA256"' in script
    assert '--submission-verification-receipt "$SUBMISSION_VERIFICATION_RECEIPT"' in script
    assert '--asset-verification-receipt "$ASSET_VERIFICATION_RECEIPT"' in script
    assert 'EVAL_RUNTIME_CAPTURE_NONCE="$(' not in script
    runtime_helper = Path(__file__).with_name("capture_robocasa365_eval_runtime.py").read_text(encoding="utf-8")
    assert "EGLGLContext(max_width=64, max_height=64, device_id=shard_index)" in runtime_helper
    assert '"CUDA_VISIBLE_DEVICES": str(shard_index)' in runtime_helper
    assert '"MUJOCO_EGL_DEVICE_ID": str(shard_index)' in runtime_helper
    assert "robocasa_egl_contexts_created_per_shard" in runtime_helper


def test_runner_verifies_submission_and_assets_before_egl_or_servers() -> None:
    script = RUNNER.read_text(encoding="utf-8")

    submission = script.index('"$COSMOS_PY" "$COSMOS_SOURCE/$SUBMISSION_VERIFY_HELPER_REL"')
    assets = script.index('"$COSMOS_PY" "$COSMOS_SOURCE/$ASSET_VERIFY_HELPER_REL"')
    runtime = script.index('"$COSMOS_PY" "$COSMOS_SOURCE/$RUNTIME_RECEIPT_HELPER_REL"')
    policy_servers = script.index('server_log="$CONTROL_DIR/server-shard${shard}.log"')
    assert submission < assets < runtime < policy_servers
    for token in (
        "EXPECTED_EVAL_SUBMISSION_PENDING_SHA256",
        "EVAL_SCHEDULER_READBACK_RECEIPT",
        "EVAL_SUBMISSION_UNKNOWN_PATH",
        "submission-unknown terminal record exists",
        "KUBEBRAIN_RJOB_NAME",
        "KUBEBRAIN_RJOB_UID",
        "cwam_robocasa_asset_manifest.py",
        "asset-tree-verification.json",
        "EXPECTED_ROBOCASA_ASSET_MANIFEST_SHA256",
        "EXPECTED_ROBOCASA_ASSET_RECEIPT_SHA256",
        "__EGL_VENDOR_LIBRARY_FILENAMES",
        "__GLX_VENDOR_LIBRARY_NAME",
        "LD_PRELOAD",
        "LD_AUDIT",
    ):
        assert token in script


def test_runner_reverifies_all_live_inputs_after_rollouts_before_summary() -> None:
    script = RUNNER.read_text(encoding="utf-8")

    assert "asset_verification_elapsed_seconds" in script
    assert "checkpoint_verification_elapsed_seconds" in script
    evaluators_reaped = script.index("EVALUATOR_PIDS=()")
    servers_stopped = script.index('stop_child_pids "${SERVER_PIDS[@]}"', evaluators_reaped)
    post_rollout = script.index("verify-post-rollout")
    summary = script.index("SUMMARIZE_ARGS=(")
    vae_post_rollout = script.index("wan2pt2_vae_post_rollout")
    assert evaluators_reaped < servers_stopped < vae_post_rollout < post_rollout < summary
    assert '--contract "$CONTRACT_PATH"' in script[post_rollout:summary]
    assert '--output "$POST_ROLLOUT_VERIFICATION_RECEIPT"' in script[post_rollout:summary]
    assert '--post-rollout-verification-receipt "$POST_ROLLOUT_VERIFICATION_RECEIPT"' in script
    assert '"$POST_ROLLOUT_VERIFICATION_RECEIPT" \\' in script


def test_runner_threads_a_bounded_asset_worker_count_through_every_full_hash() -> None:
    script = RUNNER.read_text(encoding="utf-8")

    assert ': "${ASSET_VERIFY_WORKERS:=1}"' in script
    assert "readonly MAX_ASSET_VERIFY_WORKERS=64" in script
    assert 'require_positive_int ASSET_VERIFY_WORKERS "$ASSET_VERIFY_WORKERS"' in script
    assert 'die "ASSET_VERIFY_WORKERS must be <= $MAX_ASSET_VERIFY_WORKERS"' in script
    assert 'verify \\\n    --asset-root "$ROBOCASA_ASSET_ROOT" \\\n    --workers "$ASSET_VERIFY_WORKERS" \\\n' in script
    assert script.count('--asset-workers "$ASSET_VERIFY_WORKERS"') == 2


def test_runner_reaggregates_only_after_raw_results_are_sealed() -> None:
    script = RUNNER.read_text(encoding="utf-8")
    raw_seal = script.index('chmod 0555 "$RESULTS_DIR"')
    final_verify = script.index("verify-sealed-evidence")
    success = script.index("ROBOCASA365_EVAL_EXECUTION_SUCCESS")

    assert raw_seal < final_verify < success
    assert '--output "$SEALED_EVIDENCE_VERIFICATION_RECEIPT"' in script[final_verify:success]
    assert "sealed evidence verification receipt is not mode 0444" in script
    assert script.index("require_accepted_submission_terminal_state", final_verify) < success
    directory_seal = script.index('chmod 0555 \\\n    "$CONTROL_DIR"', final_verify)
    stable_read = script.index('"O_NOFOLLOW"', directory_seal)
    effect_read = script.index('effect_pass = receipt.get("evaluation", {}).get("effect_pass")', stable_read)
    assert final_verify < directory_seal < stable_read < effect_read < success
    assert "receipt changed during final verdict read" in script
    assert "receipt SHA-256 changed after directory sealing" in script


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


def test_runner_accepts_manifest_only_immutable_cosmos_snapshot() -> None:
    script = RUNNER.read_text(encoding="utf-8")

    manifest_verification = script.index(
        'verify_sha256 source_manifest "$SOURCE_MANIFEST" "$EXPECTED_SOURCE_MANIFEST_SHA256"'
    )
    optional_git_verification = script.index('if [[ -e "$COSMOS_SOURCE/.git" || -L "$COSMOS_SOURCE/.git" ]]; then')
    demian_verification = script.index('DEMIAN_COMMIT="$(git -c safe.directory="$DEMIAN_REPO"')
    source_git_block = script[optional_git_verification:demian_verification]

    assert '[[ "$SOURCE_MANIFEST" == "$COSMOS_SOURCE/SNAPSHOT_MANIFEST.json" ]]' in script
    assert '[[ "$(stat -c %a -- "$SOURCE_MANIFEST")" == 444 ]]' in script
    assert 'manifest.get("source_head") != sys.argv[2]' in script
    assert 'manifest.get("source_dirty") is not False' in script
    assert manifest_verification < optional_git_verification < demian_verification
    assert "rev-parse --show-toplevel" in source_git_block
    assert "rev-parse HEAD" in source_git_block
    assert "status --porcelain --untracked-files=all" in source_git_block
    assert 'for repo in "$DEMIAN_REPO" "$ROBOCASA_REPO" "$ROBOSUITE_REPO"; do' in script
    assert 'for repo in "$COSMOS_SOURCE"' not in script


def test_runner_rejects_reused_outputs_and_non_nvme_scratch() -> None:
    script = RUNNER.read_text(encoding="utf-8")

    assert "Refusing to reuse EVAL_OUTPUT_ROOT" in script
    assert "Refusing to reuse experiment output root" in script
    assert 'mkdir -- "$EXPERIMENT_OUTPUT_ROOT"' in script
    assert "Refusing to reuse EVAL_NVME_ROOT" in script
    assert "EVAL_NVME_ROOT must be a strict child of /nvme" in script
    assert "/nvme is not backed by /dev/nvme1n1" in script
    assert "/nvme/.cwam-robocasa365-eval/$EVAL_LOGICAL_RUN_ID" in script
    assert 'readonly EVAL_NVME_PARENT="/nvme/.cwam-robocasa365-eval"' in script
    assert 'mkdir --mode=0700 -- "$EVAL_NVME_PARENT" 2>/dev/null || true' in script
    assert '[[ -d "$EVAL_NVME_PARENT" && ! -L "$EVAL_NVME_PARENT" ]]' in script
    assert '"$(realpath -e -- "$EVAL_NVME_PARENT")" == "$EVAL_NVME_PARENT"' in script
    assert '"$(stat -c %u:%g -- "$EVAL_NVME_PARENT")" == 0:0' in script
    assert '"$(stat -c %a -- "$EVAL_NVME_PARENT")" == 700' in script
    assert '[[ ! -e "$EVAL_NVME_ROOT" && ! -L "$EVAL_NVME_ROOT" ]]' in script
    assert 'mkdir --mode=0700 -- "$EVAL_NVME_ROOT"' in script
    assert '/usr/bin/rm --one-file-system -rf -- "$EVAL_NVME_ROOT"' in script
    assert "scoped NVMe cleanup did not remove" in script
    assert script.index("trap 'cleanup $?' EXIT") < script.index(
        'mkdir --mode=0700 -- "$EVAL_NVME_ROOT"'
    )


def test_formal_mode_can_select_the_complete_split() -> None:
    script = RUNNER.read_text(encoding="utf-8")

    assert "[[ -v EVAL_TASKS_CSV ]]" in script
    assert ': "${EVAL_TASKS_CSV:=OpenDrawer}"' not in script
    assert "formal evaluation must use the complete canonical split" in script
    assert "formal evaluation requires NUM_SHARDS=8" in script


def test_formal_prior_gate_is_reverified_before_output_or_egl() -> None:
    script = RUNNER.read_text(encoding="utf-8")
    verification = "verify-prior-gate \\\n"

    assert verification in script
    assert '--prior-gate-evidence "$PRIOR_GATE_EVIDENCE"' in script
    assert '--expected-prior-gate-evidence-sha256 "$EXPECTED_PRIOR_GATE_EVIDENCE_SHA256"' in script
    assert '--current-source-dcp-export-receipt-sha256 "$EXPECTED_SOURCE_DCP_EXPORT_RECEIPT_SHA256"' in script
    assert '--current-asset-manifest-sha256 "$EXPECTED_ROBOCASA_ASSET_MANIFEST_SHA256"' in script
    assert '--current-asset-publisher-receipt-sha256 "$EXPECTED_ROBOCASA_ASSET_RECEIPT_SHA256"' in script
    assert '--current-wan2pt2-vae-path "$PINNED_WAN22_VAE_PATH"' in script
    assert '--current-wan2pt2-vae-sha256 "$PINNED_WAN22_VAE_SHA256"' in script
    assert script.index(verification) < script.index('mkdir -- "$EXPERIMENT_OUTPUT_ROOT"')
    assert script.index(verification) < script.index("MUJOCO_GL=egl")


def test_contract_directly_binds_intent_and_source_dcp_receipt() -> None:
    script = RUNNER.read_text(encoding="utf-8")
    contract = script.index("CONTRACT_ARGS=(")
    rollout = script.index("EVALUATOR_ARGS=(")
    contract_block = script[contract:rollout]

    for token in (
        '--eval-intent-manifest "$EVAL_INTENT_MANIFEST"',
        '--source-dcp-export-receipt "$SOURCE_DCP_EXPORT_RECEIPT"',
        '--wan2pt2-vae "$PINNED_WAN22_VAE_PATH"',
        '--expected-wan2pt2-vae-sha256 "$PINNED_WAN22_VAE_SHA256"',
    ):
        assert token in contract_block


def test_runner_binds_wan2pt2_vae_into_intent_verification() -> None:
    script = RUNNER.read_text(encoding="utf-8")
    start = script.index("INTENT_VERIFY_ARGS=(")
    end = script.index('    "${INTENT_VERIFY_ARGS[@]}" >/dev/null', start)
    intent_block = script[start:end]

    assert '--wan2pt2-vae "$PINNED_WAN22_VAE_PATH"' in intent_block
    assert '--expected-wan2pt2-vae-sha256 "$PINNED_WAN22_VAE_SHA256"' in intent_block


def test_runner_has_no_hidden_phase_or_resource_defaults() -> None:
    script = RUNNER.read_text(encoding="utf-8")

    for name in (
        "EVAL_STAGE",
        "EVAL_SPLIT",
        "EPISODES_PER_TASK",
        "NUM_SHARDS",
        "BASE_SEED",
        "MAX_STEPS",
        "OPEN_LOOP_HORIZON",
        "POLICY_SAMPLER",
        "POLICY_SEED",
        "POLICY_GUIDANCE",
        "POLICY_NUM_STEPS",
        "BASE_PORT",
        "SERVER_READY_TIMEOUT_SECONDS",
        "ALLOWED_OUTPUT_PARENT",
        "EVAL_NVME_ROOT",
    ):
        assert f"${{{name}:=" not in script
        assert f"${{{name}:?" in script


def test_signal_traps_cannot_report_success() -> None:
    script = RUNNER.read_text(encoding="utf-8")

    assert "trap 'cleanup 130' INT" in script
    assert "trap 'cleanup 143' TERM" in script
    assert "trap cleanup EXIT INT TERM" not in script
    assert "readonly CLEANUP_GRACE_SECONDS=30" in script
    assert "deadline=$((SECONDS + CLEANUP_GRACE_SECONDS))" in script
    assert 'kill -KILL "$pid"' in script


def test_all_embedded_python_programs_compile() -> None:
    script = RUNNER.read_text(encoding="utf-8")
    programs = re.findall(r"<<'PY'\n(.*?)\nPY(?:\n|\Z)", script, re.DOTALL)

    assert len(programs) == 6
    for index, program in enumerate(programs, start=1):
        compile(program, f"{RUNNER.name}:heredoc-{index}", "exec")
