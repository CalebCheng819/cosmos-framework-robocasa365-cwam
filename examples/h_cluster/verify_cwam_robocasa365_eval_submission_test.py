# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import copy
import hashlib
import json
import stat
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from examples.h_cluster import cwam_robocasa365_eval_evidence as evidence
from examples.h_cluster import verify_cwam_robocasa365_eval_submission as submission

pytestmark = [pytest.mark.L0, pytest.mark.CPU]

NOW = datetime(2026, 7, 29, 2, 0, tzinfo=timezone.utc)
SUBMITTED = NOW - timedelta(minutes=10)


def _write_json(path: Path, value: object, mode: int = 0o444) -> tuple[Path, str]:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    path.chmod(mode)
    return path, hashlib.sha256(payload).hexdigest()


def _descriptor(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


@pytest.fixture
def valid_records(tmp_path: Path) -> dict[str, object]:
    scheduler_expected = {
        "showname": "cwam-eval-gate",
        "quota_group": "eailabagent_gpu",
        "private_machine": "group",
        "task_type": "normal",
        "priority": 9,
        "restart_policy": "Never",
        "termination_grace_period_seconds": 60,
        "backoff_limit": 1,
        "replicas": 1,
        "image": submission.PINNED_IMAGE,
        "cpu": 16,
        "memory_mib": 187_500,
        "gpu": 1,
        "negative_tags": submission.PINNED_NEGATIVE_TAGS,
        "store_host_nvme": True,
        "mounts": submission.PINNED_MOUNTS,
    }
    submitted = SUBMITTED.isoformat().replace("+00:00", "Z")
    expires = (
        (SUBMITTED + timedelta(seconds=submission.DEFAULT_MAX_AGE_SECONDS))
        .isoformat()
        .replace(
            "+00:00",
            "Z",
        )
    )
    repositories = {}
    for index, name in enumerate(("demian", "robocasa", "robosuite"), start=1):
        commit = str(index) * 40
        repositories[name] = {
            "root": str(tmp_path / name),
            "expected_commit": commit,
            "actual_commit": commit,
            "clean": True,
            "tracked_files": {
                "all_tracked_regular_files_match_head_blobs": True,
            },
        }
    preflight = {
        "asset_tree": {
            "schema_version": 1,
            "kind": "cwam_robocasa_asset_tree_verification_receipt",
            "status": "passed",
            "manifest_sha256": "4" * 64,
            "publisher_receipt_sha256": "5" * 64,
        },
        "repositories": repositories,
        "validation": {
            "asset_tree_full_hash_match": True,
            "repository_heads_clean": True,
            "tracked_regular_files_match_head_blobs": True,
        },
    }
    preflight_sha256 = hashlib.sha256(
        json.dumps(preflight, separators=(",", ":"), sort_keys=True).encode(),
    ).hexdigest()
    intent_value = {
        "schema_version": 1,
        "kind": "cwam_robocasa365_eval_intent",
        "identity": {
            "experiment_id": "CWAM-E35-EVAL-GATE",
            "logical_run_id": "CWAM-E35-EVAL-GATE-S42",
            "run_name": "CWAM-E35-EVAL-GATE-S42",
            "stage": "one_episode_gate",
            "evaluation_class": "exploratory",
        },
        "protocol": {
            "split": "atomic",
            "tasks": ["OpenDrawer"],
            "episodes_per_task": 1,
            "num_shards": 1,
            "base_seed": 42,
        },
        "thresholds": {
            "minimum_mean_success_rate": 0.0,
            "minimum_successful_tasks": 0,
        },
        "source": {
            "root": str(tmp_path / "source"),
            "commit": "b" * 40,
            "git_pushed": False,
            "manifest": {
                "path": str(tmp_path / "source" / "SNAPSHOT_MANIFEST.json"),
                "size_bytes": 1,
                "sha256": "c" * 64,
            },
        },
        "serving_checkpoint": {},
        "artifacts": {},
        "repositories": {},
        "runtime": {},
        "resources": {
            "cpu": 16,
            "memory_mib": 187_500,
            "gpu": 1,
            "quota_group": "eailabagent_gpu",
            "negative_tag": submission.PINNED_NEGATIVE_TAGS[0],
            "task_type": "normal",
            "private_machine": "group",
            "priority": 9,
            "restart_policy": "never",
            "backoff_limit": 1,
            "replicas": 1,
        },
        "prior_gate": None,
    }
    intent_path, _ = _write_json(tmp_path / "intent" / "eval-intent.json", intent_value)
    reservation_value = {
        "schema_version": 1,
        "kind": submission.RESERVATION_KIND,
        "status": "submitting",
        "submitted_at_utc": submitted,
        "expires_at_utc": expires,
        "experiment_id": "CWAM-E35-EVAL-GATE",
        "logical_run_id": "CWAM-E35-EVAL-GATE-S42",
        "run_name": "CWAM-E35-EVAL-GATE-S42",
        "submission_nonce": "a" * 64,
        "rjob_showname": "cwam-eval-gate",
        "source_commit": "b" * 40,
        "source_manifest_sha256": "c" * 64,
        "evaluation": {
            "stage": "one_episode_gate",
            "split": "atomic",
            "tasks_csv": "OpenDrawer",
            "episodes_per_task": 1,
            "num_shards": 1,
            "base_seed": 42,
            "prior_gate_evidence": None,
        },
        "eval_intent_manifest": _descriptor(intent_path),
        "preflight": preflight,
        "preflight_sha256": preflight_sha256,
        "scheduler_expected": scheduler_expected,
    }
    reservation, reservation_sha = _write_json(
        tmp_path / "reservation" / "submission-pending.json",
        reservation_value,
    )
    artifact_paths = {}
    for name in (
        "submission_log",
        "scheduler_expected_contract",
        "scheduler_capture_log",
        "scheduler_structured_receipt",
        "scheduler_text_readback",
    ):
        path = tmp_path / "reservation" / f"{name}.txt"
        path.write_text(f"{name}\n", encoding="utf-8")
        path.chmod(0o444)
        artifact_paths[name] = path
    readback_value = {
        "schema_version": 1,
        "kind": submission.READBACK_KIND,
        "status": "accepted",
        "accepted_at_utc": (SUBMITTED + timedelta(seconds=5)).isoformat().replace("+00:00", "Z"),
        "experiment_id": reservation_value["experiment_id"],
        "logical_run_id": reservation_value["logical_run_id"],
        "run_name": reservation_value["run_name"],
        "submission_nonce": reservation_value["submission_nonce"],
        "submitted_at_utc": submitted,
        "expires_at_utc": expires,
        "scheduler_rjob_name": "cwam-eval-gate-12345678",
        "scheduler_rjob_uid": "11111111-2222-4333-8444-555555555555",
        "scheduler_started_at_utc": (SUBMITTED + timedelta(seconds=3))
        .isoformat()
        .replace(
            "+00:00",
            "Z",
        ),
        "worker_command_sha256": "d" * 64,
        "submission_reservation": {
            "path": str(reservation),
            "sha256": reservation_sha,
        },
        "scheduler_expected": scheduler_expected,
        "artifacts": {name: _descriptor(path) for name, path in artifact_paths.items()},
    }
    readback, _ = _write_json(
        tmp_path / "reservation" / "scheduler-readback-receipt.json",
        readback_value,
    )
    return {
        "reservation": reservation,
        "reservation_sha": reservation_sha,
        "reservation_value": reservation_value,
        "readback": readback,
        "readback_value": readback_value,
        "output": tmp_path / "control" / "submission-verification.json",
        "artifact_paths": artifact_paths,
    }


def _arguments(records: dict[str, object]) -> dict[str, object]:
    reservation = records["reservation_value"]
    assert isinstance(reservation, dict)
    output = records["output"]
    assert isinstance(output, Path)
    output.parent.mkdir(exist_ok=True)
    return {
        "reservation_path": records["reservation"],
        "expected_reservation_sha256": records["reservation_sha"],
        "scheduler_readback_receipt_path": records["readback"],
        "submission_unknown_path": records["readback"].with_name("submission-unknown.json"),
        "experiment_id": reservation["experiment_id"],
        "logical_run_id": reservation["logical_run_id"],
        "run_name": reservation["run_name"],
        "submission_nonce": reservation["submission_nonce"],
        "submitted_at_utc": reservation["submitted_at_utc"],
        "observed_scheduler_rjob_name": "cwam-eval-gate-12345678",
        "observed_scheduler_rjob_uid": "11111111-2222-4333-8444-555555555555",
        "max_age_seconds": submission.DEFAULT_MAX_AGE_SECONDS,
        "max_future_skew_seconds": submission.DEFAULT_MAX_FUTURE_SKEW_SECONDS,
        "max_start_to_verification_seconds": (submission.DEFAULT_MAX_START_TO_VERIFICATION_SECONDS),
        "output": output,
        "clock": lambda: NOW,
    }


def test_happy_path_seals_external_identity_and_scheduler_artifacts(
    valid_records: dict[str, object],
) -> None:
    receipt, digest = submission.verify_submission(**_arguments(valid_records))
    output = valid_records["output"]
    assert isinstance(output, Path)
    assert receipt["identity"]["submission_nonce"] == "a" * 64
    assert receipt["freshness"]["age_seconds"] == 600.0
    assert receipt["scheduler"]["rjob_name"] == "cwam-eval-gate-12345678"
    assert receipt["scheduler"]["rjob_uid"] == "11111111-2222-4333-8444-555555555555"
    assert receipt["validation"] == {
        "submit_identity_matches": True,
        "freshness_window_passed": True,
        "scheduler_readback_matches": True,
        "scheduler_artifact_hashes_match": True,
        "eval_intent_manifest_matches": True,
        "pre_submission_full_inputs_passed": True,
        "submission_unknown_absent": True,
    }
    assert hashlib.sha256(output.read_bytes()).hexdigest() == digest
    assert stat.S_IMODE(output.stat().st_mode) == 0o444


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda value: value.update(submission_nonce="e" * 64), "submission_nonce mismatch"),
        (
            lambda value: value["scheduler_expected"].update(gpu=7),
            "resource contract does not match",
        ),
        (
            lambda value: value["scheduler_expected"].update(image="registry.example/cwam@sha256:" + "1" * 64),
            "audited digest",
        ),
        (
            lambda value: value["scheduler_expected"].update(
                mounts=["gpfs://wrong:/wrong", submission.PINNED_MOUNTS[1]]
            ),
            "mounts is not exact",
        ),
        (
            lambda value: value["evaluation"].update(episodes_per_task=20),
            "one_episode_gate contract",
        ),
        (lambda value: value.update(unexpected=True), "keys mismatch"),
    ],
)
def test_reservation_drift_is_rejected(
    valid_records: dict[str, object],
    mutation,
    match: str,
) -> None:
    value = copy.deepcopy(valid_records["reservation_value"])
    mutation(value)
    reservation = valid_records["reservation"]
    assert isinstance(reservation, Path)
    reservation.chmod(0o600)
    reservation, digest = _write_json(reservation, value)
    arguments = _arguments(valid_records)
    arguments["expected_reservation_sha256"] = digest
    with pytest.raises(submission.SubmissionVerificationError, match=match):
        submission.verify_submission(**arguments)


def test_stale_future_and_wrong_pinned_window_are_rejected(
    valid_records: dict[str, object],
) -> None:
    arguments = _arguments(valid_records)
    arguments["clock"] = lambda: SUBMITTED + timedelta(seconds=submission.DEFAULT_MAX_AGE_SECONDS + 1)
    with pytest.raises(submission.SubmissionVerificationError, match="stale"):
        submission.verify_submission(**arguments)

    arguments = _arguments(valid_records)
    arguments["clock"] = lambda: SUBMITTED - timedelta(seconds=submission.DEFAULT_MAX_FUTURE_SKEW_SECONDS + 1)
    with pytest.raises(submission.SubmissionVerificationError, match="future"):
        submission.verify_submission(**arguments)

    arguments = _arguments(valid_records)
    arguments["max_age_seconds"] = submission.DEFAULT_MAX_AGE_SECONDS - 1
    with pytest.raises(submission.SubmissionVerificationError, match="must be exactly"):
        submission.verify_submission(**arguments)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda value: value.update(status="queued"), "schema/kind/status"),
        (lambda value: value.update(submission_nonce="f" * 64), "does not match reservation"),
        (
            lambda value: value["scheduler_expected"].update(cpu=15),
            "differs from reservation",
        ),
        (
            lambda value: value["submission_reservation"].update(sha256="0" * 64),
            "exact reservation",
        ),
        (lambda value: value.update(extra=True), "keys mismatch"),
    ],
)
def test_scheduler_readback_drift_is_rejected(
    valid_records: dict[str, object],
    mutation,
    match: str,
) -> None:
    value = copy.deepcopy(valid_records["readback_value"])
    mutation(value)
    readback = valid_records["readback"]
    assert isinstance(readback, Path)
    readback.chmod(0o600)
    _write_json(readback, value)
    with pytest.raises(submission.SubmissionVerificationError, match=match):
        submission.verify_submission(**_arguments(valid_records))


def test_artifact_hash_drift_symlinks_and_output_reuse_fail_closed(
    valid_records: dict[str, object],
    tmp_path: Path,
) -> None:
    artifact_paths = valid_records["artifact_paths"]
    assert isinstance(artifact_paths, dict)
    artifact = artifact_paths["scheduler_capture_log"]
    assert isinstance(artifact, Path)
    artifact.chmod(0o600)
    artifact.write_text("drift\n", encoding="utf-8")
    with pytest.raises(submission.SubmissionVerificationError, match="does not match"):
        submission.verify_submission(**_arguments(valid_records))

    records = valid_records
    reservation = records["reservation"]
    assert isinstance(reservation, Path)
    alias = tmp_path / "reservation-alias.json"
    alias.symlink_to(reservation)
    arguments = _arguments(records)
    arguments["reservation_path"] = alias
    with pytest.raises(submission.SubmissionVerificationError, match="symlink"):
        submission.verify_submission(**arguments)

    artifact.write_text("scheduler_capture_log\n", encoding="utf-8")
    artifact.chmod(0o444)
    output = records["output"]
    assert isinstance(output, Path)
    output.write_text("owner\n", encoding="utf-8")
    with pytest.raises(submission.SubmissionVerificationError, match="refusing to overwrite"):
        submission.verify_submission(**_arguments(records))


def test_output_publication_race_never_deletes_the_competing_owner(
    valid_records: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = valid_records["output"]
    assert isinstance(output, Path)

    def racing_link(_source: Path, target: Path, **_kwargs: object) -> None:
        Path(target).write_text("competing owner\n", encoding="utf-8")
        raise FileExistsError(target)

    monkeypatch.setattr(submission.os, "link", racing_link)
    with pytest.raises(FileExistsError):
        submission.verify_submission(**_arguments(valid_records))
    assert output.read_text(encoding="utf-8") == "competing owner\n"


@pytest.mark.parametrize(
    ("argument", "value", "match"),
    [
        ("experiment_id", ".", "path-safe characters"),
        ("logical_run_id", "../escape", "path-safe characters"),
        ("logical_run_id", "OTHER-RUN", "namespaced below"),
        ("run_name", "different-run", "must equal logical_run_id"),
    ],
)
def test_run_identity_is_strict_and_namespaced(
    valid_records: dict[str, object],
    argument: str,
    value: str,
    match: str,
) -> None:
    arguments = _arguments(valid_records)
    arguments[argument] = value
    with pytest.raises(submission.SubmissionVerificationError, match=match):
        submission.verify_submission(**arguments)


def test_phase_evaluation_contracts_are_exact(tmp_path: Path) -> None:
    assert submission.ATOMIC_TASKS == frozenset(evidence.ATOMIC_TASKS)
    assert len(submission.ATOMIC_TASKS) == 17

    gate = {
        "stage": "one_episode_gate",
        "split": "atomic",
        "tasks_csv": "OpenDrawer",
        "episodes_per_task": 1,
        "num_shards": 1,
        "base_seed": 42,
        "prior_gate_evidence": None,
    }
    assert submission._validate_evaluation(gate, "gate") == gate

    gate_evidence, _ = _write_json(
        tmp_path / "evaluation_evidence.json",
        {"schema_version": 1, "execution_complete": True},
    )
    formal = {
        "stage": "formal",
        "split": "atomic",
        "tasks_csv": "",
        "episodes_per_task": 20,
        "num_shards": 8,
        "base_seed": 42,
        "prior_gate_evidence": _descriptor(gate_evidence),
    }
    assert submission._validate_evaluation(formal, "formal") == formal

    invalid_gate = {**gate, "num_shards": 8}
    with pytest.raises(submission.SubmissionVerificationError, match="one_episode_gate"):
        submission._validate_evaluation(invalid_gate, "gate")
    invalid_formal = {**formal, "num_shards": 1}
    with pytest.raises(submission.SubmissionVerificationError, match="formal contract"):
        submission._validate_evaluation(invalid_formal, "formal")

    for invalid_gate in (
        {**gate, "split": "composite"},
        {**gate, "tasks_csv": "NotACanonicalTask"},
        {**gate, "base_seed": 43},
    ):
        with pytest.raises(submission.SubmissionVerificationError):
            submission._validate_evaluation(invalid_gate, "gate")


def test_submission_unknown_terminal_record_is_rejected(
    valid_records: dict[str, object],
) -> None:
    readback = valid_records["readback"]
    assert isinstance(readback, Path)
    unknown = readback.with_name("submission-unknown.json")
    _write_json(unknown, {"schema_version": 1, "status": "unknown"})

    with pytest.raises(submission.SubmissionVerificationError, match="submission-unknown"):
        submission.verify_submission(**_arguments(valid_records))


def test_scheduler_acceptance_cannot_predate_start(
    valid_records: dict[str, object],
) -> None:
    value = copy.deepcopy(valid_records["readback_value"])
    value["scheduler_started_at_utc"] = NOW.isoformat().replace("+00:00", "Z")
    readback = valid_records["readback"]
    assert isinstance(readback, Path)
    readback.chmod(0o600)
    _write_json(readback, value)

    with pytest.raises(submission.SubmissionVerificationError, match="predates"):
        submission.verify_submission(**_arguments(valid_records))
