# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""H-workspace tests for the formal-continuation submission state machine."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

import pytest

_SNAPSHOT_ROOT = Path("/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chengjuntao/run_snapshots")
_OUTPUT_ROOT = Path("/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chengjuntao/outputs/cwam-cosmos3-formal")


def _remove_owned_tree(path: Path) -> None:
    assert path.parent == (_OUTPUT_ROOT / "formal_continuation_submission_reservations")
    assert path.name.startswith("CWAM-E35-CONT-SUBMIT-FSM-TEST-")
    if not path.exists():
        return
    path.chmod(0o700)
    for candidate in sorted(path.rglob("*"), reverse=True):
        assert path in candidate.parents
        if candidate.is_dir() and not candidate.is_symlink():
            candidate.chmod(0o700)
            candidate.rmdir()
        else:
            candidate.chmod(0o600)
            candidate.unlink()
    path.rmdir()


@pytest.fixture
def continuation_submit_fixture():
    if not _SNAPSHOT_ROOT.is_dir() or not _OUTPUT_ROOT.parent.is_dir():
        pytest.skip("H-cluster GPFS roots are unavailable")
    source = Path(tempfile.mkdtemp(prefix="cwam-cont-submit-fsm-", dir=_SNAPSHOT_ROOT))
    experiment_id = f"CWAM-E35-CONT-SUBMIT-FSM-TEST-{uuid.uuid4().hex[:12]}"
    reservation_dir = _OUTPUT_ROOT / "formal_continuation_submission_reservations" / experiment_id
    try:
        scripts = source / "examples/h_cluster"
        scripts.mkdir(parents=True)
        (source / "SNAPSHOT_MANIFEST.json").write_text('{"schema_version": 1}\n')
        (scripts / "cwam_formal_provenance.py").write_text("#!/usr/bin/env python3\nraise SystemExit(0)\n")
        (scripts / "cwam_formal_training_evidence.py").write_text("# fixture\n")
        (scripts / "run_cwam_cosmos3_nano_formal_continuation.sh").write_text("#!/usr/bin/env bash\nexit 99\n")
        (scripts / "cwam_cosmos3_formal_continuation.py").write_text(
            """#!/usr/bin/env python3
import argparse
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument("command")
parser.add_argument("--output")
args, unknown = parser.parse_known_args()
if args.command != "validate-inputs":
    raise SystemExit(2)
if "--require-torch-dcp-metadata" not in unknown:
    raise SystemExit(19)
payload = {
    "throughput": {
        "workers_per_rank": 12,
        "storage": "nvme",
        "activation_checkpointing": "none",
        "best_observed_cell_id": "workers-12__storage-nvme__ac-none",
    }
}
descriptor = os.open(
    args.output,
    os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
    0o444,
)
with os.fdopen(descriptor, "w") as handle:
    json.dump(payload, handle)
    handle.write("\\n")
"""
        )
        (scripts / "capture_cwam_formal_rjob_receipt.py").write_text(
            """#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--scheduler-rjob-name", required=True)
parser.add_argument("--expected-json", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--attempts")
parser.add_argument("--retry-seconds")
args = parser.parse_args()
if args.attempts != "90" or args.retry_seconds != "1":
    raise SystemExit(18)
if int(os.environ.get("FAKE_SCHEDULER_CAPTURE_EXIT", "0")):
    raise SystemExit(int(os.environ["FAKE_SCHEDULER_CAPTURE_EXIT"]))
expected = json.loads(args.expected_json.read_text())
script = Path(os.environ["FAKE_COMMAND_FILE"]).read_text()
host = {
    "enabled": expected["store_host_nvme"],
    "annotation": "kubebrain.brainpp.cn/writeread-hostpath-mounts",
    "value": (
        "/nvme:/mnt/nvme1n1,/cfs:/mnt/cfs"
        if expected["store_host_nvme"]
        else None
    ),
    "container_target": "/nvme" if expected["store_host_nvme"] else None,
    "host_source": "/mnt/nvme1n1" if expected["store_host_nvme"] else None,
}
if os.environ.get("FAKE_STRUCTURED_MUTATION") == "host_nvme":
    host["value"] = "/nvme:/wrong"
command = ["bash", "-lc", script]
resources = {
    "requests_equal_core_limits": True,
    "cpu": expected["cpu"],
    "memory_mib": expected["memory_mib"],
    "gpu": expected["gpu"],
    "platform_injected_limit_keys": [],
}
payload = {
    "schema_version": 1,
    "kind": "cwam_formal_rjob_scheduler_receipt",
    "validation": {
        "status": "passed",
        "expected_contract_schema_version": 2,
    },
    "metadata": {
        "name": args.scheduler_rjob_name,
        "showname": expected["showname"],
        "uid": "6d7d6919-2bff-44fa-8b8f-4d5ec49460e2",
    },
    "spec": {
        "task": {
            "mounts": expected["mounts"],
            "submission_nonce": expected["submission_nonce"],
            "submission_reservation_path": expected[
                "submission_reservation_path"
            ],
            "store_host_nvme": host,
            "container": {
                "image": expected["image"],
                "resources": resources,
                "command": command,
                "command_sha256": expected["command_sha256"],
            },
        }
    },
    "replicas": [
        {
            "replica_name": f"{args.scheduler_rjob_name}-abcde",
            "pod_name": "rjob-cont-fixture-0",
            "worker_spec": {
                "store_host_nvme": host,
                "command": [
                    "/kubebrain/worker-init",
                    "run",
                    "--enable-pid-monitor",
                    "--",
                    *command,
                ],
            },
        }
    ],
}
descriptor = os.open(
    args.output,
    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
    0o444,
)
with os.fdopen(descriptor, "w") as handle:
    json.dump(payload, handle, sort_keys=True)
    handle.write("\\n")
"""
        )
        submitter = scripts / "submit_cwam_cosmos3_nano_formal_continuation_rjob.sh"
        shutil.copy2(Path(__file__).with_name(submitter.name), submitter)
        submitter.chmod(0o755)
        formal_manifest = source / "formal-manifest.json"
        formal_manifest.write_text(json.dumps({"experiment_id": experiment_id}) + "\n")

        fake_bin = source / "fake-bin"
        fake_bin.mkdir()
        fake_chmod = fake_bin / "chmod"
        fake_chmod.write_text(
            """#!/usr/bin/env bash
set -Eeuo pipefail
if [[ -n "${FAKE_CHMOD_FAIL_ACCEPTED_ONCE:-}" &&
      ! -e "$FAKE_CHMOD_FAIL_ACCEPTED_ONCE" ]]; then
    for candidate in "$@"; do
        if [[ "$candidate" == */submission.json ]]; then
            : >"$FAKE_CHMOD_FAIL_ACCEPTED_ONCE"
            exit 81
        fi
    done
fi
exec /usr/bin/chmod "$@"
"""
        )
        fake_chmod.chmod(0o755)
        fake_command_file = source / "fake-rjob-command.txt"
        fake_rjob = fake_bin / "rjob"
        fake_rjob.write_text(
            """#!/usr/bin/env bash
set -Eeuo pipefail
action="$1"
shift
if [[ "$action" == "submit" ]]; then
    requested=""
    while (($#)); do
        if [[ "$1" == "--name" ]]; then
            requested="$2"
        fi
        if [[ "$1" == "--" ]]; then
            shift
            [[ "$1" == bash && "$2" == -lc ]]
            printf %s "$3" >"$FAKE_COMMAND_FILE"
            break
        fi
        shift
    done
    [[ -n "$requested" ]]
    if [[ -n "${FAKE_UNKNOWN_PATH:-}" ]]; then
        mkdir "$FAKE_UNKNOWN_PATH"
    fi
    if [[ -n "${FAKE_RJOB_STARTED_FILE:-}" ]]; then
        : >"$FAKE_RJOB_STARTED_FILE"
    fi
    if [[ -n "${FAKE_RJOB_SUBMIT_SLEEP:-}" ]]; then
        sleep "$FAKE_RJOB_SUBMIT_SLEEP"
    fi
    echo "01-01 00:00:00 [INFO] created rjob_name: ${requested}-12345678"
    exit "${FAKE_RJOB_SUBMIT_EXIT:-0}"
fi
if [[ "$action" == "get" ]]; then
    actual="$1"
    requested="${actual%-12345678}"
    echo "01-01 00:00:01 [INFO] rjob ${actual} (showname=${requested}): Inqueue"
    exit "${FAKE_RJOB_GET_EXIT:-0}"
fi
exit 2
"""
        )
        fake_rjob.chmod(0o755)
        digest = hashlib.sha256(formal_manifest.read_bytes()).hexdigest()
        environment = os.environ.copy()
        environment.update(
            {
                "PATH": f"{fake_bin}:{environment['PATH']}",
                "COSMOS_SOURCE": str(source),
                "FORMAL_EXPERIMENT_ID": experiment_id,
                "FORMAL_EXPERIMENT_MODE": "exploratory",
                "FORMAL_RUN_NAME": experiment_id,
                "FORMAL_MANIFEST_PATH": str(formal_manifest),
                "FORMAL_MANIFEST_SHA256": digest,
                "EXPECTED_SOURCE_COMMIT": "a" * 40,
                "EXPECTED_WORKFLOW_SNAPSHOT_SHA256": "b" * 64,
                "EXPECTED_SNAPSHOT_MANIFEST_SHA256": "c" * 64,
                "GPFS_LATENT_CACHE_ROOT": "/fixture/cache",
                "EXPECTED_CACHE_SUCCESS_SHA256": "d" * 64,
                "EXPECTED_CACHE_MANIFEST_SHA256": "e" * 64,
                "THROUGHPUT_MATRIX_SUMMARY": "/fixture/matrix.json",
                "EXPECTED_THROUGHPUT_MATRIX_SHA256": "f" * 64,
                "EXPECTED_BENCHMARK_SOURCE_ROOT": "/fixture/benchmark-snapshot",
                "EXPECTED_BENCHMARK_SOURCE_COMMIT": "9" * 40,
                "EXPECTED_BENCHMARK_SNAPSHOT_MANIFEST_SHA256": "8" * 64,
                "R6_ITER1000_CHECKPOINT_PATH": "/fixture/iter_000001000",
                "EXPECTED_R6_CHECKPOINT_FINGERPRINT_SHA256": "1" * 64,
                "RJOB_NAME": f"cwam-cont-fsm-{uuid.uuid4().hex[:12]}",
                "RJOB_PREDICT_ONLY": "0",
                "FAKE_COMMAND_FILE": str(fake_command_file),
            }
        )
        yield submitter, environment, reservation_dir, fake_command_file
    finally:
        _remove_owned_tree(reservation_dir)
        for candidate in sorted(source.rglob("*"), reverse=True):
            if candidate.is_dir() and not candidate.is_symlink():
                candidate.rmdir()
            else:
                candidate.unlink()
        source.rmdir()


def _run_fixture(fixture, **environment_overrides):
    submitter, environment, reservation_dir, command_file = fixture
    result = subprocess.run(
        ["bash", str(submitter)],
        env={**environment, **environment_overrides},
        text=True,
        capture_output=True,
        timeout=45,
        check=False,
    )
    return result, reservation_dir, command_file


def test_fake_scheduler_accepts_only_structured_v2_binding(
    continuation_submit_fixture,
) -> None:
    result, reservation_dir, command_file = _run_fixture(continuation_submit_fixture)
    assert result.returncode == 0, result.stdout + result.stderr
    reservation = json.loads((reservation_dir / "submission.json").read_text())
    assert reservation["schema_version"] == 2
    assert reservation["status"] == "accepted"
    assert reservation["store_host_nvme"] is True
    assert len(reservation["submission_nonce"]) == 64
    assert reservation["scheduler_rjob_uid"] == ("6d7d6919-2bff-44fa-8b8f-4d5ec49460e2")
    assert set(reservation["artifacts"]) == {
        "submission_log",
        "scheduler_expected_contract",
        "scheduler_capture_log",
        "scheduler_structured_receipt",
        "scheduler_text_readback",
    }
    assert reservation_dir.stat().st_mode & 0o777 == 0o555
    assert (reservation_dir / "submission.json").stat().st_mode & 0o777 == 0o444
    command = command_file.read_text()
    assert "export FORMAL_SUBMISSION_NONCE=" in command
    assert "export FORMAL_SUBMISSION_RESERVATION_PATH=" in command
    assert "exec bash " in command


def test_fake_scheduler_ambiguous_submit_seals_unknown(
    continuation_submit_fixture,
) -> None:
    result, reservation_dir, _ = _run_fixture(
        continuation_submit_fixture,
        FAKE_RJOB_SUBMIT_EXIT="42",
    )
    assert result.returncode != 0
    unknown = json.loads((reservation_dir / "submission-unknown.json").read_text())
    assert unknown["status"] == "submission_unknown"
    assert unknown["submission_unknown_reason"] == ("rjob_submit_failed_or_ambiguous")
    assert not (reservation_dir / "submission.json").exists()


def test_post_accept_sealing_failure_removes_accepted_before_unknown(
    continuation_submit_fixture,
) -> None:
    _, _, _, command_file = continuation_submit_fixture
    failure_marker = command_file.with_name("fake-chmod-failed")
    result, reservation_dir, _ = _run_fixture(
        continuation_submit_fixture,
        FAKE_CHMOD_FAIL_ACCEPTED_ONCE=str(failure_marker),
    )
    assert result.returncode != 0
    assert failure_marker.is_file()
    unknown = json.loads((reservation_dir / "submission-unknown.json").read_text())
    assert unknown["status"] == "submission_unknown"
    assert unknown["submission_unknown_reason"] == "unexpected_post_accept_failure"
    assert not (reservation_dir / "submission.json").exists()
    assert reservation_dir.stat().st_mode & 0o777 == 0o555


def test_fake_scheduler_signal_during_submit_seals_unknown(
    continuation_submit_fixture,
) -> None:
    submitter, environment, reservation_dir, _ = continuation_submit_fixture
    started = Path(environment["COSMOS_SOURCE"]) / "submit-started"
    process = subprocess.Popen(
        ["bash", str(submitter)],
        env={
            **environment,
            "FAKE_RJOB_STARTED_FILE": str(started),
            "FAKE_RJOB_SUBMIT_SLEEP": "30",
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline and not started.exists():
        if process.poll() is not None:
            break
        time.sleep(0.05)
    assert started.exists(), process.communicate(timeout=5)
    os.killpg(process.pid, signal.SIGTERM)
    stdout, stderr = process.communicate(timeout=15)

    assert process.returncode != 0, stdout + stderr
    unknown = json.loads((reservation_dir / "submission-unknown.json").read_text())
    assert unknown["submission_unknown_reason"] == ("submission_attempt_interrupted_term")
    assert reservation_dir.stat().st_mode & 0o777 == 0o555
    assert not (reservation_dir / "submission.json").exists()


def test_unknown_marker_write_failure_is_visible_and_not_sealed(
    continuation_submit_fixture,
) -> None:
    _, _, reservation_dir, _ = continuation_submit_fixture
    result, _, _ = _run_fixture(
        continuation_submit_fixture,
        FAKE_RJOB_SUBMIT_EXIT="42",
        FAKE_UNKNOWN_PATH=str(reservation_dir / "submission-unknown.json"),
    )

    assert result.returncode == 125
    assert "CRITICAL: failed to persist submission-unknown marker" in (result.stdout + result.stderr)
    assert (reservation_dir / "submission-unknown.json").is_dir()
    assert reservation_dir.stat().st_mode & 0o777 == 0o700
    assert not (reservation_dir / "submission.json").exists()


def test_fake_scheduler_capture_failure_seals_unknown(
    continuation_submit_fixture,
) -> None:
    result, reservation_dir, _ = _run_fixture(
        continuation_submit_fixture,
        FAKE_SCHEDULER_CAPTURE_EXIT="17",
    )
    assert result.returncode != 0
    unknown = json.loads((reservation_dir / "submission-unknown.json").read_text())
    assert unknown["submission_unknown_reason"] in {
        "accepted_cli_without_validated_scheduler_crd_and_replica",
        "unexpected_post_accept_failure",
        "unexpected_post_accept_exit",
    }
    assert not (reservation_dir / "submission.json").exists()


def test_fake_scheduler_post_accept_drift_seals_unknown(
    continuation_submit_fixture,
) -> None:
    result, reservation_dir, _ = _run_fixture(
        continuation_submit_fixture,
        FAKE_STRUCTURED_MUTATION="host_nvme",
    )
    assert result.returncode != 0
    unknown = json.loads((reservation_dir / "submission-unknown.json").read_text())
    assert unknown["submission_unknown_reason"] in {
        "unexpected_post_accept_failure",
        "unexpected_post_accept_exit",
    }
    assert not (reservation_dir / "submission.json").exists()
