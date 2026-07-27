# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""H-workspace integration tests for the formal submit reservation state machine."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

import pytest

_SNAPSHOT_ROOT = Path("/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chengjuntao/run_snapshots")
_OUTPUT_ROOT = Path("/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chengjuntao/outputs/cwam-cosmos3-formal")


def _remove_owned_tree(path: Path, *, expected_parent: Path) -> None:
    """Remove only the UUID-owned fixture tree created by this test."""
    assert path.parent == expected_parent
    assert path.name.startswith("CWAM-E35-SUBMIT-FSM-TEST-")
    if not path.exists():
        return
    for candidate in sorted(path.rglob("*"), reverse=True):
        assert path in candidate.parents
        if candidate.is_dir() and not candidate.is_symlink():
            candidate.rmdir()
        else:
            candidate.chmod(0o600)
            candidate.unlink()
    path.rmdir()


@pytest.fixture
def submit_fixture():
    if not _SNAPSHOT_ROOT.is_dir() or not _OUTPUT_ROOT.parent.is_dir():
        pytest.skip("H-cluster GPFS roots are unavailable")
    source = Path(tempfile.mkdtemp(prefix="cwam-submit-fsm-", dir=_SNAPSHOT_ROOT))
    experiment_id = f"CWAM-E35-SUBMIT-FSM-TEST-{uuid.uuid4().hex[:12]}"
    reservation_dir = _OUTPUT_ROOT / "formal_submission_reservations" / experiment_id
    try:
        (source / "examples/h_cluster").mkdir(parents=True)
        (source / "SNAPSHOT_MANIFEST.json").write_text('{"schema_version": 1}\n')
        (source / "examples/h_cluster/cwam_formal_provenance.py").write_text(
            "#!/usr/bin/env python3\nraise SystemExit(0)\n"
        )
        shutil.copy2(
            Path(__file__).with_name("cwam_secure_paths.py"),
            source / "examples/h_cluster/cwam_secure_paths.py",
        )
        shutil.copy2(
            Path(__file__).with_name("cwam_secure_tee.py"),
            source / "examples/h_cluster/cwam_secure_tee.py",
        )
        (source / "examples/h_cluster/capture_cwam_formal_rjob_receipt.py").write_text(
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
if int(os.environ.get("FAKE_SCHEDULER_CAPTURE_EXIT", "0")):
    raise SystemExit(int(os.environ["FAKE_SCHEDULER_CAPTURE_EXIT"]))
expected = json.loads(args.expected_json.read_text())
payload = {
    "schema_version": 1,
    "kind": "cwam_formal_rjob_scheduler_receipt",
    "validation": {"status": "passed", "expected_contract_schema_version": 1},
    "metadata": {"name": args.scheduler_rjob_name, "showname": expected["showname"]},
    "pod_names": ["rjob-fixture-0"],
    "replica_names": [f"{args.scheduler_rjob_name}-abcde"],
}
descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
with os.fdopen(descriptor, "w") as handle:
    handle.write(json.dumps(payload, sort_keys=True) + "\\n")
"""
        )
        (source / "examples/h_cluster/run_cwam_cosmos3_nano_production_then_formal.sh").write_text(
            "#!/usr/bin/env bash\nexit 99\n"
        )
        script = source / "examples/h_cluster/submit_cwam_cosmos3_nano_formal_rjob.sh"
        shutil.copy2(Path(__file__).with_name(script.name), script)
        script.chmod(0o755)
        formal_manifest = source / "formal-manifest.json"
        formal_manifest.write_text(json.dumps({"experiment_id": experiment_id}) + "\n")

        fake_bin = source / "fake-bin"
        fake_bin.mkdir()
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
            break
        fi
        shift
    done
    [[ -n "$requested" ]]
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
        manifest_sha256 = hashlib.sha256(formal_manifest.read_bytes()).hexdigest()
        environment = os.environ.copy()
        environment.update(
            {
                "PATH": f"{fake_bin}:{environment['PATH']}",
                "COSMOS_SOURCE": str(source),
                "FORMAL_EXPERIMENT_ID": experiment_id,
                "FORMAL_RUN_NAME": experiment_id,
                "FORMAL_MANIFEST_PATH": str(formal_manifest),
                "FORMAL_MANIFEST_SHA256": manifest_sha256,
                "EXPECTED_SOURCE_COMMIT": "a" * 40,
                "EXPECTED_WORKFLOW_SNAPSHOT_SHA256": "b" * 64,
                "EXPECTED_SNAPSHOT_MANIFEST_SHA256": "c" * 64,
                "NPROC_PER_NODE": "8",
                "RJOB_NAME": f"cwam-submit-fsm-{uuid.uuid4().hex[:12]}",
                "RJOB_PREDICT_ONLY": "0",
            }
        )
        yield script, environment, reservation_dir
    finally:
        _remove_owned_tree(
            reservation_dir,
            expected_parent=_OUTPUT_ROOT / "formal_submission_reservations",
        )
        for candidate in sorted(source.rglob("*"), reverse=True):
            if candidate.is_dir() and not candidate.is_symlink():
                candidate.rmdir()
            else:
                candidate.unlink()
        source.rmdir()


def test_submit_accepts_only_after_scheduler_readback(submit_fixture) -> None:
    script, environment, reservation_dir = submit_fixture
    result = subprocess.run(
        ["bash", str(script)],
        env=environment,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    reservation_path = reservation_dir / "submission.json"
    payload = json.loads(reservation_path.read_text())
    assert payload["status"] == "accepted"
    assert payload["scheduler_rjob_name"].endswith("-12345678")
    assert reservation_path.stat().st_mode & 0o777 == 0o444
    assert (reservation_dir / "submission-pending.json").stat().st_mode & 0o777 == 0o444
    assert not (reservation_dir / "submission-unknown.json").exists()
    assert (reservation_dir / "rjob-submit.log").stat().st_mode & 0o777 == 0o444
    assert (reservation_dir / "rjob-expected-contract.json").stat().st_mode & 0o777 == 0o444
    assert (reservation_dir / "rjob-scheduler-capture.log").stat().st_mode & 0o777 == 0o444
    assert (reservation_dir / "rjob-scheduler-receipt.json").stat().st_mode & 0o777 == 0o444
    assert (reservation_dir / "rjob-get.log").stat().st_mode & 0o777 == 0o444


def test_ambiguous_submit_retains_read_only_unique_lock(submit_fixture) -> None:
    script, environment, reservation_dir = submit_fixture
    environment["FAKE_RJOB_SUBMIT_EXIT"] = "42"
    result = subprocess.run(
        ["bash", str(script)],
        env=environment,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode != 0
    reservation_path = reservation_dir / "submission-unknown.json"
    payload = json.loads(reservation_path.read_text())
    assert payload["status"] == "submission_unknown"
    assert payload["rjob_exit_code"] == 42
    assert reservation_path.stat().st_mode & 0o777 == 0o444
    assert not (reservation_dir / "submission.json").exists()

    duplicate = subprocess.run(
        ["bash", str(script)],
        env={**environment, "FAKE_RJOB_SUBMIT_EXIT": "0"},
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert duplicate.returncode != 0
    assert "already has a submit-side reservation" in duplicate.stderr


def test_missing_scheduler_readback_retains_unknown_lock(submit_fixture) -> None:
    script, environment, reservation_dir = submit_fixture
    environment["FAKE_RJOB_GET_EXIT"] = "3"
    result = subprocess.run(
        ["bash", str(script)],
        env=environment,
        text=True,
        capture_output=True,
        timeout=90,
        check=False,
    )
    assert result.returncode != 0
    payload = json.loads((reservation_dir / "submission-unknown.json").read_text())
    assert payload["status"] == "submission_unknown"
    assert payload["submission_unknown_reason"] == ("accepted_cli_result_without_scheduler_readback")


def test_invalid_scheduler_spec_retains_unknown_lock(submit_fixture) -> None:
    script, environment, reservation_dir = submit_fixture
    environment["FAKE_SCHEDULER_CAPTURE_EXIT"] = "17"
    result = subprocess.run(
        ["bash", str(script)],
        env=environment,
        text=True,
        capture_output=True,
        timeout=45,
        check=False,
    )
    assert result.returncode != 0
    payload = json.loads((reservation_dir / "submission-unknown.json").read_text())
    assert payload["status"] == "submission_unknown"
    assert payload["submission_unknown_reason"] == ("accepted_cli_result_without_validated_scheduler_spec")
