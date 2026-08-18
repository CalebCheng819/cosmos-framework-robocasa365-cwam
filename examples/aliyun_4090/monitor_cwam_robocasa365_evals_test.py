"""Unit tests for the DLC evaluation lifecycle monitor."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).with_name("monitor_cwam_robocasa365_evals.py")
SPEC = importlib.util.spec_from_file_location("monitor_cwam_robocasa365_evals", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
pytestmark = [pytest.mark.L0, pytest.mark.CPU]


CONFIG = {
    "kind": "cwam_robocasa365_dlc_eval_submission",
    "experiment_id": "EXP-B2",
    "logical_run_id": "RUN-B2",
    "display_name": "exp-b2",
}
JOB_ID = "dlcexample"
COMMIT = "1" * 40


def _job(status: str = "Running") -> dict:
    return {
        "JobId": JOB_ID,
        "DisplayName": CONFIG["display_name"],
        "WorkspaceId": "270969",
        "ResourceId": "quotaksvqq2oh2pg",
        "Priority": 7,
        "Status": status,
        "Envs": {
            "EVAL_CONTROL_COMMIT": COMMIT,
            "EVAL_EXPERIMENT_ID": CONFIG["experiment_id"],
            "EVAL_LOGICAL_RUN_ID": CONFIG["logical_run_id"],
        },
        "JobSpecs": [{
            "Type": "Worker",
            "PodCount": 1,
            "RestartPolicy": "Never",
            "ResourceConfig": {
                "CPU": "126",
                "GPU": "8",
                "GPUType": "",
                "Memory": "960Gi",
                "SharedMemory": "960Gi",
            },
        }],
    }


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_csv(path: Path, rows: int) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["row"])
        writer.writeheader()
        for index in range(rows):
            writer.writerow({"row": index})


def test_validate_job_requires_exact_eight_gpu_contract() -> None:
    snapshot = MODULE.validate_job_readback(_job(), CONFIG, JOB_ID, COMMIT)
    assert snapshot["status"] == "Running"
    assert snapshot["resource_contract"]["gpus_per_worker"] == 8

    bad = _job()
    bad["JobSpecs"][0]["ResourceConfig"]["GPU"] = "4"
    with pytest.raises(RuntimeError, match="resource contract"):
        MODULE.validate_job_readback(bad, CONFIG, JOB_ID, COMMIT)


def test_submitted_jobs_binds_controller_identities() -> None:
    controller = {
        "status": "submitted",
        "evaluations": {
            CONFIG["experiment_id"]: {
                "job_id": JOB_ID,
                "logical_run_id": CONFIG["logical_run_id"],
            }
        },
    }
    assert MODULE.submitted_jobs(controller, {CONFIG["experiment_id"]: CONFIG}) == {
        CONFIG["experiment_id"]: JOB_ID
    }


def test_validate_terminal_output_requires_340_by_17_formal_result(tmp_path: Path) -> None:
    root = tmp_path / CONFIG["experiment_id"] / CONFIG["logical_run_id"]
    root.mkdir(parents=True)
    _write_json(root / "persistence-status.json", {
        "experiment_id": CONFIG["experiment_id"],
        "run_id": CONFIG["logical_run_id"],
        "control_commit": COMMIT,
        "job_id": JOB_ID,
        "launcher_exit_code": 0,
        "terminal_status": "complete",
        "publication_complete": True,
        "local_regular_file_count": 10,
        "local_regular_file_bytes": 100,
    })
    _write_json(root / "launcher-terminal.json", {
        "status": "COMPLETE",
        "exit_code": 0,
        "experiment_id": CONFIG["experiment_id"],
        "logical_run_id": CONFIG["logical_run_id"],
        "control_commit": COMMIT,
        "dlc_job_id": JOB_ID,
    })
    _write_json(root / "evaluation_evidence.json", {
        "status": "COMPLETE",
        "experiment_id": CONFIG["experiment_id"],
        "logical_run_id": CONFIG["logical_run_id"],
        "protocol": {"task_count": 17, "episodes_per_task": 20, "episode_count": 340},
        "result": {"episodes": 340, "successes": 170, "mean_success_rate": 0.5, "per_task": [{}] * 17},
        "validation": {"exactly_17_tasks_and_340_episodes": True},
        "execution": {"physical_gpus": 8},
    })
    _write_json(root / "status.json", {
        "status": "COMPLETE",
        "experiment_id": CONFIG["experiment_id"],
        "logical_run_id": CONFIG["logical_run_id"],
        "episodes": 340,
        "successes": 170,
    })
    _write_csv(root / "summary.csv", 340)
    _write_csv(root / "task_success_rates.csv", 17)

    result = MODULE.validate_terminal_output(tmp_path, CONFIG, JOB_ID, COMMIT)
    assert result["status"] == "complete"
    assert result["successes"] == 170

    _write_csv(root / "task_success_rates.csv", 16)
    with pytest.raises(RuntimeError, match="expected 17 rows"):
        MODULE.validate_terminal_output(tmp_path, CONFIG, JOB_ID, COMMIT)
