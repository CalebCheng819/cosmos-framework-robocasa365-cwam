"""Unit tests for the transfer-gated DLC evaluation controller."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).with_name("wait_and_submit_cwam_robocasa365_evals.py")
SPEC = importlib.util.spec_from_file_location("wait_and_submit_cwam_robocasa365_evals", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def _write_state(path: Path, destination: str, done: int, total: int, status: str) -> None:
    path.write_text(
        json.dumps(
            {
                "destination": destination,
                "completed_bytes": done,
                "expected_bytes": total,
                "last_error": None,
                "status": status,
            }
        ),
        encoding="utf-8",
    )


def test_transfer_snapshot_requires_all_exactly_complete(tmp_path: Path) -> None:
    _write_state(tmp_path / "a.json", "/dst/a", 10, 10, "complete")
    _write_state(tmp_path / "b.json", "/dst/b", 4, 10, "copying")

    partial = MODULE.transfer_snapshot(tmp_path, 2)
    assert partial["complete"] is False
    assert partial["completed_bytes"] == 14

    _write_state(tmp_path / "b.json", "/dst/b", 10, 10, "complete")
    complete = MODULE.transfer_snapshot(tmp_path, 2)
    assert complete["complete"] is True
    assert complete["statuses"] == {"complete": 2}


def test_transfer_snapshot_rejects_duplicate_destination(tmp_path: Path) -> None:
    _write_state(tmp_path / "a.json", "/dst/same", 10, 10, "complete")
    _write_state(tmp_path / "b.json", "/dst/same", 10, 10, "complete")

    with pytest.raises(RuntimeError, match="duplicate transfer destination"):
        MODULE.transfer_snapshot(tmp_path, 2)


def test_receipt_mode_is_exact_once_and_reconcilable(tmp_path: Path) -> None:
    receipt = tmp_path / "receipt.json"
    assert MODULE.receipt_mode(receipt, "exp", "run") == "--submit"
    receipt.write_text(
        json.dumps({"experiment_id": "exp", "logical_run_id": "run", "status": "CREATE_REQUEST_FAILED"}),
        encoding="utf-8",
    )
    assert MODULE.receipt_mode(receipt, "exp", "run") == "--reconcile"
    receipt.write_text(
        json.dumps(
            {
                "experiment_id": "exp",
                "logical_run_id": "run",
                "status": "SUBMITTED",
                "job_id": "dlc-example",
            }
        ),
        encoding="utf-8",
    )
    assert MODULE.receipt_mode(receipt, "exp", "run") == "already-submitted"
