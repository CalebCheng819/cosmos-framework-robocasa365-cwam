# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Behavioral tests for local formal-evidence persistence."""

from __future__ import annotations

import json
import stat
import subprocess
import tarfile
from pathlib import Path

import pytest

pytestmark = [pytest.mark.L0, pytest.mark.CPU]

HERE = Path(__file__).resolve().parent
HELPER = HERE / "cwam_dlc_persist_eval_output.py"
COMMIT = "d" * 40
EXPERIMENT = "CWAM-EVAL-R11"
RUN = "CWAM-EVAL-R11-RUN"


def _invoke(
    tmp_path: Path,
    *,
    launcher_exit_code: int,
    create_local_root: bool = True,
) -> tuple[subprocess.CompletedProcess[str], Path, Path]:
    local_parent = tmp_path / "local"
    persistent_parent = tmp_path / "persistent"
    local_parent.mkdir()
    persistent_parent.mkdir()
    local_root = local_parent / EXPERIMENT / RUN
    if create_local_root:
        control = local_root / "control"
        results = local_root / "results"
        control.mkdir(parents=True)
        results.mkdir()
        receipt = control / "receipt.json"
        receipt.write_text('{"status":"complete"}\n', encoding="utf-8")
        receipt.chmod(0o444)
        (results / "summary.csv").write_text("task,success\nexample,1\n", encoding="utf-8")
    persistent_root = persistent_parent / EXPERIMENT / RUN
    completed = subprocess.run(
        [
            "python3",
            str(HELPER),
            "--local-output-parent",
            str(local_parent),
            "--local-output-root",
            str(local_root),
            "--persistent-output-parent",
            str(persistent_parent),
            "--persistent-output-root",
            str(persistent_root),
            "--experiment-id",
            EXPERIMENT,
            "--run-id",
            RUN,
            "--control-commit",
            COMMIT,
            "--launcher-exit-code",
            str(launcher_exit_code),
            "--job-id",
            "dlc-test",
        ],
        capture_output=True,
        text=True,
    )
    return completed, local_root, persistent_root


def test_success_publishes_mirror_and_mode_preserving_archive(tmp_path: Path) -> None:
    completed, _, persistent_root = _invoke(tmp_path, launcher_exit_code=0)

    assert completed.returncode == 0, completed.stderr
    assert (persistent_root / "control" / "receipt.json").is_file()
    status_document = json.loads(
        (persistent_root / "persistence-status.json").read_text(encoding="utf-8")
    )
    assert status_document["terminal_status"] == "complete"
    assert status_document["publication_complete"] is True
    assert status_document["local_regular_file_count"] == 2
    assert status_document["artifact_hashes"] is None
    assert status_document["expanded_mirror_preserves_unix_modes"] is False
    assert status_document["formal_evidence_archive_preserves_unix_modes"] is True
    with tarfile.open(persistent_root / "formal-evidence-tree.tar", mode="r") as archive:
        receipt = archive.getmember("control/receipt.json")
    assert stat.S_IMODE(receipt.mode) == 0o444


def test_failed_formal_run_publishes_partial_tree_without_masking_status(tmp_path: Path) -> None:
    completed, _, persistent_root = _invoke(tmp_path, launcher_exit_code=17)

    assert completed.returncode == 0, completed.stderr
    status_document = json.loads(
        (persistent_root / "persistence-status.json").read_text(encoding="utf-8")
    )
    assert status_document["launcher_exit_code"] == 17
    assert status_document["terminal_status"] == "failed_partial"


def test_failure_before_output_creation_publishes_status_only(tmp_path: Path) -> None:
    completed, _, persistent_root = _invoke(
        tmp_path, launcher_exit_code=23, create_local_root=False
    )

    assert completed.returncode == 0, completed.stderr
    status_document = json.loads(
        (persistent_root / "persistence-status.json").read_text(encoding="utf-8")
    )
    assert status_document["terminal_status"] == "failed_before_formal_output"
    assert status_document["formal_evidence_archive"] is None
    assert not (persistent_root / "formal-evidence-tree.tar").exists()


def test_existing_persistent_root_is_never_replaced(tmp_path: Path) -> None:
    completed, _, persistent_root = _invoke(tmp_path, launcher_exit_code=0)
    assert completed.returncode == 0, completed.stderr

    second = subprocess.run(
        completed.args,
        capture_output=True,
        text=True,
    )

    assert second.returncode == 1
    assert "refusing to replace persistent run root" in second.stderr
    assert persistent_root.is_dir()
