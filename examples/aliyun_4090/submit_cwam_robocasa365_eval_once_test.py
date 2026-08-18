"""Structural tests for the configuration-bound RoboCasa365 DLC submitter."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


SUBMITTER = Path(__file__).with_name("submit_cwam_robocasa365_eval_once.py")
pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def _tree() -> ast.Module:
    return ast.parse(SUBMITTER.read_text(encoding="utf-8"))


def _function(name: str) -> ast.FunctionDef:
    return next(
        node for node in _tree().body if isinstance(node, ast.FunctionDef) and node.name == name
    )


def test_submitter_requests_one_eight_gpu_worker() -> None:
    source = SUBMITTER.read_text(encoding="utf-8")

    assert '"PodCount": 1' in source
    assert '"GPU": "8"' in source
    assert '"CPU": "126"' in source
    assert '"Priority": 7' in source
    assert '"SuccessPolicy": "AllWorkers"' in source


def test_submitter_binds_all_launcher_inputs_and_notion_page() -> None:
    source = SUBMITTER.read_text(encoding="utf-8")

    for name in (
        "EVAL_CONTROL_COMMIT",
        "EVAL_EXPERIMENT_ID",
        "EVAL_LOGICAL_RUN_ID",
        "EVAL_CONTROL_ROOT",
        "EVAL_SOURCE_ROOT",
        "EVAL_SOURCE_COMMIT",
        "EVAL_SOURCE_RUN_ID",
        "EVAL_CHECKPOINT_NAME",
        "EVAL_RUN_NAME",
        "EVAL_EXPECTED_EMA_KEY_COUNT",
        "EVAL_NOTION_PAGE_ID",
    ):
        assert f'"{name}"' in source


def test_submitter_accepts_readonly_snapshot_markers() -> None:
    source = SUBMITTER.read_text(encoding="utf-8")

    assert 'marker = root / "GIT_REVISION"' in source
    assert 'run_id = source_root / "RUN_ID"' in source
    assert 'not marker.is_symlink()' in source
    assert 'not run_id.is_symlink()' in source


def test_submitter_requires_exact_checkpoint_and_model_spec_inventory() -> None:
    source = SUBMITTER.read_text(encoding="utf-8")

    assert 'observed == config["checkpoint"]["inventory"]' in source
    assert 'expected_from_model_spec == config["checkpoint"]["inventory"]' in source
    assert 'not any(path.is_symlink() for path in checkpoint_root.rglob("*"))' in source


def test_reconciliation_never_creates_job() -> None:
    function = _function("reconcile_existing")
    attributes = {node.attr for node in ast.walk(function) if isinstance(node, ast.Attribute)}

    assert "create_job" not in attributes
    assert "get_job" not in attributes
    assert "stable_matching_jobs" in {
        node.func.id
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }


def test_submitter_persists_job_id_before_readback() -> None:
    source = SUBMITTER.read_text(encoding="utf-8")
    accepted = source.index('"status": "CREATE_ACCEPTED_READBACK_PENDING"')
    persisted = source.index('write_receipt(paths["receipt"], accepted, exclusive=False)')
    readback = source.index("get_job_readback(client, identifier)", persisted)

    assert accepted < persisted < readback
    assert 'mode.add_argument("--reconcile"' in source
