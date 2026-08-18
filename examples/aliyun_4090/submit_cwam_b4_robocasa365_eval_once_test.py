"""Structural tests for the model-specified exact-once DLC submitter."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SUBMITTER = Path(__file__).with_name("submit_cwam_b4_robocasa365_eval_once.py")

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def _tree() -> ast.Module:
    return ast.parse(SUBMITTER.read_text(encoding="utf-8"))


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


def test_get_job_uses_required_request_argument() -> None:
    calls = [
        node
        for node in ast.walk(_tree())
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "get_job"
    ]
    assert len(calls) == 1
    assert len(calls[0].args) == 2


def test_reconciliation_never_creates_a_job() -> None:
    function = _function(_tree(), "reconcile_existing")
    attributes = {
        node.attr for node in ast.walk(function) if isinstance(node, ast.Attribute)
    }
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
    persisted = source.index("write_receipt(RECEIPT, accepted, exclusive=False)")
    readback = source.index("readback = get_job_readback(client, identifier)", persisted)
    assert accepted < persisted < readback
    assert 'mode.add_argument("--reconcile"' in source


def test_submitter_requires_local_output_persistence_controls() -> None:
    source = SUBMITTER.read_text(encoding="utf-8")

    assert '"cwam_b4_eval_shell_utils.sh"' in source
    assert '"cwam_dlc_persist_eval_output.py"' in source
    assert '"Priority": 7' in source


def test_submitter_binds_generic_model_identity() -> None:
    source = SUBMITTER.read_text(encoding="utf-8")

    for name in (
        "EVAL_CHECKPOINT_NAME",
        "EVAL_CONTROL_ROOT",
        "EVAL_EXPECTED_EMA_KEY_COUNT",
        "EVAL_RUN_NAME",
        "EVAL_SOURCE_COMMIT",
        "EVAL_SOURCE_ROOT",
        "EVAL_SOURCE_RUN_ID",
    ):
        assert name in source
    assert 'MODEL_SPEC = CONTROL_ROOT / "model-spec.json"' in source
    assert 'expected = expected_checkpoint_inventory(load_model_spec())' in source
    assert '"checkpoint inventory differs from model spec"' in source


def test_source_snapshot_supports_metadata_only_export() -> None:
    function = _function(_tree(), "source_revision")
    constants = {
        node.value for node in ast.walk(function) if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }

    assert "RUN_ID" in constants
    assert "GIT_REVISION" in constants
    assert "rev-parse" in constants


def test_reconciliation_binds_model_specific_identity() -> None:
    source = SUBMITTER.read_text(encoding="utf-8")

    for key in ("source_run_id", "checkpoint_name", "model_id", "expected_ema_key_count"):
        assert f'"{key}"' in source
