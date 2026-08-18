"""Structural tests for the exact-once B4 DLC submitter."""

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
