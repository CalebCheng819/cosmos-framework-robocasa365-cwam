# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import json
from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.callbacks.action_gradient_evidence import ActionGradientEvidence


class _ActionHeads(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.action2llm = torch.nn.Linear(2, 3)
        self.llm2action = torch.nn.Linear(3, 2)
        self.action_modality_embed = torch.nn.Parameter(torch.ones(3))


def _callback(tmp_path) -> ActionGradientEvidence:
    callback = ActionGradientEvidence(every_n=1)
    callback.config = SimpleNamespace(job=SimpleNamespace(path_local=str(tmp_path)))
    return callback


def test_action_gradient_evidence_writes_passing_report(tmp_path) -> None:
    model = _ActionHeads()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)

    callback = _callback(tmp_path)
    callback.on_after_backward(model, iteration=0)
    callback.on_before_optimizer_step(
        model,
        optimizer,
        scheduler=SimpleNamespace(),
        grad_scaler=SimpleNamespace(),
        iteration=0,
    )
    callback.on_training_step_batch_end(
        model,
        data_batch={},
        output_batch={"flow_matching_loss_action": torch.tensor(0.25)},
        loss=torch.tensor(0.5),
        iteration=0,
    )

    report = (tmp_path / "action_gradient_evidence/iteration_000000001.json").read_text()
    assert '"status": "passed"' in report
    assert '"gradient_l2_norm"' in report
    assert '"final_microbatch_flow_matching_loss_action_rank_mean": 0.25' in report


def test_action_gradient_evidence_fails_on_missing_gradient(tmp_path) -> None:
    model = _ActionHeads()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    for name, parameter in model.named_parameters():
        if "llm2action" not in name:
            parameter.grad = torch.ones_like(parameter)

    with pytest.raises(RuntimeError, match="llm2action.*have a gradient"):
        _callback(tmp_path).on_after_backward(model, iteration=0)


def test_action_gradient_evidence_excludes_declared_inactive_parameters(tmp_path) -> None:
    model = _ActionHeads()
    model.llm2action.bias.requires_grad_(False)
    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=0.1,
    )
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.grad = torch.ones_like(parameter)

    callback = ActionGradientEvidence(
        every_n=1,
        excluded_parameter_patterns=[r"llm2action\.bias$"],
    )
    callback.config = SimpleNamespace(job=SimpleNamespace(path_local=str(tmp_path)))
    callback.on_after_backward(model, iteration=0)
    callback.on_before_optimizer_step(
        model,
        optimizer,
        scheduler=SimpleNamespace(),
        grad_scaler=SimpleNamespace(),
        iteration=0,
    )
    callback.on_training_step_batch_end(
        model,
        data_batch={},
        output_batch={"flow_matching_loss_action": torch.tensor(0.25)},
        loss=torch.tensor(0.5),
        iteration=0,
    )

    report = json.loads((tmp_path / "action_gradient_evidence/iteration_000000001.json").read_text())
    assert report["families"]["llm2action"]["excluded_parameter_elements"] == 2
    assert report["families"]["llm2action"]["parameter_elements"] == 6


def test_action_gradient_evidence_accepts_optimizer_container(tmp_path) -> None:
    model = _ActionHeads()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    container = SimpleNamespace(optimizers=[optimizer])
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)

    callback = _callback(tmp_path)
    callback.on_after_backward(model, iteration=0)
    callback.on_before_optimizer_step(
        model,
        container,
        scheduler=SimpleNamespace(),
        grad_scaler=SimpleNamespace(),
        iteration=0,
    )
    callback.on_training_step_batch_end(
        model,
        data_batch={},
        output_batch={"flow_matching_loss_action": torch.tensor(0.25)},
        loss=torch.tensor(0.5),
        iteration=0,
    )

    assert '"status": "passed"' in (tmp_path / "action_gradient_evidence/iteration_000000001.json").read_text()


def test_action_gradient_evidence_waits_for_grad_accum_optimizer_boundary(tmp_path) -> None:
    model = _ActionHeads()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    callback = _callback(tmp_path)

    # First microbatch: backward ran, but no optimizer step/report is due.
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    callback.on_after_backward(model, iteration=0)
    callback.on_training_step_batch_end(
        model,
        data_batch={},
        output_batch={"flow_matching_loss_action": torch.tensor(0.2)},
        loss=torch.tensor(0.4),
        iteration=0,
    )
    assert not (tmp_path / "action_gradient_evidence").exists()

    # Final microbatch: the optimizer-boundary hook makes the report eligible.
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, 2.0)
    callback.on_after_backward(model, iteration=0)
    callback.on_before_optimizer_step(
        model,
        optimizer,
        scheduler=SimpleNamespace(),
        grad_scaler=SimpleNamespace(),
        iteration=0,
    )
    callback.on_training_step_batch_end(
        model,
        data_batch={},
        output_batch={"flow_matching_loss_action": torch.tensor(0.3)},
        loss=torch.tensor(0.6),
        iteration=0,
    )
    report = json.loads((tmp_path / "action_gradient_evidence/iteration_000000001.json").read_text())
    assert report["losses"]["scope"] == "final_microbatch_at_optimizer_boundary"
    assert report["losses"]["final_microbatch_flow_matching_loss_action_rank_mean"] == pytest.approx(0.3)


def test_action_gradient_evidence_always_records_nonperiodic_final_iteration(tmp_path) -> None:
    callback = ActionGradientEvidence(every_n=50)
    callback.config = SimpleNamespace(
        job=SimpleNamespace(path_local=str(tmp_path)),
        trainer=SimpleNamespace(max_iter=13340),
    )
    assert callback._should_record(0)
    assert callback._should_record(49)
    assert callback._should_record(13339)
    assert not callback._should_record(13338)
