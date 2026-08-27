# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from pathlib import Path

import pytest
import torch

from cosmos_framework.model.generator.mot.action_relevance_replay import (
    FixedInterventionReplayBank,
    build_fixed_intervention_bank,
)


def _write_trace(path: Path, *, episode: int, task: str, step: int, value: float) -> None:
    torch.save(
        {
            "schema_version": 1,
            "metadata": [
                {
                    "evidence": {
                        "episode_key": f"libero:{episode:06d}",
                        "task_id": task,
                        "control_step": step,
                    }
                }
            ],
            "layers": [
                {
                    "module": "blocks.0.action_relevance",
                    "intervention_mode": "normal",
                    "routed_c": torch.full((1, 2, 4), value),
                }
            ],
        },
        path,
    )


def _complete_bank(tmp_path: Path) -> FixedInterventionReplayBank:
    for episode, task, value in (
        (0, "0", 1.0),
        (1, "0", 2.0),
        (2, "1", 3.0),
        (3, "1", 4.0),
    ):
        _write_trace(tmp_path / f"trace_{episode:08d}.pt", episode=episode, task=task, step=4, value=value)
    return FixedInterventionReplayBank(build_fixed_intervention_bank(tmp_path.glob("trace_*.pt"), seed=0))


def test_shuffle_uses_same_task_exact_step(tmp_path: Path) -> None:
    bank = _complete_bank(tmp_path)
    layers, donor, donor_task = bank.routed_c(
        condition="shuffle", episode_key="libero:000000", task_id=0, control_step=4
    )
    assert donor == "libero:000001"
    assert donor_task == "0"
    torch.testing.assert_close(
        layers["blocks.0.action_relevance"],
        torch.full((1, 2, 4), 2.0, dtype=torch.float16),
    )


def test_cross_episode_uses_different_task_exact_step(tmp_path: Path) -> None:
    bank = _complete_bank(tmp_path)
    _layers, donor, donor_task = bank.routed_c(
        condition="cross_episode", episode_key="libero:000000", task_id=0, control_step=4
    )
    assert donor != "libero:000000"
    assert donor_task == "1"


def test_bank_fails_closed_on_missing_pair_or_target(tmp_path: Path) -> None:
    _write_trace(tmp_path / "trace_00000000.pt", episode=0, task="0", step=0, value=1.0)
    _write_trace(tmp_path / "trace_00000001.pt", episode=1, task="1", step=0, value=2.0)
    with pytest.raises(ValueError, match="shuffle requires at least two episodes"):
        build_fixed_intervention_bank(tmp_path.glob("trace_*.pt"), seed=0)

    bank = _complete_bank(tmp_path)
    with pytest.raises(KeyError, match="no exact target"):
        bank.routed_c(condition="shuffle", episode_key="libero:000000", task_id=0, control_step=5)
