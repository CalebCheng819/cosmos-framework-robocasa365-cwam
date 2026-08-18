"""CPU contracts for the C1 diagnostic summarizer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cosmos_framework.scripts.robocasa_stage_diagnostics import STAGE_ORDER, SUPPORTED_TASKS
from examples.h_cluster import summarize_cwam_c1_diagnostics as diagnostic


def _episode(terminal_index: int, *, success: bool = False) -> dict[str, object]:
    reached = {stage: index <= terminal_index for index, stage in enumerate(STAGE_ORDER)}
    first_step = {stage: index * 10 if reached[stage] else None for index, stage in enumerate(STAGE_ORDER)}
    return {
        "success": success,
        "place_anytime": terminal_index >= 4,
        "stage_reached": reached,
        "stage_first_step": first_step,
    }


def _write_condition(root: Path, condition: str, terminal_index: int) -> None:
    output = root / condition / "results" / condition
    output.mkdir(parents=True)
    value = {
        task: {
            "complete": True,
            "episode_results": [
                _episode(terminal_index, success=terminal_index == len(STAGE_ORDER) - 1),
                _episode(terminal_index, success=False),
            ],
        }
        for task in SUPPORTED_TASKS
    }
    (output / "results_shard0.json").write_text(json.dumps(value), encoding="utf-8")


@pytest.mark.parametrize("missing", ["normal_h12", "cross_episode_h12"])
def test_all_conditions_are_required(tmp_path: Path, missing: str) -> None:
    for condition in diagnostic.EXPECTED_CONDITIONS:
        if condition != missing:
            _write_condition(tmp_path, condition, 5)
    with pytest.raises(ValueError, match="Expected one result file"):
        diagnostic.summarize(tmp_path, episodes_per_task=2)


def test_stage_and_intervention_deltas(tmp_path: Path) -> None:
    for condition in diagnostic.EXPECTED_CONDITIONS:
        terminal = 5
        if condition in {"zero_h12", "shuffle_h12", "cross_episode_h12", "complement_h12"}:
            terminal = 2
        _write_condition(tmp_path, condition, terminal)

    result = diagnostic.summarize(tmp_path, episodes_per_task=2)

    assert result["conditions"]["normal_h12"]["aggregate"]["stages"]["release"]["reach_rate"] == 1.0
    assert result["conditions"]["zero_h12"]["aggregate"]["stages"]["transport"]["reach_rate"] == 0.0
    assert (
        result["interventions_vs_normal_h12"]["zero_h12"]["stage_reach_delta_vs_normal"]["release"]
        == -1.0
    )
    assert (
        result["interventions_vs_normal_h12"]["selected_h12"]["strict_success_delta_vs_normal"]
        == 0.0
    )
