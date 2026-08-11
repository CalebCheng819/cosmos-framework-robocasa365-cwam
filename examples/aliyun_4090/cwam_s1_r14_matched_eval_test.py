# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pytest

from examples.aliyun_4090 import cwam_s1_r14_matched_eval as matched


def _args(*, candidate: bool) -> dict[str, object]:
    return {
        "tasks": list(matched.TASKS) if candidate else [],
        "split": "atomic",
        "task_shard": 0,
        "num_shards": 1 if candidate else 8,
        "episodes": 20,
        "episodes_start": 0,
        "max_steps": 400,
        "open_loop_horizon": 12,
        "checkpoint": "/checkpoint/s1",
        "save_video": False,
        "headless": True,
        "camera_height": 256,
        "camera_width": 256,
        "seed": 42,
    }


def _task(successes: list[bool], *, complete: bool) -> dict[str, object]:
    return {
        "episodes": 20,
        "successes": sum(successes),
        "success_rate": sum(successes) / len(successes),
        "complete": complete,
        "episode_results": [{"success": success, "steps": 100 + index} for index, success in enumerate(successes)],
    }


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _namespace(tmp_path: Path) -> argparse.Namespace:
    paths = {
        name: tmp_path / name
        for name in (
            "candidate_args",
            "candidate_results",
            "baseline_args",
            "baseline_results",
            "summary_csv",
            "task_success_rates_csv",
            "comparison_json",
            "evidence_json",
            "verification_receipt",
        )
    }
    _write(paths["candidate_args"], _args(candidate=True))
    _write(paths["baseline_args"], _args(candidate=False))
    _write(
        paths["candidate_results"],
        {
            matched.TASKS[0]: _task([True] + [False] * 19, complete=True),
            matched.TASKS[1]: _task([False, True] + [False] * 18, complete=True),
        },
    )
    _write(
        paths["baseline_results"],
        {
            matched.TASKS[0]: _task([False] * 20, complete=True),
            matched.TASKS[1]: _task([False, False, True] + [False] * 15, complete=False),
        },
    )
    return argparse.Namespace(
        **{key: str(value) for key, value in paths.items()},
        experiment_id="experiment",
        logical_run_id="candidate-run",
        baseline_run_id="baseline-run",
        source_commit="a" * 40,
        checkpoint_path="/checkpoint/s1",
    )


def test_compare_writes_full_and_matched_results(tmp_path: Path) -> None:
    args = _namespace(tmp_path)
    matched.compare(args)

    comparison = json.loads(Path(args.comparison_json).read_text(encoding="utf-8"))
    assert comparison["status"] == "passed"
    assert comparison["aggregate"]["matched_episodes"] == 38
    assert comparison["aggregate"]["candidate_only"] == 2
    assert comparison["aggregate"]["baseline_only"] == 1
    with Path(args.summary_csv).open(encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 40


def test_compare_rejects_incomplete_candidate(tmp_path: Path) -> None:
    args = _namespace(tmp_path)
    candidate = json.loads(Path(args.candidate_results).read_text(encoding="utf-8"))
    candidate[matched.TASKS[0]] = _task([False] * 19, complete=False)
    _write(Path(args.candidate_results), candidate)

    with pytest.raises(ValueError, match="exactly 20"):
        matched.compare(args)


def test_compare_rejects_protocol_drift(tmp_path: Path) -> None:
    args = _namespace(tmp_path)
    candidate_args = json.loads(Path(args.candidate_args).read_text(encoding="utf-8"))
    candidate_args["open_loop_horizon"] = 8
    _write(Path(args.candidate_args), candidate_args)

    with pytest.raises(ValueError, match="open_loop_horizon mismatch"):
        matched.compare(args)


def test_mcnemar_exact_p() -> None:
    assert matched.mcnemar_exact_p(0, 0) == 1.0
    assert matched.mcnemar_exact_p(3, 0) == 0.25
