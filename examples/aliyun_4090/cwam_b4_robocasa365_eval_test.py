#!/usr/bin/env python3
"""Focused tests for B4 no-hash checkpoint and result validation."""

from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace


MODULE_PATH = Path(__file__).with_name("cwam_b4_robocasa365_eval.py")
SPEC = importlib.util.spec_from_file_location("cwam_b4_eval", MODULE_PATH)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _episode(success: bool = False) -> dict[str, object]:
    return {
        "success": success,
        "place_anytime": success,
        "steps": 2,
        "instruction": "perform the task",
        "action_log": [[0.0] * 7, [0.0] * 7],
    }


def _formal_tree(root: Path) -> SimpleNamespace:
    results_root = root / "results"
    run_name = "formal"
    results_dir = results_root / run_name
    results_dir.mkdir(parents=True)
    checkpoint = root / "checkpoint" / "model"
    checkpoint.mkdir(parents=True)
    for shard in range(8):
        args = {
            "task_shard": shard,
            "num_shards": 8,
            "episodes": 20,
            "episodes_start": 0,
            "max_steps": 400,
            "open_loop_horizon": 12,
            "seed": 42,
            "checkpoint": str(checkpoint.resolve()),
            "run_name": run_name,
            "results_dir": str(results_root.resolve()),
            "eval_dataset": None,
            "headless": True,
            "save_video": False,
            "split": "atomic",
        }
        _write(results_dir / f"args_shard{shard}.json", args)
        shard_results = {}
        for task in [task for index, task in enumerate(module.ATOMIC_TASKS) if index % 8 == shard]:
            episodes = [_episode(index == 0) for index in range(20)]
            shard_results[task] = {
                "success_rate": 0.05,
                "successes": 1,
                "place_anytime_rate": 0.05,
                "place_anytime_count": 1,
                "episodes": 20,
                "episode_results": episodes,
                "split": "atomic",
                "complete": True,
            }
        _write(results_dir / f"results_shard{shard}.json", shard_results)
    runtime = [root / "runtime-a.json", root / "runtime-b.json"]
    shutdown = [root / "shutdown-a.json", root / "shutdown-b.json"]
    gate = [root / "gate-a.json", root / "gate-b.json"]
    server_log = [root / "server-a.log", root / "server-b.log"]
    topology = (
        ("server-a", "ws://127.0.0.1:16000", "0,1,2,3"),
        ("server-b", "ws://127.0.0.1:16001", "4,5,6,7"),
    )
    for index, (label, uri, devices) in enumerate(topology):
        _write(runtime[index], {"status": "passed", "server_label": label, "uri": uri, "visible_devices": devices})
        _write(shutdown[index], {"status": "passed", "server_label": label, "uri": uri})
        _write(gate[index], {"status": "passed", "server_label": label, "policy_uri": uri})
        server_log[index].write_text("server exited cleanly\n", encoding="utf-8")
    return SimpleNamespace(
        experiment_id="exp",
        logical_run_id="run",
        source_commit=module.EXPECTED_SOURCE_COMMIT,
        demian_commit="2600ece5bf0eeacf010a95f084dcda2de6d30d59",
        run_name=run_name,
        results_dir=results_dir,
        checkpoint=checkpoint,
        runtime_receipt=runtime,
        gate_receipt=gate,
        shutdown_receipt=shutdown,
        server_log=server_log,
        server_exit_code=[0, 0],
        summary_csv=root / "summary.csv",
        task_csv=root / "tasks.csv",
        evidence_json=root / "evidence.json",
        status_json=root / "status.json",
    )


def test_complete_result() -> None:
    with tempfile.TemporaryDirectory() as directory:
        args = _formal_tree(Path(directory))
        evidence = module.summarize(args)
        assert evidence["status"] == "COMPLETE"
        assert evidence["result"]["successes"] == 17
        assert evidence["result"]["episodes"] == 340
        assert evidence["execution"]["policy_server_count"] == 2


def test_reject_bad_action_width() -> None:
    with tempfile.TemporaryDirectory() as directory:
        args = _formal_tree(Path(directory))
        path = args.results_dir / "results_shard0.json"
        value = json.loads(path.read_text())
        value["CloseBlenderLid"]["episode_results"][0]["action_log"][0] = [0.0] * 6
        _write(path, value)
        try:
            module.summarize(args)
        except module.ContractError as error:
            assert "width is not 7" in str(error)
        else:
            raise AssertionError("invalid action width was accepted")


if __name__ == "__main__":
    test_complete_result()
    test_reject_bad_action_width()
    print("2 tests passed")
