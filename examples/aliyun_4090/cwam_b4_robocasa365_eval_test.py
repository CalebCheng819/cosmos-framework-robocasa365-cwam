#!/usr/bin/env python3
"""Focused tests for B4 no-hash checkpoint and result validation."""

from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace


MODULE_PATH = Path(__file__).with_name("cwam_b4_robocasa365_eval.py")
LAUNCHER_PATH = Path(__file__).with_name("launch_cwam_b4_robocasa365_eval_dlc.sh")
RUNNER_PATH = Path(__file__).with_name("run_cwam_b4_robocasa365_eval_dlc.sh")
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


def test_worker_scratch_is_local() -> None:
    launcher = LAUNCHER_PATH.read_text(encoding="utf-8")
    runner = RUNNER_PATH.read_text(encoding="utf-8")
    assert 'export EVAL_WORKER_SCRATCH="$SCRATCH_ROOT/runtime/eval"' in launcher
    assert 'readonly SCRATCH_DIR="$EVAL_WORKER_SCRATCH"' in runner
    assert '[[ "$SCRATCH_DIR" == /tmp/* ]]' in runner
    assert 'readonly SCRATCH_DIR="$EVAL_OUTPUT_ROOT/scratch"' not in runner


def test_model_spec_binds_incomplete_b3_checkpoint() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory) / "checkpoint"
        model = root / "model"
        model.mkdir(parents=True)
        (model / ".metadata").write_bytes(b"meta")
        (model / "__0_0.distcp").write_bytes(b"weights")
        model_files = {".metadata": 4, "__0_0.distcp": 7}
        receipt = {
            "kind": "cwam_serving_ema_bf16_metadata_receipt",
            "status": "succeeded",
            "selection": {
                "key_count": 984,
                "chunk_count": 3931,
                "source_logical_bytes": 61_238_891_512,
                "target_dtype": "torch.bfloat16",
                "target_logical_bytes": 30_619_445_756,
            },
            "source": {"git_commit": "b" * 40, "run_id": "CWAM-B3-R2"},
            "target": {"file_count": 2, "shard_count": 1, "total_physical_bytes": 11},
            "validation": {"full_semantic_readback": True, "source_file_identities_stable": True},
        }
        manifest = {
            "kind": "cwam_serving_checkpoint_size_manifest",
            "file_count": 2,
            "total_physical_bytes": 11,
            "files": [{"relative_path": name, "size_bytes": size} for name, size in model_files.items()],
        }
        _write(root / "ema-bf16-metadata-receipt.json", receipt)
        _write(root / "serving-checkpoint-size-manifest.json", manifest)
        aux_files = {
            "ema-bf16-metadata-receipt.json": (root / "ema-bf16-metadata-receipt.json").stat().st_size,
            "serving-checkpoint-size-manifest.json": (root / "serving-checkpoint-size-manifest.json").stat().st_size,
        }
        spec = {
            "kind": "cwam_robocasa365_eval_model_spec",
            "model_id": "CWAM-B3-DSelect-K-R2",
            "checkpoint_name": "b3-r2-iter17500",
            "source": {"revision": "b" * 40, "run_id": "CWAM-B3-R2"},
            "training": {"checkpoint_iteration": 17_500, "target_iteration": 20_000, "complete": False},
            "ema": {
                "key_count": 984,
                "chunk_count": 3931,
                "source_logical_bytes": 61_238_891_512,
                "target_logical_bytes": 30_619_445_756,
            },
            "artifact": {
                "model_files": model_files,
                "aux_files": aux_files,
                "shard_count": 1,
                "file_count": 2,
                "total_physical_bytes": 11,
            },
        }
        spec_path = Path(directory) / "model-spec.json"
        _write(spec_path, spec)
        result = module.verify_checkpoint(root, Path(directory) / "verification.json", spec_path)
        assert result["model_id"] == "CWAM-B3-DSelect-K-R2"
        assert result["training"]["complete"] is False


def test_model_spec_rejects_false_training_complete_claim() -> None:
    spec = module._legacy_b4_model_spec()
    spec["training"] = {"checkpoint_iteration": 17_500, "target_iteration": 20_000, "complete": True}
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "bad-spec.json"
        _write(path, spec)
        try:
            module._load_model_spec(path)
        except module.ContractError as error:
            assert "training state" in str(error)
        else:
            raise AssertionError("false training-complete claim was accepted")


if __name__ == "__main__":
    test_complete_result()
    test_reject_bad_action_width()
    test_worker_scratch_is_local()
    test_model_spec_binds_incomplete_b3_checkpoint()
    test_model_spec_rejects_false_training_complete_claim()
    print("5 tests passed")
