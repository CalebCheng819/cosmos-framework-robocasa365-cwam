#!/usr/bin/env python3

import argparse
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("cwam_robocasa365_matched_fast_eval.py")
RUNNER_PATH = Path(__file__).with_name("run_cwam_cosmos3_robocasa365_matched_fast_eval.sh")
FULL_RUNNER_PATH = Path(__file__).with_name("run_cwam_cosmos3_robocasa365_full_fast_eval.sh")
FULL_SUBMITTER_PATH = Path(__file__).with_name("submit_cwam_cosmos3_robocasa365_full_fast_eval_rjob.sh")
SPEC = importlib.util.spec_from_file_location("matched_fast_eval", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class MatchedFastEvalTest(unittest.TestCase):
    def test_serving_checkpoint_root_resolves_to_model_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "serving"
            model = root / "model"
            model.mkdir(parents=True)
            self.assertEqual(MODULE.resolved_server_checkpoint(root), model.resolve())
            self.assertEqual(MODULE.resolved_server_checkpoint(model), model.resolve())

    def test_evaluators_use_the_dynamic_server_ports(self) -> None:
        runner = RUNNER_PATH.read_text()
        self.assertIn(
            '"${SERVER_PORTS[${EVAL_SERVER_INDEX[0]}]}" "${EVAL_GPU_DEVICES[0]}" 42',
            runner,
        )
        self.assertIn(
            '"${SERVER_PORTS[${EVAL_SERVER_INDEX[1]}]}" "${EVAL_GPU_DEVICES[1]}" 10042',
            runner,
        )
        self.assertNotIn('"$RUN_A" 16000', runner)
        self.assertNotIn('"$RUN_B" 16001', runner)

    def test_runner_supports_six_and_eight_gpu_layouts(self) -> None:
        runner = RUNNER_PATH.read_text()
        self.assertIn("SERVER_DEVICES=(0,1,2,3)", runner)
        self.assertIn("EVAL_SERVER_INDEX=(0 0)", runner)
        self.assertIn("EVAL_GPU_DEVICES=(4 5)", runner)
        self.assertIn("SERVER_DEVICES=(0,1,2,3 4,5,6,7)", runner)
        self.assertIn("EVAL_SERVER_INDEX=(0 1)", runner)

    def test_sanitized_keeps_parallelism_and_removes_digest_fields(self) -> None:
        value = MODULE.sanitized({
            "dp_shard_size": 4,
            "checkpoint_metadata_sha256": "unused",
            "checkpoint_weight_receipt": {"key_count": 822},
        })
        self.assertEqual(value, {"dp_shard_size": 4})

    def test_summary_accepts_two_complete_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            specs = []
            for task, successes in (("CloseBlenderLid", 2), ("PickPlaceCounterToStove", 1)):
                path = root / f"{task}.json"
                episodes = [{"success": index < successes} for index in range(3)]
                path.write_text(json.dumps({task: {"complete": True, "episode_results": episodes}}))
                specs.append(f"{task}={path}")
            receipts = []
            for name in ("runtime-a", "runtime-b", "shutdown-a", "shutdown-b"):
                path = root / f"{name}.json"
                path.write_text(json.dumps({"status": "passed"}))
                receipts.append(path)
            checkpoint = root / "model"
            checkpoint.mkdir()
            progress = root / "progress.ndjson"
            progress.write_text("{}\n")
            args = argparse.Namespace(
                experiment_id="experiment",
                logical_run_id="run",
                model_label="model",
                source_revision="a" * 40,
                checkpoint=checkpoint,
                node="node/0252",
                rjob_name="test-rjob",
                task_spec=specs,
                expected_episodes=3,
                runtime_receipt=receipts[:2],
                shutdown_receipt=receipts[2:],
                progress_log=progress,
                output=root / "summary.json",
                csv_output=root / "summary.csv",
            )
            value = MODULE.summarize(args)
            self.assertEqual(value["aggregate"], {"episodes": 6, "successes": 3, "success_rate": 0.5})
            self.assertTrue(args.output.is_file())
            self.assertTrue(args.csv_output.is_file())

    def test_check_task_requires_exact_complete_episode_count(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "result.json"
            path.write_text(json.dumps({"Task": {"complete": True, "episode_results": [{"success": True}] * 100}}))
            state = MODULE.check_task(argparse.Namespace(task="Task", result=path, expected_episodes=100))
            self.assertEqual(state["completed"], 100)
            with self.assertRaises(MODULE.ContractError):
                MODULE.check_task(argparse.Namespace(task="Task", result=path, expected_episodes=99))

    def test_full_runner_has_canonical_tasks_and_durable_progress(self) -> None:
        runner = FULL_RUNNER_PATH.read_text()
        expected_tasks = (
            "CloseBlenderLid", "CloseFridge", "CloseToasterOvenDoor", "CoffeeSetupMug",
            "OpenCabinet", "OpenDrawer", "OpenStandMixerHead", "PickPlaceCounterToCabinet",
            "PickPlaceCounterToStove", "PickPlaceDrawerToCounter", "PickPlaceSinkToCounter",
            "PickPlaceToasterToCounter", "SlideDishwasherRack", "TurnOffStove",
            "TurnOnElectricKettle", "TurnOnMicrowave", "TurnOnSinkFaucet",
        )
        for task in expected_tasks:
            self.assertIn(task, runner)
        self.assertIn("readonly EPISODES_PER_TASK=100", runner)
        self.assertIn("readonly -a SERVER_DEVICES=(0,1,2,3 4,5,6,7)", runner)
        self.assertIn("NEXT_TASK[$slot]=$((next + 2))", runner)
        self.assertIn('--pid "${SERVER_PIDS[0]}" --pid "${SERVER_PIDS[1]}"', runner)
        self.assertIn('--seed 42 --split atomic', runner)
        self.assertIn('>"$CONTROL_DIR/check-$task.json"', runner)
        self.assertIn("cwam-robocasa365-full-fast-eval/runs", runner)

    def test_full_submitter_is_pinned_to_0252_eight_gpu_priority_nine(self) -> None:
        submitter = FULL_SUBMITTER_PATH.read_text()
        self.assertIn("gpu-l-lg-cmc-h-h200-0252.host.h.pjlab.org.cn", submitter)
        self.assertIn("--private-machine group --priority 9", submitter)
        self.assertIn("--cpu 126 --gpu 8 --memory 900000", submitter)
        self.assertIn("--positive-tags \"$EXPECTED_NODE_TAG\"", submitter)
        self.assertIn("--predict-only true", submitter)


if __name__ == "__main__":
    unittest.main()
