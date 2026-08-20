#!/usr/bin/env python3

import argparse
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("cwam_robocasa365_matched_fast_eval.py")
SPEC = importlib.util.spec_from_file_location("matched_fast_eval", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class MatchedFastEvalTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
