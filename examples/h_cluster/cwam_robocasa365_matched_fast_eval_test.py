#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("cwam_robocasa365_matched_fast_eval.py")
SPEC = importlib.util.spec_from_file_location("matched_fast_eval", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class WireContractTest(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("msgpack"), "msgpack is unavailable")
    def test_unpack_msgpack_object_accepts_binary_object(self) -> None:
        import msgpack

        wire = msgpack.packb({"status": "ok"}, use_bin_type=True)
        self.assertEqual(MODULE.unpack_msgpack_object(wire, "reply"), {"status": "ok"})

    def test_unpack_msgpack_object_reports_server_text_error(self) -> None:
        with self.assertRaisesRegex(
            MODULE.ContractError,
            "shutdown acknowledgement returned a server-side text error:.*rank 0",
        ):
            MODULE.unpack_msgpack_object(
                "Traceback: Controlled shutdown is valid only on rank 0\n",
                "shutdown acknowledgement",
            )


def write_result(
    path: Path,
    task: str,
    successes: list[bool],
    *,
    start: int | None = None,
    stop: int | None = None,
    target: int | None = None,
    complete: bool = False,
) -> None:
    episodes = []
    for offset, success in enumerate(successes):
        episode = {"success": success, "place_anytime": False}
        if start is not None:
            episode["episode_index"] = start + offset
        episodes.append(episode)
    value = {
        "successes": sum(successes),
        "success_rate": sum(successes) / len(successes),
        "episodes": len(successes),
        "episode_results": episodes,
        "complete": complete,
    }
    if start is not None:
        value.update(
            episode_range_start=start,
            episode_range_stop=stop,
            target_episodes=target,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({task: value}), encoding="utf-8")


class ResumeMergeTest(unittest.TestCase):
    def test_resume_state_and_strict_merge(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            task = "TurnOnMicrowave"
            source = root / "source.json"
            first = root / "first.json"
            second = root / "second.json"
            output = root / "merged.json"
            receipt = root / "receipt.json"
            write_result(source, task, [True, False])
            write_result(first, task, [False, True], start=2, stop=4, target=5, complete=True)
            write_result(second, task, [True], start=4, stop=5, target=5, complete=True)

            state = MODULE.resume_task_state(task, source, [(2, 4, first), (4, 5, second)], 5)
            self.assertEqual(state["completed"], 5)
            self.assertEqual(state["successes"], 3)
            self.assertTrue(state["complete"])

            merged = MODULE.merge_resume_fragments(
                argparse.Namespace(
                    task=task,
                    source=source,
                    fragment=[f"2:4={first}", f"4:5={second}"],
                    expected_episodes=5,
                    output=output,
                    receipt=receipt,
                )
            )
            self.assertEqual(merged["final_episodes"], 5)
            self.assertEqual(merged["final_successes"], 3)
            final = json.loads(output.read_text(encoding="utf-8"))[task]
            self.assertTrue(final["complete"])
            self.assertEqual(len(final["episode_results"]), 5)

    def test_merge_rejects_duplicate_episode(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            task = "TurnOnSinkFaucet"
            source = root / "source.json"
            fragment = root / "fragment.json"
            write_result(source, task, [False, True, False])
            write_result(fragment, task, [True, False], start=2, stop=4, target=4, complete=True)
            with self.assertRaisesRegex(MODULE.ContractError, "duplicate absolute episode"):
                MODULE.merge_resume_fragments(
                    argparse.Namespace(
                        task=task,
                        source=source,
                        fragment=[f"2:4={fragment}"],
                        expected_episodes=4,
                        output=root / "merged.json",
                        receipt=root / "receipt.json",
                    )
                )

    def test_merge_rejects_gap(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            task = "TurnOnSinkFaucet"
            source = root / "source.json"
            fragment = root / "fragment.json"
            write_result(source, task, [False])
            write_result(fragment, task, [True], start=2, stop=3, target=3, complete=True)
            with self.assertRaisesRegex(MODULE.ContractError, "inventory mismatch"):
                MODULE.merge_resume_fragments(
                    argparse.Namespace(
                        task=task,
                        source=source,
                        fragment=[f"2:3={fragment}"],
                        expected_episodes=3,
                        output=root / "merged.json",
                        receipt=root / "receipt.json",
                    )
                )


if __name__ == "__main__":
    unittest.main()
