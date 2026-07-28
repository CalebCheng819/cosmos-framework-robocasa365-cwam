# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import subprocess
import unittest
from pathlib import Path


class FormalContinuationShellContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source_root = Path(__file__).resolve().parents[2]
        cls.submit_path = (
            cls.source_root
            / "examples/h_cluster/submit_cwam_cosmos3_nano_formal_continuation_rjob.sh"
        )
        cls.runner_path = (
            cls.source_root
            / "examples/h_cluster/run_cwam_cosmos3_nano_formal_continuation.sh"
        )
        cls.submit = cls.submit_path.read_text()
        cls.runner = cls.runner_path.read_text()

    def test_scripts_are_valid_bash(self) -> None:
        for path in (self.submit_path, self.runner_path):
            with self.subTest(path=path.name):
                subprocess.run(
                    ["bash", "-n", str(path)],
                    check=True,
                    capture_output=True,
                    text=True,
                )

    def test_submitter_derives_resources_from_the_pinned_best_cell(self) -> None:
        self.assertIn('if [[ "$SELECTED_WORKERS" == 12 || "$SELECTED_WORKERS" == 16 ]]', self.submit)
        self.assertIn("RJOB_CPU=160", self.submit)
        self.assertIn("RJOB_CPU=128", self.submit)
        self.assertIn('if [[ "$SELECTED_STORAGE" == nvme ]]', self.submit)
        self.assertIn("SUBMIT_ARGS+=(--store-host-nvme)", self.submit)
        self.assertNotIn("--store-host-nvme true", self.submit)
        self.assertIn("--expected-rjob-cpu \"$RJOB_CPU\"", self.submit)
        self.assertIn(
            "--expected-formal-wrapper run_cwam_cosmos3_nano_formal_continuation.sh",
            self.submit,
        )
        self.assertIn("--predict-only true", self.submit)
        self.assertIn('RJOB_PREDICT_ONLY:=1', self.submit)

    def test_runner_uses_exact_continuation_semantics(self) -> None:
        required = (
            "unset ROBOCASA365_FIXED_INDEX_TRACE_PATH",
            "export RESUME_EXISTING_RUN=1",
            "export ALLOW_DATA_STREAM_REPLAY=1",
            "trainer.max_iter=10000",
            "checkpoint.load_training_state=true",
            "checkpoint.keys_to_skip_loading=[]",
            "checkpoint.keys_not_to_resume=[dataloader]",
            "checkpoint.strict_resume=true",
            "--mode continuation",
        )
        for marker in required:
            with self.subTest(marker=marker):
                self.assertIn(marker, self.runner)
        self.assertNotIn("export ROBOCASA365_FIXED_INDEX_TRACE_PATH=", self.runner)

    def test_nvme_is_requested_and_staged_only_when_selected(self) -> None:
        self.assertIn('if [[ "$SELECTED_STORAGE" == nvme ]]; then', self.runner)
        self.assertIn("stage-nvme", self.runner)
        self.assertIn('--source "$GPFS_LATENT_CACHE_ROOT"', self.runner)
        self.assertIn('--destination "$NVME_CACHE_ROOT"', self.runner)
        self.assertIn("/dev/nvme1n1", self.runner)
        self.assertIn("RUNTIME_LATENT_CACHE_ROOT=\"$NVME_CACHE_ROOT\"", self.runner)


if __name__ == "__main__":
    unittest.main()
