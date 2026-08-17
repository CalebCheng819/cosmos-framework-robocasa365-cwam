# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import unittest

from examples.h_cluster.validate_cwam_b2_training_log import (
    parse_training_records,
    validate_training_records,
)


class ValidateCwamB2TrainingLogTest(unittest.TestCase):
    def test_accepts_current_iter_speed_format_with_all_ranks(self) -> None:
        text = "\n".join(
            f"[08-11 09:54:03|INFO|iter_speed.py] [RANK {rank}] "
            f"Iteration 2: Hit counter: 2/50 | Loss: {11.0 + rank / 10:.4f} | Time: 75.79s"
            for rank in range(8)
        )
        result = validate_training_records(
            parse_training_records(text), expected_iteration=2, world_size=8
        )
        self.assertEqual(result["format"], "iter_speed")
        self.assertEqual(result["observed_ranks"], list(range(8)))

    def test_accepts_legacy_metric_format(self) -> None:
        text = "Iteration: 2, data loss 1.0, total loss 1.25"
        result = validate_training_records(
            parse_training_records(text), expected_iteration=2, world_size=8
        )
        self.assertEqual(result["format"], "legacy")
        self.assertAlmostEqual(result["mean_loss"], 1.25)

    def test_rejects_missing_rank_in_current_format(self) -> None:
        text = "\n".join(
            f"[RANK {rank}] Iteration 2: Hit counter: 2/50 | Loss: 1.0 | Time: 1.0s"
            for rank in range(7)
        )
        with self.assertRaisesRegex(ValueError, "rank coverage mismatch"):
            validate_training_records(
                parse_training_records(text), expected_iteration=2, world_size=8
            )

    def test_rejects_nonfinite_loss(self) -> None:
        text = "\n".join(
            f"[RANK {rank}] Iteration 2: Hit counter: 2/50 | Loss: "
            f"{'nan' if rank == 3 else '1.0'} | Time: 1.0s"
            for rank in range(8)
        )
        with self.assertRaisesRegex(ValueError, "non-finite"):
            validate_training_records(
                parse_training_records(text), expected_iteration=2, world_size=8
            )

    def test_rejects_absent_expected_iteration(self) -> None:
        text = "[RANK 0] Iteration 1: Hit counter: 1/50 | Loss: 1.0 | Time: 1.0s"
        with self.assertRaisesRegex(ValueError, "iteration 2 absent"):
            validate_training_records(
                parse_training_records(text), expected_iteration=2, world_size=1
            )


if __name__ == "__main__":
    unittest.main()
