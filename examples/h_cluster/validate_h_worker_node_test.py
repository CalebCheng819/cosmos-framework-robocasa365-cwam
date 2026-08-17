# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import unittest

from examples.h_cluster.validate_h_worker_node import validate_worker_node


class ValidateHWorkerNodeTest(unittest.TestCase):
    def test_accepts_scheduler_node_when_pod_hostname_differs(self) -> None:
        expected = "gpu-l-lg-cmc-h-h200-0258.host.h.pjlab.org.cn"
        result = validate_worker_node(expected, {"KUBEBRAIN_NODE_NAME": expected})
        self.assertEqual(result["scheduler_node"], expected)
        self.assertIn("pod_hostname", result)

    def test_rejects_wrong_scheduler_node(self) -> None:
        with self.assertRaisesRegex(ValueError, "expected scheduler node"):
            validate_worker_node(
                "gpu-l-lg-cmc-h-h200-0258.host.h.pjlab.org.cn",
                {
                    "KUBEBRAIN_NODE_NAME": (
                        "gpu-l-lg-cmc-h-h200-0257.host.h.pjlab.org.cn"
                    )
                },
            )

    def test_rejects_missing_scheduler_node(self) -> None:
        with self.assertRaisesRegex(ValueError, "KUBEBRAIN_NODE_NAME is absent"):
            validate_worker_node("gpu-l-lg-cmc-h-h200-0258.host.h.pjlab.org.cn", {})


if __name__ == "__main__":
    unittest.main()
