#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Validate the physical H-cluster node assigned by the scheduler."""

from __future__ import annotations

import argparse
import json
import os
import socket
from collections.abc import Mapping


def validate_worker_node(expected: str, environ: Mapping[str, str]) -> dict[str, str]:
    scheduler_node = environ.get("KUBEBRAIN_NODE_NAME")
    if not scheduler_node:
        raise ValueError("KUBEBRAIN_NODE_NAME is absent from the worker environment")
    if scheduler_node != expected:
        raise ValueError(
            f"expected scheduler node {expected}, got {scheduler_node}"
        )
    return {
        "expected_node": expected,
        "scheduler_node": scheduler_node,
        "pod_hostname": socket.gethostname(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected", required=True)
    args = parser.parse_args()
    try:
        result = validate_worker_node(args.expected, os.environ)
    except ValueError as error:
        raise SystemExit(f"ERROR: {error}") from error
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
