# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""CPU tests for generic DCP warm-start state-dict transformations."""

import json

import pytest
import torch
import torch.distributed.checkpoint as dcp

from cosmos_framework.checkpoint.dcp import (
    CustomLoadPlanner,
    _copy_reg_weights_to_ema,
    _state_dict_has_ema_tree,
    _write_local_json_atomic,
)


@pytest.mark.L0
@pytest.mark.CPU
def test_copy_regular_weights_to_ema_resets_every_matching_tensor() -> None:
    state_dict = {
        "net.layer.weight": torch.tensor([1.0, 2.0]),
        "net.layer.bias": torch.tensor([3.0]),
        "net_ema.layer.weight": torch.tensor([-1.0, -2.0]),
        "net_ema.layer.bias": torch.tensor([-3.0]),
        "other": torch.tensor([9.0]),
    }

    _copy_reg_weights_to_ema(state_dict)

    torch.testing.assert_close(state_dict["net_ema.layer.weight"], state_dict["net.layer.weight"])
    torch.testing.assert_close(state_dict["net_ema.layer.bias"], state_dict["net.layer.bias"])
    torch.testing.assert_close(state_dict["other"], torch.tensor([9.0]))


@pytest.mark.L0
@pytest.mark.CPU
def test_copy_regular_weights_to_ema_rejects_an_incomplete_ema_tree() -> None:
    with pytest.raises(AssertionError, match="net_ema.layer.weight"):
        _copy_reg_weights_to_ema({"net.layer.weight": torch.tensor([1.0])})


@pytest.mark.L0
@pytest.mark.CPU
def test_state_dict_has_ema_tree_distinguishes_ema_disabled_model() -> None:
    assert _state_dict_has_ema_tree(
        {
            "net.layer.weight": torch.tensor([1.0]),
            "net_ema.layer.weight": torch.tensor([1.0]),
        }
    )
    assert not _state_dict_has_ema_tree({"net.layer.weight": torch.tensor([1.0])})


@pytest.mark.L0
@pytest.mark.CPU
def test_custom_load_planner_loads_ema_tree_into_regular_target(tmp_path) -> None:
    regular = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
    ema = torch.tensor([3.125, 4.25], dtype=torch.float32)
    dcp.save(
        {
            "net.layer.weight": regular,
            "net_ema.layer.weight": ema,
        },
        checkpoint_id=tmp_path,
    )

    target = {"net.layer.weight": torch.empty_like(regular)}
    dcp.load(
        target,
        checkpoint_id=tmp_path,
        planner=CustomLoadPlanner(load_ema_to_reg=True),
    )

    torch.testing.assert_close(target["net.layer.weight"], ema.to(dtype=torch.bfloat16))
    assert not torch.equal(target["net.layer.weight"], regular)


@pytest.mark.L0
@pytest.mark.CPU
def test_checkpoint_completion_marker_is_published_as_complete_json(tmp_path) -> None:
    marker = tmp_path / "iter_000013000" / ".cwam_checkpoint_complete.json"
    payload = {"status": "complete", "iteration": 13_000}

    _write_local_json_atomic(str(marker), payload)

    assert json.loads(marker.read_text(encoding="utf-8")) == payload
    assert not list(marker.parent.glob(f"{marker.name}.tmp-*"))
