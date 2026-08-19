"""CPU tests for selective DCP extraction and subtree overlay."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import pytest
import torch
import torch.distributed.checkpoint as dcp

from examples.aliyun_4090 import cwam_dcp_prefix_overlay as tool


def _load(path: Path) -> dict[str, torch.Tensor]:
    with (path / ".metadata").open("rb") as stream:
        metadata = pickle.load(stream)
    state = {
        key: torch.empty(tuple(value.size), dtype=value.properties.dtype)
        for key, value in metadata.state_dict_metadata.items()
    }
    dcp.load(state, checkpoint_id=path)
    return state


@pytest.mark.CPU
def test_extract_then_overlay_replaces_only_selected_prefix(tmp_path: Path) -> None:
    prefix = "net_ema.strong_action_expert."
    base_model = tmp_path / "base-model"
    full_model = tmp_path / "full-model"
    config = tmp_path / "config.yaml"
    config.write_text("model: test\n", encoding="utf-8")
    base = {
        "net_ema.language_model.weight": torch.tensor([1, 2], dtype=torch.bfloat16),
        f"{prefix}old": torch.tensor([3], dtype=torch.bfloat16),
    }
    full = {
        "net_ema.language_model.weight": torch.tensor([1, 2], dtype=torch.bfloat16),
        f"{prefix}new": torch.tensor([7, 8], dtype=torch.bfloat16),
        f"{prefix}memory": torch.tensor([9], dtype=torch.bfloat16),
    }
    dcp.save(base, checkpoint_id=base_model)
    dcp.save(full, checkpoint_id=full_model)

    delta_artifact = tmp_path / "delta"
    tool.extract(
        type(
            "Args",
            (),
            {
                "source_model": full_model,
                "target_artifact": delta_artifact,
                "prefix": prefix,
                "expected_key_count": 2,
                "expected_logical_bytes": 6,
                "source_config": config,
                "source_revision": "test-revision",
                "source_run_id": "test-run",
                "seal": False,
            },
        )()
    )
    output_artifact = tmp_path / "output"
    tool.overlay(
        type(
            "Args",
            (),
            {
                "base_model": base_model,
                "delta_model": delta_artifact / "model",
                "contract_model": full_model,
                "target_artifact": output_artifact,
                "prefix": prefix,
                "expected_base_key_count": 2,
                "expected_kept_key_count": 1,
                "expected_delta_key_count": 2,
                "expected_output_key_count": 3,
                "expected_logical_bytes": 10,
                "source_config": config,
                "source_revision": "test-revision",
                "source_run_id": "test-run",
                "seal": False,
            },
        )()
    )

    output = _load(output_artifact / "model")
    assert set(output) == set(full)
    for key in full:
        torch.testing.assert_close(output[key], full[key])
    receipt = json.loads((output_artifact / "dcp-prefix-overlay-receipt.json").read_text())
    assert receipt["status"] == "succeeded"
    assert receipt["kept_base_key_count"] == 1
    assert receipt["delta_key_count"] == 2


@pytest.mark.CPU
def test_overlay_rejects_delta_key_outside_prefix(tmp_path: Path) -> None:
    base_model = tmp_path / "base-model"
    delta_model = tmp_path / "delta-model"
    dcp.save({"net_ema.base": torch.ones(1, dtype=torch.bfloat16)}, checkpoint_id=base_model)
    dcp.save({"net_ema.wrong": torch.ones(1, dtype=torch.bfloat16)}, checkpoint_id=delta_model)
    args = type(
        "Args",
        (),
        {
            "base_model": base_model,
            "delta_model": delta_model,
            "contract_model": None,
            "target_artifact": tmp_path / "output",
            "prefix": "net_ema.strong_action_expert.",
            "expected_base_key_count": None,
            "expected_kept_key_count": None,
            "expected_delta_key_count": None,
            "expected_output_key_count": None,
            "expected_logical_bytes": None,
            "source_config": None,
            "source_revision": "test-revision",
            "source_run_id": "test-run",
            "seal": False,
        },
    )()
    with pytest.raises(tool.OverlayError, match="outside the replacement prefix"):
        tool.overlay(args)
