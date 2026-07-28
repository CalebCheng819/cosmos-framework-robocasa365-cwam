# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Environment-boundary tests for the RoboCasa365 video-tokenizer selection."""

from __future__ import annotations

from pathlib import Path

import pytest

from cosmos_framework.configs.base.experiment.action.posttrain_config.action_policy_robocasa365_nano import (
    _action_policy_robocasa365_nano_model_config,
)
from cosmos_framework.configs.toml_config.sft_config import load_experiment_from_toml


_METADATA_TARGET = (
    "cosmos_framework.model.generator.tokenizers.precomputed_wan2pt2."
    "PrecomputedWan2pt2LatentTokenizer"
)
_REAL_WAN_TARGET = (
    "cosmos_framework.model.generator.tokenizers.wan2pt2_vae_4x16x16."
    "Wan2pt2VAEInterface"
)


def _model_config(
    monkeypatch: pytest.MonkeyPatch,
    *,
    metadata_flag: str,
    cache_mode: str,
) -> dict:
    monkeypatch.setenv("CWAM_METADATA_ONLY_VIDEO_TOKENIZER", metadata_flag)
    monkeypatch.setenv("ROBOCASA365_LATENT_CACHE_MODE", cache_mode)
    if cache_mode == "required":
        monkeypatch.setenv("ROBOCASA365_LATENT_CACHE_ROOT", "/tmp/robocasa365-cache")
    else:
        monkeypatch.delenv("ROBOCASA365_LATENT_CACHE_ROOT", raising=False)
    return _action_policy_robocasa365_nano_model_config()


def test_metadata_only_flag_requires_required_cache_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(
        ValueError,
        match=(
            "CWAM_METADATA_ONLY_VIDEO_TOKENIZER=1 requires "
            "ROBOCASA365_LATENT_CACHE_MODE=required"
        ),
    ):
        _model_config(monkeypatch, metadata_flag="1", cache_mode="disabled")


@pytest.mark.parametrize("invalid_flag", ["true", "yes", "2", "-1", ""])
def test_metadata_only_flag_rejects_every_value_other_than_zero_or_one(
    monkeypatch: pytest.MonkeyPatch,
    invalid_flag: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="CWAM_METADATA_ONLY_VIDEO_TOKENIZER must be exactly 0 or 1",
    ):
        _model_config(monkeypatch, metadata_flag=invalid_flag, cache_mode="required")


def test_zero_metadata_only_flag_keeps_the_real_wan_tokenizer_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _model_config(monkeypatch, metadata_flag="0", cache_mode="disabled")
    repository = Path(__file__).resolve().parents[4]
    monkeypatch.chdir(repository)
    monkeypatch.setenv("ROBOCASA365_ROOT", "/tmp/robocasa365-view")
    monkeypatch.setenv("ROBOCASA365_SOURCE_ROOT", "/tmp/robocasa365-source")
    monkeypatch.setenv("BASE_CHECKPOINT_PATH", "/tmp/cosmos3-nano")
    monkeypatch.setenv("WAN_VAE_PATH", "/tmp/Wan2.2_VAE.pth")
    monkeypatch.setenv("QWEN3_VL_TOKENIZER_PATH", "/tmp/Qwen3-VL-8B-Instruct")
    monkeypatch.setenv("IMAGINAIRE_OUTPUT_ROOT", "/tmp/cwam-output")
    config = load_experiment_from_toml(
        "examples/toml/sft_config/action_policy_robocasa365_nano.toml"
    )
    target = config.model.config.tokenizer["_target_"]

    assert f"{target.__module__}.{target.__qualname__}" == _REAL_WAN_TARGET


def test_one_metadata_only_flag_selects_metadata_target_with_required_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _model_config(monkeypatch, metadata_flag="1", cache_mode="required")

    assert config["tokenizer"]["_target_"] == _METADATA_TARGET
