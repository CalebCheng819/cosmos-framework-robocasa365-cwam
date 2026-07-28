# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from cosmos_framework.model.generator.tokenizers.precomputed_wan2pt2 import (
    PrecomputedWan2pt2LatentTokenizer,
)


def test_precomputed_wan_tokenizer_exposes_exact_metadata_without_parameters() -> None:
    tokenizer = PrecomputedWan2pt2LatentTokenizer(
        chunk_duration=93,
        spatial_compression_factor=16,
        temporal_compression_factor=4,
        causal=True,
        latent_ch=48,
        vae_path="/must/not/be/opened",
    )

    assert tokenizer.latent_ch == 48
    assert tokenizer.spatial_compression_factor == 16
    assert tokenizer.temporal_compression_factor == 4
    assert tokenizer.is_causal is True
    assert tokenizer.get_latent_num_frames(17) == 5
    assert tokenizer.get_pixel_num_frames(5) == 17
    assert tokenizer.latent_chunk_duration == 24


def test_precomputed_wan_tokenizer_fails_closed_on_pixel_operations() -> None:
    tokenizer = PrecomputedWan2pt2LatentTokenizer()

    with pytest.raises(RuntimeError, match="cannot encode pixels"):
        tokenizer.encode(torch.zeros(1, 3, 17, 160, 320))
    with pytest.raises(RuntimeError, match="cannot decode latents"):
        tokenizer.decode(torch.zeros(1, 48, 5, 10, 20))


def test_robocasa_toml_resolves_and_instantiates_metadata_only_tokenizer() -> None:
    repository = Path(__file__).resolve().parents[4]
    script = """
import json
from cosmos_framework.configs.toml_config.sft_config import load_experiment_from_toml
from cosmos_framework.utils.lazy_config import instantiate

config = load_experiment_from_toml("examples/toml/sft_config/action_policy_robocasa365_nano.toml")
tokenizer_config = config.model.config.tokenizer
tokenizer = instantiate(tokenizer_config)
print(json.dumps({
    "target": str(tokenizer_config["_target_"]),
    "class": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
    "latent_ch": tokenizer.latent_ch,
    "latent_frames_17": tokenizer.get_latent_num_frames(17),
    "has_model_weights": hasattr(tokenizer, "model"),
}))
"""
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": str(repository),
            "ROBOCASA365_ROOT": "/tmp/robocasa365-view",
            "ROBOCASA365_SOURCE_ROOT": "/tmp/robocasa365-source",
            "ROBOCASA365_LATENT_CACHE_ROOT": "/tmp/robocasa365-cache",
            "ROBOCASA365_LATENT_CACHE_MODE": "required",
            "CWAM_METADATA_ONLY_VIDEO_TOKENIZER": "1",
            "BASE_CHECKPOINT_PATH": "/tmp/cosmos3-nano",
            "WAN_VAE_PATH": "/tmp/Wan2.2_VAE.pth",
            "QWEN3_VL_TOKENIZER_PATH": "/tmp/Qwen3-VL-8B-Instruct",
            "IMAGINAIRE_OUTPUT_ROOT": "/tmp/cwam-output",
        }
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repository,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        # A cold import of the full Hydra/Megatron stack from H-cluster GPFS
        # can exceed three minutes even though the metadata tokenizer itself
        # is weight-free.  Keep the subprocess isolation, but do not turn
        # shared-filesystem import latency into a false functional failure.
        timeout=600,
    )
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    target = (
        "cosmos_framework.model.generator.tokenizers.precomputed_wan2pt2."
        "PrecomputedWan2pt2LatentTokenizer"
    )
    assert payload == {
        "target": target,
        "class": target,
        "latent_ch": 48,
        "latent_frames_17": 5,
        "has_model_weights": False,
    }
