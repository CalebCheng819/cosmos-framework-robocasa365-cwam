# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Config-compatibility tests for switching cached training to online Wan serving."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.model.generator.tokenizers import wan2pt2_vae_4x16x16 as wan

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def test_online_wan_accepts_metadata_only_latent_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict] = []

    def fake_wan_vae(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(dtype=torch.bfloat16)

    monkeypatch.setattr(wan, "WanVAE", fake_wan_vae)

    tokenizer = wan.Wan2pt2VAEInterface(
        vae_path="/tmp/Wan2.2_VAE.pth",
        latent_ch=48,
        encode_exact_durations=[17, 61, 73],
    )

    assert tokenizer.latent_ch == 48
    assert calls == [
        {
            "dtype": torch.bfloat16,
            "vae_pth": "/tmp/Wan2.2_VAE.pth",
            "object_store_credential_path_pretrained": "",
            "temporal_window": {"256": 68, "480": 24, "720": 8},
            "encode_exact_durations": [17, 61, 73],
        }
    ]


@pytest.mark.parametrize("latent_ch", [32, 48.5, "48", True])
def test_online_wan_rejects_incompatible_latent_width(
    monkeypatch: pytest.MonkeyPatch,
    latent_ch: object,
) -> None:
    monkeypatch.setattr(
        wan,
        "WanVAE",
        lambda **_kwargs: pytest.fail("Wan weights must not load for an invalid latent contract"),
    )

    with pytest.raises(ValueError, match="latent_ch=48"):
        wan.Wan2pt2VAEInterface(
            vae_path="/tmp/Wan2.2_VAE.pth",
            latent_ch=latent_ch,
        )
