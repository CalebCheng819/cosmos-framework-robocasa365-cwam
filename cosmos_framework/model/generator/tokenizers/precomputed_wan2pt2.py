# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Metadata-only Wan2.2 tokenizer for strict precomputed-latent training.

The regular :class:`Wan2pt2VAEInterface` eagerly loads roughly 2.8 GB of VAE
weights.  A training recipe that requires a checksum-bound latent cache never
calls ``encode`` or ``decode``, so retaining those weights only consumes GPU
memory.  This implementation preserves the tokenizer shape/time contract while
failing closed if a pixel encode/decode is attempted.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from cosmos_framework.model.generator.tokenizers.interface import VideoTokenizerInterface


class PrecomputedWan2pt2LatentTokenizer(VideoTokenizerInterface):
    """Expose the Wan2.2 latent contract without loading its VAE weights."""

    def __init__(
        self,
        *,
        chunk_duration: int = 93,
        spatial_compression_factor: int = 16,
        temporal_compression_factor: int = 4,
        causal: bool = True,
        latent_ch: int = 48,
        bucket_name: str = "",
        object_store_credential_path_pretrained: str = "",
        vae_path: str = "",
        keep_decoder_cache: bool = False,
        use_streaming_encode: bool = False,
        encode_chunk_frames: int | Mapping[str, int] = 4,
        encode_exact_durations: list[int] | None = None,
        temporal_window: int | None = None,
        encode_bucket_multiple: int | None = None,
    ) -> None:
        # Hydra merges the recipe's tokenizer override with the registered Wan
        # tokenizer config. Accept and intentionally ignore weight-loading-only
        # fields such as vae_path, bucket_name and encode_chunk_frames.
        del (
            bucket_name,
            object_store_credential_path_pretrained,
            vae_path,
            keep_decoder_cache,
            use_streaming_encode,
            encode_chunk_frames,
            encode_exact_durations,
            temporal_window,
            encode_bucket_multiple,
        )
        if int(spatial_compression_factor) != 16:
            raise ValueError("Precomputed Wan2.2 latents require spatial_compression_factor=16.")
        if int(temporal_compression_factor) != 4:
            raise ValueError("Precomputed Wan2.2 latents require temporal_compression_factor=4.")
        if bool(causal) is not True:
            raise ValueError("Precomputed Wan2.2 latents require causal=True.")
        if int(latent_ch) != 48:
            raise ValueError("Precomputed Wan2.2 latents require latent_ch=48.")
        if int(chunk_duration) <= 0:
            raise ValueError("chunk_duration must be positive.")

        self._chunk_duration = int(chunk_duration)
        self._spatial_compression_factor = 16
        self._temporal_compression_factor = 4
        self._latent_ch = 48
        self._causal = True

    def reset_dtype(self) -> None:
        """No-op: this metadata-only tokenizer owns no parameters."""

    def encode(self, state: torch.Tensor) -> torch.Tensor:
        del state
        raise RuntimeError(
            "PrecomputedWan2pt2LatentTokenizer cannot encode pixels; "
            "ROBOCASA365_LATENT_CACHE_MODE=required must supply video_latent."
        )

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        del latent
        raise RuntimeError("PrecomputedWan2pt2LatentTokenizer cannot decode latents.")

    def get_latent_num_frames(self, num_pixel_frames: int) -> int:
        if int(num_pixel_frames) <= 0:
            raise ValueError("num_pixel_frames must be positive.")
        return 1 + (int(num_pixel_frames) - 1) // self._temporal_compression_factor

    def get_pixel_num_frames(self, num_latent_frames: int, **kwargs: Any) -> int:
        del kwargs
        if int(num_latent_frames) <= 0:
            raise ValueError("num_latent_frames must be positive.")
        return (int(num_latent_frames) - 1) * self._temporal_compression_factor + 1

    @property
    def spatial_compression_factor(self) -> int:
        return self._spatial_compression_factor

    @property
    def temporal_compression_factor(self) -> int:
        return self._temporal_compression_factor

    @property
    def spatial_resolution(self) -> int:
        return 512

    @property
    def pixel_chunk_duration(self) -> int:
        return self._chunk_duration

    @property
    def latent_chunk_duration(self) -> int:
        return self.get_latent_num_frames(self._chunk_duration)

    @property
    def latent_ch(self) -> int:
        return self._latent_ch

    @property
    def name(self) -> str:
        return "precomputed_wan2pt2_latent_tokenizer"
