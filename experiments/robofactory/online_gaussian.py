"""Online NoPoSplat conditioning matching the Table11 training cache contract."""

from __future__ import annotations

import os
import stat
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch


POLICY_LIGHTNING_COMMIT = "c944b4989a89c99c69d2572ea870f6a04680f5e7"
NOPOSPLAT_CHECKPOINT_BYTES = 2_448_469_295
COMPACT_HEIGHT = 28
COMPACT_WIDTH = 40
MOMENT_MATCH_METHOD = "opacity-aware-moment-matching-cell-mean-alpha-v2"


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _file_record(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    record = os.lstat(resolved)
    if stat.S_ISLNK(record.st_mode) or not stat.S_ISREG(record.st_mode):
        raise ValueError(f"Expected a direct regular file: {resolved}")
    return {
        "path": str(resolved),
        "bytes": int(record.st_size),
        "mtime_ns": int(record.st_mtime_ns),
    }


def unpack_gaussian_channels(
    tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if tensor.ndim < 3 or tensor.shape[-3] != 13:
        raise ValueError(f"Expected [...,13,H,W], got {tuple(tensor.shape)}")
    prefix = tuple(tensor.shape[:-3])
    height, width = tensor.shape[-2:]
    return (
        tensor[..., :3, :, :],
        tensor[..., 3:12, :, :].reshape(*prefix, 3, 3, height, width),
        tensor[..., 12:13, :, :],
    )


def pack_gaussian_channels(
    means: torch.Tensor,
    covariance: torch.Tensor,
    opacity: torch.Tensor,
) -> torch.Tensor:
    prefix = tuple(means.shape[:-3])
    height, width = means.shape[-2:]
    if tuple(means.shape[-3:]) != (3, height, width):
        raise ValueError(f"Invalid Gaussian mean shape: {tuple(means.shape)}")
    if tuple(covariance.shape) != (*prefix, 3, 3, height, width):
        raise ValueError(f"Invalid Gaussian covariance shape: {tuple(covariance.shape)}")
    if tuple(opacity.shape) != (*prefix, 1, height, width):
        raise ValueError(f"Invalid Gaussian opacity shape: {tuple(opacity.shape)}")
    return torch.cat(
        (means, covariance.reshape(*prefix, 9, height, width), opacity), dim=-3
    )


def correct_policy_lightning_legacy_covariance_order(
    tensor: torch.Tensor,
) -> torch.Tensor:
    if tensor.ndim < 3 or tensor.shape[-3] != 13:
        raise ValueError(f"Expected [...,13,H,W], got {tuple(tensor.shape)}")
    prefix = tuple(tensor.shape[:-3])
    height, width = tensor.shape[-2:]
    covariance_hw_ij = tensor[..., 3:12, :, :].reshape(
        *prefix, height, width, 3, 3
    )
    prefix_ndim = len(prefix)
    covariance = covariance_hw_ij.permute(
        *range(prefix_ndim),
        prefix_ndim + 2,
        prefix_ndim + 3,
        prefix_ndim,
        prefix_ndim + 1,
    ).contiguous()
    return pack_gaussian_channels(
        tensor[..., :3, :, :], covariance, tensor[..., 12:13, :, :]
    )


def _cell_index(
    height: int,
    width: int,
    output_height: int,
    output_width: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    rows = torch.div(
        torch.arange(height, device=device) * output_height,
        height,
        rounding_mode="floor",
    )
    columns = torch.div(
        torch.arange(width, device=device) * output_width,
        width,
        rounding_mode="floor",
    )
    return (rows[:, None] * output_width + columns[None, :]).reshape(-1).long()


def opacity_aware_moment_match(
    gaussian: torch.Tensor,
    *,
    output_size: tuple[int, int] = (COMPACT_HEIGHT, COMPACT_WIDTH),
    eps: float = 1e-8,
) -> torch.Tensor:
    """Apply the exact FP32 moment matcher used to build the training cache."""

    if gaussian.ndim < 3 or gaussian.shape[-3] != 13:
        raise ValueError(f"gaussian must be [...,13,H,W], got {tuple(gaussian.shape)}")
    output_height, output_width = map(int, output_size)
    height, width = map(int, gaussian.shape[-2:])
    if not (0 < output_height <= height and 0 < output_width <= width):
        raise ValueError(f"Invalid compact size {output_size} from {(height, width)}")
    prefix = tuple(gaussian.shape[:-3])
    flat_count = int(np.prod(prefix, dtype=np.int64)) if prefix else 1
    values = gaussian.reshape(flat_count, 13, height, width).float()
    means, covariance, opacity = unpack_gaussian_channels(values)
    pixel_count = height * width
    cell_count = output_height * output_width
    cells = _cell_index(
        height, width, output_height, output_width, device=values.device
    )
    means = means.reshape(flat_count, 3, pixel_count)
    covariance = covariance.reshape(flat_count, 3, 3, pixel_count)
    covariance = 0.5 * (covariance + covariance.transpose(1, 2))
    alpha = opacity.reshape(flat_count, 1, pixel_count).clamp(0.0, 1.0)
    scalar_cells = cells.view(1, 1, pixel_count).expand(flat_count, 1, -1)
    mean_cells = cells.view(1, 1, pixel_count).expand(flat_count, 3, -1)
    matrix_cells = cells.view(1, 1, pixel_count).expand(flat_count, 9, -1)
    weight = torch.zeros((flat_count, 1, cell_count), device=values.device)
    weight.scatter_add_(2, scalar_cells, alpha)
    weighted_mean = torch.zeros((flat_count, 3, cell_count), device=values.device)
    weighted_mean.scatter_add_(2, mean_cells, alpha * means)
    safe_weight = weight.clamp_min(float(eps))
    compact_mean = weighted_mean / safe_weight
    second_moment = covariance + means.unsqueeze(2) * means.unsqueeze(1)
    weighted_second = torch.zeros((flat_count, 9, cell_count), device=values.device)
    weighted_second.scatter_add_(
        2,
        matrix_cells,
        (alpha.unsqueeze(2) * second_moment).reshape(flat_count, 9, pixel_count),
    )
    compact_second = (weighted_second / safe_weight).reshape(
        flat_count, 3, 3, cell_count
    )
    compact_covariance = compact_second - compact_mean.unsqueeze(2) * compact_mean.unsqueeze(1)
    compact_covariance = 0.5 * (
        compact_covariance + compact_covariance.transpose(1, 2)
    )
    valid = weight > float(eps)
    compact_mean = torch.where(valid.expand(-1, 3, -1), compact_mean, 0.0)
    compact_covariance = torch.where(
        valid.unsqueeze(2).expand(-1, 3, 3, -1), compact_covariance, 0.0
    )
    cell_pixels = torch.zeros((flat_count, 1, cell_count), device=values.device)
    cell_pixels.scatter_add_(2, scalar_cells, torch.ones_like(alpha))
    compact_opacity = (weight / cell_pixels.clamp_min(1.0)).clamp(0.0, 1.0)
    packed = pack_gaussian_channels(
        compact_mean.reshape(flat_count, 3, output_height, output_width),
        compact_covariance.reshape(flat_count, 3, 3, output_height, output_width),
        compact_opacity.reshape(flat_count, 1, output_height, output_width),
    )
    if not bool(torch.isfinite(packed).all().item()):
        raise OverflowError("Compact Gaussian contains non-finite FP32 values")
    if float(packed.abs().max().item()) > float(torch.finfo(torch.float16).max):
        raise OverflowError("Compact Gaussian exceeds finite FP16 range")
    return packed.reshape(*prefix, 13, output_height, output_width).to(torch.float16)


def _contains_defaults(value: Any) -> bool:
    if isinstance(value, Mapping):
        return "defaults" in value or any(_contains_defaults(v) for v in value.values())
    if isinstance(value, list):
        return any(_contains_defaults(v) for v in value)
    return False


def _compose_encoder_config(repo_path: Path, config_path: Path):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    raw = OmegaConf.load(config_path)
    if _contains_defaults(OmegaConf.to_container(raw, resolve=False)):
        config_root = (repo_path / "config").resolve()
        relative = config_path.resolve().relative_to(config_root)
        with initialize_config_dir(
            version_base=None,
            config_dir=str(config_root),
            job_name="imagewam_online_gaussian",
        ):
            composed = compose(config_name=relative.with_suffix("").as_posix())
    else:
        composed = raw
    encoder_config = composed.encoder if "encoder" in composed else composed
    if _contains_defaults(OmegaConf.to_container(encoder_config, resolve=False)):
        raise ValueError("NoPoSplat encoder config still contains unresolved defaults")

    overrides: list[str] = []

    def force_unify(node, prefix: str = "") -> None:
        if OmegaConf.is_dict(node):
            for key in list(node.keys()):
                path = f"{prefix}.{key}" if prefix else str(key)
                if str(key) == "coor_type":
                    node[key] = "unify"
                    overrides.append(path)
                else:
                    force_unify(node[key], path)
        elif OmegaConf.is_list(node):
            for index, value in enumerate(node):
                force_unify(value, f"{prefix}[{index}]")

    force_unify(encoder_config, "encoder")
    if not overrides:
        raise ValueError("NoPoSplat config has no coor_type field to pin to unify")
    OmegaConf.resolve(encoder_config)
    return encoder_config, overrides


class OnlineNoPoSplatTeacher:
    """Pinned Policy-Lightning/NoPoSplat encoder for one policy query."""

    def __init__(
        self,
        *,
        repo_path: str | Path,
        checkpoint_path: str | Path,
        expected_commit: str = POLICY_LIGHTNING_COMMIT,
        expected_checkpoint_bytes: int = NOPOSPLAT_CHECKPOINT_BYTES,
        device: str | torch.device = "cuda",
    ) -> None:
        self.repo_path = Path(repo_path).expanduser().resolve(strict=True)
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve(strict=True)
        self.device = torch.device(device)
        self.expected_commit = str(expected_commit).lower()
        actual_commit = _git(self.repo_path, "rev-parse", "HEAD").lower()
        if actual_commit != self.expected_commit:
            raise ValueError(
                f"Policy-Lightning commit mismatch: expected={self.expected_commit} actual={actual_commit}"
            )
        if _git(self.repo_path, "status", "--porcelain=v1", "-uall"):
            raise ValueError(f"Policy-Lightning checkout is dirty: {self.repo_path}")
        checkpoint_record = _file_record(self.checkpoint_path)
        if checkpoint_record["bytes"] != int(expected_checkpoint_bytes):
            raise ValueError(
                "NoPoSplat checkpoint size mismatch: "
                f"expected={expected_checkpoint_bytes} actual={checkpoint_record['bytes']}"
            )
        self._checkpoint_record = checkpoint_record
        self._actual_commit = actual_commit
        self._encoder, self._coor_type_paths = self._load_encoder()

    def _load_encoder(self):
        config_path = self.repo_path / "config/encoder/noposplat.yaml"
        if not config_path.is_file():
            raise FileNotFoundError(f"Missing NoPoSplat config: {config_path}")
        existing_model = sys.modules.get("model")
        if existing_model is not None:
            existing_path = Path(getattr(existing_model, "__file__", "")).resolve()
            if self.repo_path not in existing_path.parents:
                raise RuntimeError("A different top-level model package is already imported")
        sys.path.insert(0, str(self.repo_path))
        try:
            from model.noposplat.encoder import get_encoder

            encoder_config, overrides = _compose_encoder_config(
                self.repo_path, config_path
            )
            encoder = get_encoder(encoder_config)
        finally:
            try:
                sys.path.remove(str(self.repo_path))
            except ValueError:
                pass
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, Mapping) or "state_dict" not in checkpoint:
            raise ValueError("NoPoSplat checkpoint must contain state_dict")
        weights = {
            str(key)[8:]: value
            for key, value in checkpoint["state_dict"].items()
            if str(key).startswith("encoder.")
        }
        if not weights:
            raise ValueError("NoPoSplat checkpoint has no encoder.* weights")
        missing, unexpected = encoder.load_state_dict(weights, strict=False)
        critical_missing = [
            str(key)
            for key in missing
            if "backbone" in str(key).lower() or "head" in str(key).lower()
        ]
        if critical_missing:
            raise ValueError(f"NoPoSplat is missing critical weights: {critical_missing[:32]}")
        self._missing_keys = sorted(map(str, missing))
        self._unexpected_keys = sorted(map(str, unexpected))
        encoder.to(self.device).eval()
        return encoder, overrides

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 5 or images.shape[2] != 3:
            raise ValueError(f"Teacher images must be [B,V,3,H,W], got {tuple(images.shape)}")
        raw = self._encoder(
            {"image": images.to(self.device, dtype=torch.float32, non_blocking=True)}
        )
        if not isinstance(raw, torch.Tensor) or raw.ndim != 5 or raw.shape[2] != 13:
            raise ValueError(f"NoPoSplat output must be [B,V,13,H,W], got {getattr(raw, 'shape', None)}")
        corrected = correct_policy_lightning_legacy_covariance_order(raw)
        if not bool(torch.isfinite(corrected).all().item()):
            raise ValueError("NoPoSplat emitted non-finite values")
        result = corrected.to(device="cpu", dtype=torch.float16).contiguous()
        if not bool(torch.isfinite(result).all().item()):
            raise OverflowError("NoPoSplat output overflows FP16")
        return result

    def provenance(self) -> dict[str, Any]:
        return {
            "kind": "external-policy-lightning-noposplat-online",
            "repository_commit": self._actual_commit,
            "repository_clean": True,
            "checkpoint": self._checkpoint_record,
            "config_relative_path": "config/encoder/noposplat.yaml",
            "coor_type": "unify",
            "coor_type_paths": self._coor_type_paths,
            "legacy_covariance_layout_corrected": True,
            "missing_weight_keys": self._missing_keys,
            "unexpected_weight_keys": self._unexpected_keys,
        }


def _camera_rgb_uint8(observation: Mapping[str, Any], camera_name: str) -> torch.Tensor:
    sensor_data = observation.get("sensor_data")
    if not isinstance(sensor_data, Mapping) or camera_name not in sensor_data:
        raise KeyError(f"RoboFactory observation lacks camera {camera_name!r}")
    image = torch.as_tensor(np.asarray(sensor_data[camera_name]["rgb"]))
    while image.ndim > 3 and image.shape[0] == 1:
        image = image[0]
    if image.ndim != 3:
        raise ValueError(f"{camera_name} RGB must be 3D, got {tuple(image.shape)}")
    if tuple(image.shape[:2]) == (240, 320) and image.shape[-1] >= 3:
        image = image[..., :3].permute(2, 0, 1)
    elif image.shape[0] >= 3 and tuple(image.shape[1:]) == (240, 320):
        image = image[:3]
    else:
        raise ValueError(f"{camera_name} RGB must be 240x320 HWC/CHW, got {tuple(image.shape)}")
    if image.dtype == torch.uint8:
        return image.contiguous()
    image = image.float()
    if not bool(torch.isfinite(image).all().item()):
        raise ValueError(f"{camera_name} RGB contains non-finite values")
    if float(image.min().item()) >= 0.0 and float(image.max().item()) <= 1.0 + 1e-6:
        image = image.mul(255.0)
    if float(image.min().item()) < 0.0 or float(image.max().item()) > 255.0 + 1e-6:
        raise ValueError(f"{camera_name} RGB range is invalid")
    return image.round().to(torch.uint8).contiguous()


def encode_compact_agent_gaussian(
    teacher: OnlineNoPoSplatTeacher,
    observation: Mapping[str, Any],
    agent_names: Sequence[str],
) -> torch.Tensor:
    sensor_data = observation.get("sensor_data")
    if not isinstance(sensor_data, Mapping):
        raise KeyError("RoboFactory observation lacks sensor_data")
    global_camera = (
        "head_camera_global"
        if "head_camera_global" in sensor_data
        else "head_camera"
    )
    global_rgb = _camera_rgb_uint8(observation, global_camera)
    # The single-robot Table11 environments expose only ``head_camera``.  The
    # training cache contract pairs that reference view with itself; N=2/3/4
    # environments expose the separate global and per-agent cameras.
    if len(agent_names) == 1 and "head_camera_agent0" not in sensor_data:
        agent_rgb = global_rgb.unsqueeze(0)
    else:
        agent_rgb = torch.stack(
            [
                _camera_rgb_uint8(observation, f"head_camera_agent{index}")
                for index, _ in enumerate(agent_names)
            ]
        )
    global_views = global_rgb.unsqueeze(0).expand(len(agent_names), -1, -1, -1)
    pairs = torch.stack((global_views, agent_rgb), dim=1).float().div(127.5).sub(1.0)
    pair_gaussian = teacher.encode(pairs.contiguous())
    expected = (len(agent_names), 2, 13, 240, 320)
    if tuple(pair_gaussian.shape) != expected or pair_gaussian.dtype != torch.float16:
        raise ValueError(f"Teacher output must be FP16 {expected}, got {tuple(pair_gaussian.shape)} {pair_gaussian.dtype}")
    compact = opacity_aware_moment_match(pair_gaussian[:, 1])
    compact_expected = (len(agent_names), 13, COMPACT_HEIGHT, COMPACT_WIDTH)
    if tuple(compact.shape) != compact_expected or compact.dtype != torch.float16:
        raise ValueError(f"Compact Gaussian must be FP16 {compact_expected}, got {tuple(compact.shape)} {compact.dtype}")
    return compact.contiguous()
