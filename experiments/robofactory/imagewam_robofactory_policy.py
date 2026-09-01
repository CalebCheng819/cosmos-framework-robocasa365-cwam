"""Closed-loop RoboFactory adapter for the ImageWAM Table11 checkpoint."""

from __future__ import annotations

import math
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate

from experiments.robofactory.online_gaussian import (
    MOMENT_MATCH_METHOD,
    NOPOSPLAT_CHECKPOINT_BYTES,
    POLICY_LIGHTNING_COMMIT,
    OnlineNoPoSplatTeacher,
    encode_compact_agent_gaussian,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_NAME = "train"
TASK_CONFIG_NAME = "robofactory_table11_flux2_klein_4b_videogen_hub_spatial_gaussian"
TASK_INSTRUCTIONS = {
    "LongPipelineDelivery-rf": "Deliver the objects through the long pipeline with four robots.",
    "StackCube-rf": "Stack the cubes with one robot.",
    "StrikeCube-rf": "Strike the cube to the target with one robot.",
    "TakePhoto-rf": "Coordinate four robots to take the photo.",
    "PlaceFood-rf": "Place the food at the target location with two robots.",
    "PickMeat-rf": "Pick up the meat with one robot.",
    "PassShoe-rf": "Pass the shoe between two robots.",
    "CameraAlignment-rf": "Align the camera with three robots.",
    "LiftBarrier-rf": "Lift the barrier with two robots.",
    "ThreeRobotsStackCube-rf": "Stack the cubes with three robots.",
    "TwoRobotsStackCube-rf": "Stack the cubes with two robots.",
}
QWEN_CACHE_FILES = {
    "LongPipelineDelivery-rf": "0dafeab0c51dec7e7aafa3012810b07d2184e5e4dacab7e29d0b51e42cc3c2a7.qwen3_flux2_len512.pt",
    "StackCube-rf": "d5ec92b7176b5a2930eef72fe1935d4f5827869aa345765c75adfef7efb48534.qwen3_flux2_len512.pt",
    "StrikeCube-rf": "475be1d3a0ce20e11fc57da16a04dd0d1af394187774e2e3bef6a79747d55652.qwen3_flux2_len512.pt",
    "TakePhoto-rf": "38677553da38890d4f02f229ce5f2da4392bb8df6c862c7404d6cfc12e047268.qwen3_flux2_len512.pt",
    "PlaceFood-rf": "add229e33d7067de82a044aa86c2e3c0321958213c4a1476e5ec6687b5235d7d.qwen3_flux2_len512.pt",
    "PickMeat-rf": "1269964016a34c228fab3fbc675e7741a8456cfb991b2ccc812e75a8e47ef2f3.qwen3_flux2_len512.pt",
    "PassShoe-rf": "5b6966f47c44bc6cf2c90879527bfef824a53af406ecc2c2aab2604b80fad575.qwen3_flux2_len512.pt",
    "CameraAlignment-rf": "5442aeecca69454f31b3f7c8256edd87558dbdc3a29b6778ce477b12da496575.qwen3_flux2_len512.pt",
    "LiftBarrier-rf": "051d350fa53ef7fa023738e4021955a187de61a17f693c68900f886512141651.qwen3_flux2_len512.pt",
    "ThreeRobotsStackCube-rf": "41184d6b2dadec5002ca4f8ba0f7284c5928de37dc991f42089f04c8a068ac75.qwen3_flux2_len512.pt",
    "TwoRobotsStackCube-rf": "432f0cf7455a7f553922ea1ea20b5de92b7816fe9314398cf7654f37aec74f77.qwen3_flux2_len512.pt",
}
_AGENT_NAME = re.compile(r"^panda-(\d+)$")


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _file_record(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def ordered_agent_names(observation: Mapping[str, Any]) -> tuple[str, ...]:
    agents = observation.get("agent")
    if not isinstance(agents, Mapping) or not agents:
        raise KeyError("RoboFactory observation lacks a non-empty agent mapping")
    indexed: list[tuple[int, str]] = []
    for raw_name in agents:
        name = str(raw_name)
        match = _AGENT_NAME.fullmatch(name)
        if match is None:
            raise ValueError(f"Unexpected RoboFactory agent name: {name!r}")
        indexed.append((int(match.group(1)), name))
    indexed.sort()
    ids = [index for index, _ in indexed]
    if ids != list(range(len(ids))):
        raise ValueError(f"RoboFactory agent ids must be contiguous, got {ids}")
    return tuple(name for _, name in indexed)


def _cpu_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    return torch.as_tensor(np.asarray(value))


def _flat_state(value: Any, *, label: str) -> torch.Tensor:
    tensor = _cpu_tensor(value)
    while tensor.ndim > 1 and tensor.shape[0] == 1:
        tensor = tensor[0]
    tensor = tensor.reshape(-1).to(torch.float32)
    if tensor.shape != (9,):
        raise ValueError(f"{label} must contain 9 values, got {tuple(tensor.shape)}")
    if not bool(torch.isfinite(tensor).all().item()):
        raise FloatingPointError(f"{label} contains non-finite values")
    return tensor


def raw_agent_state(
    observation: Mapping[str, Any], agent_names: Sequence[str]
) -> torch.Tensor:
    agents = observation["agent"]
    rows = []
    for name in agent_names:
        record = agents[name]
        rows.append(
            torch.cat(
                (
                    _flat_state(record["qpos"], label=f"{name}.qpos"),
                    _flat_state(record["qvel"], label=f"{name}.qvel"),
                )
            )
        )
    state = torch.stack(rows)
    if tuple(state.shape) != (len(agent_names), 18):
        raise ValueError(f"Unexpected multi-agent state shape: {tuple(state.shape)}")
    return state


def raw_agent_qpos(
    observation: Mapping[str, Any], agent_names: Sequence[str]
) -> dict[str, np.ndarray]:
    """Return finite, unbatched 9D qpos vectors in evaluator agent order."""

    agents = observation["agent"]
    return {
        name: np.ascontiguousarray(
            _flat_state(agents[name]["qpos"], label=f"{name}.qpos").numpy(),
            dtype=np.float32,
        )
        for name in agent_names
    }


def global_rgb_tensor(observation: Mapping[str, Any]) -> torch.Tensor:
    sensor_data = observation["sensor_data"]
    if "head_camera_global" in sensor_data:
        camera = sensor_data["head_camera_global"]
    elif "head_camera" in sensor_data:
        # RoboFactory's N=1 table tasks use ``head_camera`` while the N=2/3/4
        # tasks and the variable-cardinality training loader use
        # ``head_camera_global``.  Both are the task-level 240x320 policy view.
        camera = sensor_data["head_camera"]
    else:
        raise KeyError(
            "RoboFactory observation has neither head_camera_global nor head_camera"
        )
    value = camera["rgb"]
    image = _cpu_tensor(value)
    while image.ndim > 3 and image.shape[0] == 1:
        image = image[0]
    if image.ndim != 3:
        raise ValueError(f"Global RGB must be 3D, got {tuple(image.shape)}")
    if tuple(image.shape[:2]) == (240, 320) and image.shape[-1] >= 3:
        image = image[..., :3].permute(2, 0, 1)
    elif image.shape[0] >= 3 and tuple(image.shape[1:]) == (240, 320):
        image = image[:3]
    else:
        raise ValueError(f"Global RGB must be HWC/CHW at 240x320, got {tuple(image.shape)}")
    image = image.to(torch.float32)
    if not bool(torch.isfinite(image).all().item()):
        raise FloatingPointError("Global RGB contains non-finite values")
    minimum = float(image.min().item())
    maximum = float(image.max().item())
    if minimum >= -1e-6 and maximum <= 1.0 + 1e-6:
        image = image * 2.0 - 1.0
    elif minimum >= -1e-6 and maximum <= 255.0 + 1e-6:
        image = image * (2.0 / 255.0) - 1.0
    else:
        raise ValueError(f"Global RGB range is invalid: [{minimum}, {maximum}]")
    return image.unsqueeze(0).contiguous()


def _load_qwen_context(cache_dir: Path, task_name: str) -> tuple[Path, torch.Tensor, torch.Tensor]:
    cache_path = (cache_dir / QWEN_CACHE_FILES[task_name]).resolve(strict=True)
    payload = torch.load(cache_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise TypeError(f"Qwen context cache is not a mapping: {cache_path}")
    expected_keys = {"text_hidden_states", "text_attention_mask"}
    if set(payload) != expected_keys:
        raise KeyError(
            f"Qwen context cache keys must be {sorted(expected_keys)}, got {sorted(payload)}"
        )
    context = torch.as_tensor(payload["text_hidden_states"]).detach().cpu()
    mask = torch.as_tensor(payload["text_attention_mask"]).detach().cpu()
    if tuple(context.shape) != (512, 7680):
        raise ValueError(f"Qwen context must be [512,7680], got {tuple(context.shape)}")
    if tuple(mask.shape) != (512,):
        raise ValueError(f"Qwen context mask must be [512], got {tuple(mask.shape)}")
    if not bool(torch.isfinite(context).all().item()):
        raise FloatingPointError(f"Qwen context contains non-finite values: {cache_path}")
    return cache_path, context.contiguous(), mask.to(dtype=torch.bool).contiguous()


class ImageWAMRoboFactoryPolicy:
    """Expose ImageWAM actions as chronological RoboFactory flat chunks."""

    def __init__(
        self,
        *,
        checkpoint_path: str | Path,
        stats_path: str | Path,
        flux2_src_path: str | Path,
        flux2_model_path: str | Path,
        ae_model_path: str | Path,
        qwen_cache_dir: str | Path,
        policy_lightning_repo: str | Path,
        noposplat_checkpoint: str | Path,
        task_name: str,
        policy_lightning_commit: str = POLICY_LIGHTNING_COMMIT,
        noposplat_checkpoint_bytes: int = NOPOSPLAT_CHECKPOINT_BYTES,
        device: str = "cuda:0",
        model_dtype: torch.dtype = torch.bfloat16,
        action_horizon: int = 32,
        num_inference_steps: int = 20,
        sigma_shift: float | None = None,
        seed: int = 10000,
        rand_device: str = "cpu",
        gaussian_input_scale: float = 1.0,
        project_root: str | Path = PROJECT_ROOT,
    ) -> None:
        if task_name not in TASK_INSTRUCTIONS:
            raise KeyError(f"Unsupported task {task_name!r}")
        self.project_root = Path(project_root).expanduser().resolve(strict=True)
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve(strict=True)
        self.stats_path = Path(stats_path).expanduser().resolve(strict=True)
        self.flux2_src_path = Path(flux2_src_path).expanduser().resolve(strict=True)
        self.flux2_model_path = Path(flux2_model_path).expanduser().resolve(strict=True)
        self.ae_model_path = Path(ae_model_path).expanduser().resolve(strict=True)
        self.qwen_cache_dir = Path(qwen_cache_dir).expanduser().resolve(strict=True)
        self.task_name = task_name
        self.device = torch.device(device)
        self.action_horizon = int(action_horizon)
        self.num_inference_steps = int(num_inference_steps)
        self.sigma_shift = None if sigma_shift is None else float(sigma_shift)
        self.seed = int(seed)
        self.rand_device = str(rand_device)
        self.gaussian_input_scale = float(gaussian_input_scale)
        if self.action_horizon < 1 or self.num_inference_steps < 1:
            raise ValueError("action_horizon and num_inference_steps must be positive")
        if not math.isfinite(self.gaussian_input_scale) or self.gaussian_input_scale < 0:
            raise ValueError("gaussian_input_scale must be finite and non-negative")
        self.qwen_cache_path, self.text_context, self.text_context_mask = (
            _load_qwen_context(self.qwen_cache_dir, self.task_name)
        )
        # Load the external teacher before ImageWAM/Flux imports can populate an
        # unrelated top-level ``model`` module.  Each policy worker owns one
        # teacher and uses it at every closed-loop query.
        self.gaussian_teacher = OnlineNoPoSplatTeacher(
            repo_path=policy_lightning_repo,
            checkpoint_path=noposplat_checkpoint,
            expected_commit=policy_lightning_commit,
            expected_checkpoint_bytes=noposplat_checkpoint_bytes,
            device=self.device,
        )

        if str(self.flux2_src_path) not in os.sys.path:
            os.sys.path.insert(0, str(self.flux2_src_path))
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
        with initialize_config_dir(
            version_base="1.3", config_dir=str(self.project_root / "configs")
        ):
            cfg = compose(
                config_name=CONFIG_NAME,
                overrides=[f"task={TASK_CONFIG_NAME}"],
            )
        cfg.model.flux2_src_path = str(self.flux2_src_path)
        cfg.model.flux2_model_path = str(self.flux2_model_path)
        cfg.model.ae_model_path = str(self.ae_model_path)
        cfg.model.load_text_encoder = False

        self.model = instantiate(
            cfg.model,
            model_dtype=model_dtype,
            device=str(self.device),
        )
        self.model.load_checkpoint(
            str(self.checkpoint_path),
            strict_model=True,
        )
        self.checkpoint_load_report = dict(self.model.last_checkpoint_load_report)
        self.model = self.model.to(self.device).eval()
        self.processor = instantiate(cfg.data.val.processor).eval()
        # Keep the ImageWAM training package optional for Cosmos3 evaluation.
        # Cosmos3 reuses the observation helpers in this module, but performs
        # normalization in its action server and must remain importable in the
        # lightweight RoboFactory simulator environment.
        from imagewam.datasets.lerobot.utils.normalizer import (
            load_dataset_stats_from_json,
        )

        stats = load_dataset_stats_from_json(str(self.stats_path))
        self.processor.set_normalizer_from_stats(stats)
        self._observation: Mapping[str, Any] | None = None
        self._episode_seed = self.seed
        self._query_index = 0

    def start_episode(self, policy_seed: int | None = None) -> None:
        self._observation = None
        self._episode_seed = self.seed if policy_seed is None else int(policy_seed)
        self._query_index = 0

    def update_obs(
        self,
        observation: Mapping[str, Any],
        env_state: Mapping[str, Any] | None = None,
    ) -> None:
        del env_state
        names = ordered_agent_names(observation)
        if len(names) not in (1, 2, 3, 4):
            raise ValueError(f"Checkpoint supports N=1/2/3/4, got N={len(names)}")
        global_rgb_tensor(observation)
        raw_agent_state(observation, names)
        self._observation = observation

    def _normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        state_meta = self.processor.shape_meta["state"]
        if len(state_meta) != 1:
            raise ValueError("Expected one merged state key")
        key = state_meta[0]["key"]
        batch = {"state": {key: state.unsqueeze(0)}}
        batch = self.processor.action_state_transform(batch)
        batch = self.processor.normalizer.forward(batch)
        normalized = batch["state"][key][0]
        if not bool(torch.isfinite(normalized).all().item()):
            raise FloatingPointError("Normalized state contains non-finite values")
        return normalized

    def _action_views(self, action: torch.Tensor) -> dict[str, np.ndarray]:
        if action.ndim != 3 or action.shape[-1] != 8:
            raise ValueError(f"ImageWAM action must be [N,H,8], got {tuple(action.shape)}")
        agent_count, horizon, _ = action.shape
        action_meta = self.processor.shape_meta["action"]
        if len(action_meta) != 1:
            raise ValueError("Expected one merged action key")
        key = action_meta[0]["key"]
        normalizer = self.processor.normalizer.normalizers["action"][key]
        flat = action.reshape(1, agent_count * horizon, 8).to(
            device="cpu", dtype=torch.float32
        )
        normalized = flat.reshape(agent_count, horizon, 8)
        denormalized = normalizer.backward(flat).reshape(agent_count, horizon, 8)
        if not bool(torch.isfinite(normalized).all().item()):
            raise FloatingPointError("Normalized model action contains non-finite values")
        if not bool(torch.isfinite(denormalized).all().item()):
            raise FloatingPointError("Denormalized action contains non-finite values")
        normalized_chronological = normalized.permute(1, 0, 2).reshape(
            horizon, agent_count * 8
        )
        denormalized_chronological = denormalized.permute(1, 0, 2).reshape(
            horizon, agent_count * 8
        )
        return {
            "normalized": np.ascontiguousarray(
                normalized_chronological.numpy(), dtype=np.float32
            ),
            "denormalized": np.ascontiguousarray(
                denormalized_chronological.numpy(), dtype=np.float32
            ),
        }

    def _denormalize_action(self, action: torch.Tensor) -> np.ndarray:
        """Backward-compatible denormalized-only view used by older callers."""

        return self._action_views(action)["denormalized"]

    @torch.inference_mode()
    def get_action_with_trace(self) -> dict[str, Any]:
        if self._observation is None:
            raise RuntimeError("No observation is recorded; call update_obs first")
        names = ordered_agent_names(self._observation)
        state = self._normalize_state(raw_agent_state(self._observation, names))
        image = global_rgb_tensor(self._observation).to(
            device=self.device, dtype=self.model.torch_dtype
        )
        agent_gaussian = encode_compact_agent_gaussian(
            self.gaussian_teacher, self._observation, names
        )
        if self.gaussian_input_scale != 1.0:
            agent_gaussian = agent_gaussian.mul(self.gaussian_input_scale)
        query_seed = int(self._episode_seed) + int(self._query_index)
        prediction = self.model.infer_action(
            prompt=None,
            context=self.text_context,
            context_mask=self.text_context_mask,
            input_image=image,
            action_horizon=len(names) * self.action_horizon,
            agent_count=len(names),
            action_horizon_per_agent=self.action_horizon,
            agent_state=state,
            agent_meta=torch.empty((len(names), 0), dtype=torch.float32),
            agent_gaussian=agent_gaussian,
            num_inference_steps=self.num_inference_steps,
            sigma_shift=self.sigma_shift,
            seed=query_seed,
            rand_device=self.rand_device,
        )
        action = torch.as_tensor(prediction["action"], dtype=torch.float32)
        expected = (len(names), self.action_horizon, 8)
        if tuple(action.shape) != expected:
            raise ValueError(f"ImageWAM returned {tuple(action.shape)}, expected {expected}")
        views = self._action_views(action)
        query_index = int(self._query_index)
        self._query_index += 1
        return {
            **views,
            "query_index": query_index,
            "query_seed": query_seed,
        }

    @torch.inference_mode()
    def get_action(self) -> np.ndarray:
        """Return the historical denormalized action interface unchanged."""

        return self.get_action_with_trace()["denormalized"]

    def provenance(self) -> dict[str, Any]:
        return {
            "adapter": "ImageWAMRoboFactoryPolicy",
            "task_name": self.task_name,
            "checkpoint": _file_record(self.checkpoint_path),
            "dataset_stats": _file_record(self.stats_path),
            "flux2_model": _file_record(self.flux2_model_path),
            "ae_model": _file_record(self.ae_model_path),
            "qwen_context_cache": _file_record(self.qwen_cache_path),
            "qwen_context_shape": list(self.text_context.shape),
            "qwen_context_mask_shape": list(self.text_context_mask.shape),
            "imagewam_commit": _git(self.project_root, "rev-parse", "HEAD"),
            "flux2_commit": _git(self.flux2_src_path, "rev-parse", "HEAD"),
            "gaussian_teacher": self.gaussian_teacher.provenance(),
            "gaussian_pairing": "robofactory_reference_agent_or_self_unify_v1",
            "gaussian_selected_view": "agent_element_1_of_[global,agent_i]",
            "gaussian_compaction": MOMENT_MATCH_METHOD,
            "gaussian_shape": ["N", 13, 28, 40],
            "gaussian_input_scale": self.gaussian_input_scale,
            "checkpoint_load_report": self.checkpoint_load_report,
            "spatial_gaussian_output_gates": [
                {
                    "name": name,
                    "raw": float(parameter.detach().float().cpu().item()),
                    "tanh": float(parameter.detach().float().tanh().cpu().item()),
                }
                for name, parameter in self.model.mot.named_parameters()
                if name.endswith("gaussian_conditioner.output_gate")
            ],
            "action_horizon_per_agent": self.action_horizon,
            "num_inference_steps": self.num_inference_steps,
            "seed_schedule": "episode_policy_seed_plus_query_index_v1",
            "action_interface": "raw_[H,Nx8]_agent_major_per_timestep",
            "action_trace_interface": (
                "normalized_and_denormalized_[H,Nx8]_agent_major_per_timestep"
            ),
            "observation_interface": "global_rgb_240x320_plus_per_agent_qpos9_qvel9",
        }
