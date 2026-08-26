# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""CPU contract tests for the CWAM/Cosmos3 RoboCasa365 policy adapter."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import sys
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.distributed.checkpoint as dcp
from omegaconf import OmegaConf
from PIL import Image

_DATASETS_PACKAGE = "cosmos_framework.data.generator.action.datasets"
_previous_datasets_package = sys.modules.get(_DATASETS_PACKAGE)
_isolated_datasets_package = ModuleType(_DATASETS_PACKAGE)
_isolated_datasets_package.__path__ = [str(Path(__file__).parents[1] / "data" / "generator" / "action" / "datasets")]
sys.modules[_DATASETS_PACKAGE] = _isolated_datasets_package
try:
    # This service test imports only RoboCasa365SegmentDataset.  Bypass the
    # datasets package's unrelated eager LeRobot imports so CPU contract tests
    # remain collectable in the repository's minimal local test environment.
    with patch("cosmos_framework.inference.common.init._init_script", lambda **kwargs: None):
        for module_name in (
            "cosmos_framework.scripts.action_policy_server_robocasa365",
            "cosmos_framework.scripts.action_policy_server_robolab",
            "cosmos_framework.scripts.action_policy_server_libero",
        ):
            if module_name in sys.modules:
                del sys.modules[module_name]
        from cosmos_framework.model.generator.tokenizers import wan2pt2_vae_4x16x16 as wan  # noqa: E402
        from cosmos_framework.scripts import action_policy_server_robocasa365 as server  # noqa: E402
        from cosmos_framework.utils.lazy_config import instantiate  # noqa: E402
finally:
    if _previous_datasets_package is None:
        del sys.modules[_DATASETS_PACKAGE]
    else:
        sys.modules[_DATASETS_PACKAGE] = _previous_datasets_package

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def _observation() -> dict:
    return {
        "observation/agentview_left": np.full((256, 256, 3), 11, dtype=np.uint8),
        "observation/eye_in_hand": np.full((256, 256, 3), 22, dtype=np.uint8),
        "observation/agentview_right": np.full((256, 256, 3), 33, dtype=np.uint8),
        "observation/state": np.arange(16, dtype=np.float64),
        "prompt": "open the drawer",
    }


def _stats_file(path: Path) -> Path:
    stats = {
        "global_raw": {
            "mean": [0.0] * 10,
            "std": [1.0] * 10,
            "min": [-4.0] * 10,
            "max": [4.0] * 10,
            "q01": [-2.0] * 10,
            "q99": [2.0] * 10,
        }
    }
    path.write_text(json.dumps(stats), encoding="utf-8")
    return path


def test_websocket_server_receipt_binds_pinned_demian_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "websocket_policy_server.py"
    source_path.write_bytes(b"class WebsocketPolicyServer: pass\n")

    class FakeWebsocketServer:
        pass

    FakeWebsocketServer.__module__ = "openpi.serving.websocket_policy_server"
    FakeWebsocketServer.__qualname__ = "WebsocketPolicyServer"
    monkeypatch.setattr(
        server.inspect,
        "getsourcefile",
        lambda _server_cls: str(source_path),
    )

    assert server._websocket_server_receipt(FakeWebsocketServer) == {
        "module": "openpi.serving.websocket_policy_server",
        "qualname": "WebsocketPolicyServer",
        "source_path": str(source_path),
        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
    }


def test_server_args_disable_generative_media_guardrails() -> None:
    args = server.RoboCasa365ServerArgs.model_validate(
        {
            "checkpoint": {"checkpoint_path": "/checkpoint/model"},
            "launch_nonce": "a" * 64,
            "expected_ema_key_count": 809,
        }
    )

    overrides = args.build_setup_overrides()

    assert overrides.guardrails is False
    assert overrides.parallelism_preset == "throughput"
    assert overrides.dp_shard_size == 1
    assert overrides.dp_replicate_size == 1
    assert overrides.cp_size == 1
    assert overrides.cfgp_size == 1
    assert overrides.experiment_overrides[-2] == (
        "model.config.tokenizer._target_="
        "cosmos_framework.model.generator.tokenizers.wan2pt2_vae_4x16x16."
        "Wan2pt2VAEInterface"
    )
    assert overrides.experiment_overrides[-1] == "~model.config.tokenizer.latent_ch"
    assert not any("vae_path=" in value for value in overrides.experiment_overrides)
    assert not any("pretrained_model_name=" in value for value in overrides.experiment_overrides)


def test_server_args_apply_canonical_portable_model_asset_overrides(tmp_path: Path) -> None:
    vae_path = tmp_path / "Wan2.2_VAE.pth"
    vae_path.write_bytes(b"vae")
    qwen_path = tmp_path / "Qwen3-VL-8B"
    qwen_path.mkdir()
    args = server.RoboCasa365ServerArgs.model_validate(
        {
            "checkpoint": {"checkpoint_path": "/checkpoint/model"},
            "launch_nonce": "a" * 64,
            "expected_ema_key_count": 809,
            "wan2pt2_vae_path": vae_path,
            "qwen_tokenizer_path": qwen_path,
        }
    )

    overrides = args.build_setup_overrides()

    assert f"model.config.tokenizer.vae_path={vae_path}" in overrides.experiment_overrides
    assert (f"model.config.vlm_config.tokenizer.pretrained_model_name={qwen_path}") in overrides.experiment_overrides
    resolved = OmegaConf.from_dotlist(overrides.experiment_overrides)
    assert resolved.model.config.tokenizer.vae_path == str(vae_path)
    assert resolved.model.config.vlm_config.tokenizer.pretrained_model_name == str(qwen_path)
    assert args.resolve_external_model_paths() == (vae_path, qwen_path)


@pytest.mark.parametrize("asset", ["vae", "qwen"])
def test_server_args_reject_symlinked_portable_model_assets(
    tmp_path: Path,
    asset: str,
) -> None:
    real_vae = tmp_path / "real-vae.pth"
    real_vae.write_bytes(b"vae")
    linked_vae = tmp_path / "linked-vae.pth"
    linked_vae.symlink_to(real_vae)
    real_qwen = tmp_path / "real-qwen"
    real_qwen.mkdir()
    linked_qwen = tmp_path / "linked-qwen"
    linked_qwen.symlink_to(real_qwen, target_is_directory=True)
    values: dict[str, object] = {
        "checkpoint": {"checkpoint_path": "/checkpoint/model"},
        "launch_nonce": "a" * 64,
        "expected_ema_key_count": 809,
    }
    values["wan2pt2_vae_path" if asset == "vae" else "qwen_tokenizer_path"] = (
        linked_vae if asset == "vae" else linked_qwen
    )
    args = server.RoboCasa365ServerArgs.model_validate(values)

    with pytest.raises(ValueError, match="canonical and non-symlink"):
        args.build_setup_overrides()


def test_server_args_reject_relative_portable_model_asset_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    (tmp_path / "vae.pth").write_bytes(b"vae")
    monkeypatch.chdir(tmp_path)
    args = server.RoboCasa365ServerArgs.model_validate(
        {
            "checkpoint": {"checkpoint_path": "/checkpoint/model"},
            "launch_nonce": "a" * 64,
            "expected_ema_key_count": 809,
            "wan2pt2_vae_path": Path("vae.pth"),
        }
    )

    with pytest.raises(ValueError, match="absolute and canonical"):
        args.build_setup_overrides()


def test_policy_metadata_records_actual_portable_model_asset_paths(tmp_path: Path) -> None:
    vae_path = tmp_path / "Wan2.2_VAE.pth"
    vae_path.write_bytes(b"vae")
    qwen_path = tmp_path / "Qwen3-VL-8B"
    qwen_path.mkdir()
    stats_path = _stats_file(tmp_path / "stats.json")
    checkpoint_path = tmp_path / "checkpoint"
    checkpoint_path.mkdir()
    metadata_path = checkpoint_path / ".metadata"
    metadata_path.write_bytes(b"metadata")
    config_path = tmp_path / "config.yaml"
    config_path.write_text("model: test\n", encoding="utf-8")
    service = object.__new__(server.RoboCasa365PolicyService)
    service.cfg = SimpleNamespace(
        action_stats_path=stats_path,
        checkpoint_dir="/checkpoint/model",
        fps=20,
        action_chunk_size=16,
        seed=42,
        guidance=1.0,
        num_steps=30,
    )
    service.setup_args = SimpleNamespace(
        sampler="unipc",
        use_ema_weights=True,
        guardrails=False,
        dp_shard_size=4,
        dp_replicate_size=1,
        cp_size=1,
        cfgp_size=1,
    )
    service.launch_nonce = "a" * 64
    service.wan2pt2_vae_path = vae_path
    service.qwen_tokenizer_path = qwen_path
    service.distributed_world_size = 4
    service.checkpoint_path = checkpoint_path
    service.training_config_path = config_path
    service.action_stats_path = stats_path
    service._checkpoint_weight_receipt = {
        "checkpoint_metadata_sha256": server._sha256_regular_file(metadata_path),
        "selected_weight_tree": "net_ema",
    }

    metadata = service.metadata()

    assert metadata["wan2pt2_vae_path"] == str(vae_path)
    assert metadata["qwen_tokenizer_path"] == str(qwen_path)
    assert metadata["checkpoint"] == str(checkpoint_path)
    assert metadata["checkpoint_weight_tree"] == "net_ema"
    assert metadata["config_file"] == str(config_path)
    assert metadata["config_file_sha256"] == server._sha256_regular_file(config_path)
    assert metadata["action_stats_path"] == str(stats_path)
    assert metadata["action_stats_sha256"] == server._sha256_regular_file(stats_path)


@pytest.mark.parametrize("key_count", [809, 822])
def test_dcp_weight_receipt_requires_exact_serving_only_ema_bf16_tree(
    tmp_path: Path,
    key_count: int,
) -> None:
    checkpoint = tmp_path / "model"
    state = {f"net_ema.layer_{index}.weight": torch.tensor([index], dtype=torch.bfloat16) for index in range(key_count)}
    dcp.save(state, checkpoint_id=checkpoint)

    receipt = server._dcp_selected_weight_receipt(
        checkpoint,
        use_ema_weights=True,
        require_serving_only_ema_bf16=True,
        expected_ema_key_count=key_count,
    )

    assert receipt["selected_weight_tree"] == "net_ema"
    assert receipt["selected_key_count"] == key_count
    assert receipt["selected_source_dtype_census"] == {"torch.bfloat16": key_count}
    assert receipt["regular_key_count"] == 0
    assert receipt["ema_key_count"] == key_count
    assert receipt["other_key_count"] == 0
    assert receipt["serving_only_selected_tree"] is True
    assert receipt["load_planner"] == "CustomLoadPlanner(load_ema_to_reg=True)"


def test_dcp_weight_receipt_rejects_manifest_key_count_mismatch(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model"
    state = {f"net_ema.layer_{index}.weight": torch.tensor([index], dtype=torch.bfloat16) for index in range(3)}
    dcp.save(state, checkpoint_id=checkpoint)

    with pytest.raises(RuntimeError, match="manifest-bound number"):
        server._dcp_selected_weight_receipt(
            checkpoint,
            use_ema_weights=True,
            require_serving_only_ema_bf16=True,
            expected_ema_key_count=4,
        )


def test_frozen_cached_config_resolves_to_online_wan_for_serving(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = server.RoboCasa365ServerArgs.model_validate(
        {
            "checkpoint": {"checkpoint_path": "/checkpoint/model"},
            "launch_nonce": "a" * 64,
            "expected_ema_key_count": 809,
        }
    )
    frozen = OmegaConf.create(
        {
            "model": {
                "config": {
                    "tokenizer": {
                        "_target_": (
                            "cosmos_framework.model.generator.tokenizers.precomputed_wan2pt2."
                            "PrecomputedWan2pt2LatentTokenizer"
                        ),
                        "vae_path": "/tmp/Wan2.2_VAE.pth",
                        "latent_ch": 48,
                        "spatial_compression_factor": 16,
                        "temporal_compression_factor": 4,
                        "causal": True,
                    }
                }
            }
        }
    )
    setup_overrides = args.build_setup_overrides().experiment_overrides
    assert "~model.config.tokenizer.latent_ch" in setup_overrides
    del frozen.model.config.tokenizer.latent_ch
    merged = OmegaConf.merge(
        frozen,
        OmegaConf.from_dotlist([value for value in setup_overrides if not value.startswith("~")]),
    )
    monkeypatch.setattr(
        wan,
        "WanVAE",
        lambda **_kwargs: SimpleNamespace(dtype=torch.bfloat16),
    )

    tokenizer = instantiate(merged.model.config.tokenizer)

    assert type(tokenizer) is wan.Wan2pt2VAEInterface
    assert "latent_ch" not in merged.model.config.tokenizer


def test_server_args_pin_pure_fsdp_to_torchrun_world_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server, "_declared_world_size", lambda: 4)
    args = server.RoboCasa365ServerArgs.model_validate(
        {
            "checkpoint": {"checkpoint_path": "/checkpoint/model"},
            "launch_nonce": "a" * 64,
            "expected_ema_key_count": 809,
        }
    )

    overrides = args.build_setup_overrides()

    assert overrides.parallelism_preset == "throughput"
    assert overrides.dp_shard_size == 4
    assert overrides.dp_replicate_size == 1
    assert overrides.cp_size == 1
    assert overrides.cfgp_size == 1


def test_validate_fsdp_topology_rejects_context_or_replication_drift() -> None:
    with pytest.raises(RuntimeError, match="pure FSDP topology"):
        server._validate_fsdp_topology(
            SimpleNamespace(
                dp_shard_size=2,
                dp_replicate_size=2,
                cp_size=1,
                cfgp_size=1,
            ),
            world_size=4,
        )

    with pytest.raises(RuntimeError, match="pure FSDP topology"):
        server._validate_fsdp_topology(
            SimpleNamespace(
                dp_shard_size=4,
                dp_replicate_size=1,
                cp_size=2,
                cfgp_size=1,
            ),
            world_size=4,
        )


def test_validate_fsdp_topology_rejects_non_four_rank_distributed_service() -> None:
    with pytest.raises(RuntimeError, match="four-rank FSDP"):
        server._validate_fsdp_topology(
            SimpleNamespace(
                dp_shard_size=2,
                dp_replicate_size=1,
                cp_size=1,
                cfgp_size=1,
            ),
            world_size=2,
        )


def test_extract_training_observation_uses_left_then_wrist_only() -> None:
    image, prompt = server._extract_training_observation(
        _observation(),
        camera_height=256,
        camera_width=256,
    )

    assert image.shape == (256, 512, 3)
    assert image.dtype == np.uint8
    assert np.all(image[:, :256] == 11)
    assert np.all(image[:, 256:] == 22)
    assert not np.any(image == 33)
    assert prompt == "open the drawer"


def test_extract_training_observation_rejects_noncausal_reasoning() -> None:
    obs = _observation()
    obs["reasoning_text"] = "oracle future action"

    with pytest.raises(ValueError, match="not part of the causal"):
        server._extract_training_observation(obs, camera_height=256, camera_width=256)


def test_extract_training_observation_rejects_dtype_and_state_drift() -> None:
    obs = _observation()
    obs["observation/agentview_left"] = obs["observation/agentview_left"].astype(np.float32)
    with pytest.raises(TypeError, match="dtype uint8"):
        server._extract_training_observation(obs, camera_height=256, camera_width=256)

    obs = _observation()
    obs["observation/state"] = np.zeros(15, dtype=np.float64)
    with pytest.raises(ValueError, match="shape \\(16,\\)"):
        server._extract_training_observation(obs, camera_height=256, camera_width=256)


def test_quantile_rot_normalizer_round_trip_uses_training_resolver(tmp_path: Path) -> None:
    normalizer = server._load_required_quantile_rot_normalizer(_stats_file(tmp_path / "stats.json"))
    raw = torch.tensor(
        [[-1.5, -1.0, -0.5, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.75]],
        dtype=torch.float32,
    )

    recovered = normalizer.denormalize_action(normalizer.normalize_action(raw))

    torch.testing.assert_close(recovered, raw)


def test_rot6d10_to_axisangle7_preserves_translation_and_gripper() -> None:
    raw = np.array(
        [
            [0.1, -0.2, 0.3, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0],
            [-0.4, 0.5, -0.6, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.25],
        ],
        dtype=np.float32,
    )

    env_action = server._rot6d10_to_axisangle7(raw)

    np.testing.assert_allclose(env_action[:, :3], raw[:, :3])
    np.testing.assert_allclose(env_action[:, 6], raw[:, 9])
    np.testing.assert_allclose(env_action[0, 3:6], np.zeros(3), atol=1e-6)
    np.testing.assert_allclose(env_action[1, 3:6], np.array([np.pi / 2, 0.0, 0.0]), atol=1e-6)


def test_rot6d10_to_axisangle7_rejects_degenerate_rotation() -> None:
    raw = np.zeros((1, 10), dtype=np.float32)

    with pytest.raises(ValueError, match="degenerate"):
        server._rot6d10_to_axisangle7(raw)


def test_json_prompt_includes_exact_training_camera_description() -> None:
    service = object.__new__(server.RoboCasa365PolicyService)
    service.cfg = SimpleNamespace(fps=20, action_chunk_size=16, max_action_dim=64)
    service._prompt_json_formatter = server.ActionPromptJsonFormatter(
        caption_key="ai_caption",
        include_idle_frames=False,
    )
    video = torch.zeros((3, 17, 256, 512), dtype=torch.uint8)
    image_size = torch.tensor([256, 512, 256, 512], dtype=torch.float32)

    encoded = service._build_json_prompt("open the drawer", video=video, image_size=image_size)
    prompt = json.loads(encoded)

    framing = prompt["cinematography"]["framing"]
    assert server._TRAINING_CAMERA_DESCRIPTION in framing
    assert prompt["actions"][0]["description"] == "open the drawer."
    assert "idle_frame" not in prompt["actions"][0]
    assert prompt["fps"] == 20.0


def test_infer_returns_demian_actions_contract_and_png_camera_order() -> None:
    service = object.__new__(server.RoboCasa365PolicyService)
    service.camera_height = 256
    service.camera_width = 256
    service.domain_name = "robocasa365"
    service.distributed_rank = 0
    service.distributed_world_size = 1
    service.cfg = SimpleNamespace(action_chunk_size=16)
    identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    raw = np.array([[0.1, 0.2, 0.3, *identity, -0.75]] * 16, dtype=np.float32)
    captured: list[dict] = []

    def fake_predict(reqs: list[dict]) -> dict:
        captured.extend(reqs)
        return {"actions": [raw.tolist()]}

    service.predict_policy_batch = fake_predict
    response = service.infer(_observation())

    assert set(response) == {"actions"}
    assert response["actions"].shape == (16, 7)
    np.testing.assert_allclose(response["actions"][:, 6], -0.75)
    assert captured[0]["domain_name"] == "robocasa365"
    assert captured[0]["image_size"] == 256
    image_bytes = base64.b64decode(captured[0]["image"])
    image = np.asarray(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
    assert image.shape == (256, 512, 3)
    assert np.all(image[:, :256] == 11)
    assert np.all(image[:, 256:] == 22)


def _distributed_test_service(*, rank: int) -> server.RoboCasa365PolicyService:
    service = object.__new__(server.RoboCasa365PolicyService)
    service.camera_height = 256
    service.camera_width = 256
    service.domain_name = "robocasa365"
    service.distributed_rank = rank
    service.distributed_world_size = 4
    service._dispatch_lock = threading.Lock()
    service._next_distributed_sequence = 0
    service._shutdown_requested = False
    service.launch_nonce = "a" * 64
    service.cfg = SimpleNamespace(action_chunk_size=16)
    return service


def _identity_raw_action() -> np.ndarray:
    identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    return np.array([[0.1, 0.2, 0.3, *identity, -0.75]] * 16, dtype=np.float32)


def test_distributed_rank0_broadcasts_canonical_request_and_executes_same_model_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _distributed_test_service(rank=0)
    raw = _identity_raw_action()
    broadcasts: list[dict] = []
    predicted_requests: list[dict] = []
    prepared_requests: list[dict] = []
    stages: list[str] = []

    def broadcast(objects: list[object], *, src: int) -> None:
        assert src == 0
        broadcasts.append(objects[0])

    def all_gather(reports: list[object], local_report: dict) -> None:
        stages.append(local_report["stage"])
        for rank in range(4):
            reports[rank] = {**local_report, "rank": rank}

    def predict(reqs: list[dict]) -> dict:
        predicted_requests.extend(reqs)
        return {"actions": [raw.tolist()]}

    monkeypatch.setattr(server.torch.distributed, "broadcast_object_list", broadcast)
    monkeypatch.setattr(server.torch.distributed, "all_gather_object", all_gather)
    service._prep_policy_item = prepared_requests.append
    service.predict_policy_batch = predict

    response = service.infer(_observation())

    assert response["actions"].shape == (16, 7)
    assert len(broadcasts) == 1
    command = broadcasts[0]
    assert command["schema_version"] == 1
    assert command["sequence"] == 0
    assert command["op"] == "infer"
    assert set(command["payload"]) == {"image", "prompt", "domain_name", "image_size"}
    assert prepared_requests == [command["payload"]]
    assert predicted_requests == [command["payload"]]
    assert stages == ["prepare", "postprocess"]


def test_distributed_rank0_model_failure_exits_torchrun_worker_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _distributed_test_service(rank=0)
    canonical = {
        "image": "encoded",
        "prompt": "open the drawer",
        "domain_name": "robocasa365",
        "image_size": 256,
    }

    def all_gather(reports: list[object], local_report: dict) -> None:
        for rank in range(4):
            reports[rank] = {**local_report, "rank": rank}

    def fail(_requests: list[dict]) -> dict:
        raise RuntimeError("fatal FSDP failure")

    exit_codes: list[int] = []

    def fail_exit(code: int) -> None:
        exit_codes.append(code)
        raise SystemExit(code)

    monkeypatch.setattr(server.torch.distributed, "all_gather_object", all_gather)
    monkeypatch.setattr(server.os, "_exit", fail_exit)
    service._prep_policy_item = lambda _request: None
    service.predict_policy_batch = fail

    with pytest.raises(SystemExit, match="70"):
        service._execute_canonical_request(canonical)
    assert exit_codes == [70]


def test_distributed_worker_executes_infer_resets_and_discards_result() -> None:
    service = _distributed_test_service(rank=2)
    canonical = {
        "image": "encoded",
        "prompt": "open the drawer",
        "domain_name": "robocasa365",
        "image_size": 256,
    }
    commands = iter((("infer", canonical), ("reset", None), ("shutdown", None)))
    executed: list[dict] = []
    synchronized: list[str] = []

    service._receive_distributed_command = lambda: next(commands)

    def execute(request: dict) -> dict:
        executed.append(request)
        return {"actions": "discard me"}

    def synchronize(stage: str, operation: object) -> object:
        synchronized.append(stage)
        return operation()

    service._execute_canonical_request = execute
    service._run_synchronized_stage = synchronize

    service.run_distributed_worker()

    assert executed == [canonical]
    assert synchronized == ["reset", "shutdown"]


def test_rank0_nonce_bound_shutdown_control_acks_and_stops_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _distributed_test_service(rank=0)
    broadcasts: list[dict] = []
    stages: list[str] = []
    scheduled_targets: list[object] = []

    def broadcast(objects: list[object], *, src: int) -> None:
        assert src == 0
        broadcasts.append(objects[0])

    def all_gather(reports: list[object], local_report: dict) -> None:
        stages.append(local_report["stage"])
        for rank in range(4):
            reports[rank] = {**local_report, "rank": rank}

    class FakeThread:
        def __init__(self, *, target: object, name: str, daemon: bool) -> None:
            assert name == "cwam-rank0-clean-shutdown"
            assert daemon is True
            scheduled_targets.append(target)

        def start(self) -> None:
            return None

    monkeypatch.setattr(server.torch.distributed, "broadcast_object_list", broadcast)
    monkeypatch.setattr(server.torch.distributed, "all_gather_object", all_gather)
    monkeypatch.setattr(server.threading, "Thread", FakeThread)

    response = service.infer(
        {
            "__cwam_control__": "shutdown",
            "launch_nonce": "a" * 64,
        }
    )

    assert response == {
        "actions": [],
        "cwam_control": "shutdown_ack",
        "launch_nonce": "a" * 64,
    }
    assert broadcasts[0]["op"] == "shutdown"
    assert broadcasts[0]["payload"] is None
    assert stages == ["shutdown"]
    assert scheduled_targets == [service._exit_rank0_after_shutdown_ack]
    assert service._shutdown_requested is True


def test_single_rank_nonce_bound_shutdown_control_acks_without_broadcast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _distributed_test_service(rank=0)
    service.distributed_world_size = 1
    shutdown_worker_calls: list[bool] = []
    scheduled_targets: list[object] = []

    service._shutdown_distributed_workers_locked = lambda: shutdown_worker_calls.append(True)

    class FakeThread:
        def __init__(self, *, target: object, name: str, daemon: bool) -> None:
            assert name == "cwam-rank0-clean-shutdown"
            assert daemon is True
            scheduled_targets.append(target)

        def start(self) -> None:
            return None

    monkeypatch.setattr(server.threading, "Thread", FakeThread)

    response = service.infer(
        {
            "__cwam_control__": "shutdown",
            "launch_nonce": "a" * 64,
        }
    )

    assert response == {
        "actions": [],
        "cwam_control": "shutdown_ack",
        "launch_nonce": "a" * 64,
    }
    assert shutdown_worker_calls == []
    assert scheduled_targets == [service._exit_rank0_after_shutdown_ack]
    assert service._shutdown_requested is True


def test_distributed_worker_model_exception_fails_closed() -> None:
    service = _distributed_test_service(rank=3)
    canonical = {
        "image": "encoded",
        "prompt": "open the drawer",
        "domain_name": "robocasa365",
        "image_size": 256,
    }
    service._receive_distributed_command = lambda: ("infer", canonical)

    def fail(_request: dict) -> dict:
        raise RuntimeError("worker model failed")

    service._execute_canonical_request = fail

    with pytest.raises(RuntimeError, match="worker model failed"):
        service.run_distributed_worker()


def test_distributed_stage_propagates_any_rank_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _distributed_test_service(rank=0)

    def all_gather(reports: list[object], local_report: dict) -> None:
        for rank in range(4):
            reports[rank] = {
                **local_report,
                "rank": rank,
                "error": "RuntimeError: peer failed" if rank == 2 else None,
            }

    monkeypatch.setattr(server.torch.distributed, "all_gather_object", all_gather)

    with pytest.raises(RuntimeError, match="rank 2: RuntimeError: peer failed"):
        service._run_synchronized_stage("prepare", lambda: "ok")


def test_serve_nonzero_rank_never_constructs_websocket_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker_calls: list[bool] = []
    fake_service = SimpleNamespace(
        distributed_rank=2,
        distributed_world_size=4,
        run_distributed_worker=lambda: worker_calls.append(True),
    )
    monkeypatch.setattr(server, "RoboCasa365PolicyService", lambda _args: fake_service)

    def forbidden_loader() -> object:
        raise AssertionError("nonzero rank must not load or bind the WebSocket server")

    monkeypatch.setattr(server, "_load_openpi_websocket_policy_server", forbidden_loader)

    server.serve(SimpleNamespace(host="127.0.0.1", port=16000))

    assert worker_calls == [True]


def test_serve_rank0_binds_once_and_shutdowns_workers_after_normal_return(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_calls: list[bool] = []
    server_inits: list[dict] = []
    fake_service = SimpleNamespace(
        distributed_rank=0,
        distributed_world_size=4,
        metadata=lambda: {"distributed_world_size": 4},
        shutdown_distributed_workers=lambda: shutdown_calls.append(True),
    )
    monkeypatch.setattr(server, "RoboCasa365PolicyService", lambda _args: fake_service)

    class FakeWebsocketServer:
        def __init__(self, **kwargs: object) -> None:
            server_inits.append(kwargs)

        def serve_forever(self) -> None:
            return None

    monkeypatch.setattr(server, "_load_openpi_websocket_policy_server", lambda: FakeWebsocketServer)
    websocket_receipt = {
        "module": "openpi.serving.websocket_policy_server",
        "qualname": "WebsocketPolicyServer",
        "source_path": "/inputs/demian/websocket_policy_server.py",
        "source_sha256": "a" * 64,
    }
    monkeypatch.setattr(
        server,
        "_websocket_server_receipt",
        lambda _server_cls: websocket_receipt,
    )

    server.serve(SimpleNamespace(host="127.0.0.1", port=16000))

    assert len(server_inits) == 1
    assert server_inits[0]["policy"] is fake_service
    assert server_inits[0]["host"] == "127.0.0.1"
    assert server_inits[0]["port"] == 16000
    assert server_inits[0]["metadata"]["websocket_server"] == websocket_receipt
    assert shutdown_calls == [True]


def test_serve_rank0_exception_does_not_attempt_unsafe_collective_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_calls: list[bool] = []
    fake_service = SimpleNamespace(
        distributed_rank=0,
        distributed_world_size=4,
        metadata=lambda: {},
        shutdown_distributed_workers=lambda: shutdown_calls.append(True),
    )
    monkeypatch.setattr(server, "RoboCasa365PolicyService", lambda _args: fake_service)

    class FailingWebsocketServer:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def serve_forever(self) -> None:
            raise RuntimeError("port bind failed")

    monkeypatch.setattr(server, "_load_openpi_websocket_policy_server", lambda: FailingWebsocketServer)
    monkeypatch.setattr(
        server,
        "_websocket_server_receipt",
        lambda _server_cls: {
            "module": "openpi.serving.websocket_policy_server",
            "qualname": "WebsocketPolicyServer",
            "source_path": "/inputs/demian/websocket_policy_server.py",
            "source_sha256": "a" * 64,
        },
    )

    with pytest.raises(RuntimeError, match="port bind failed"):
        server.serve(SimpleNamespace(host="127.0.0.1", port=16000))

    assert shutdown_calls == []
