# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Serve Cosmos3 RoboFactory multi-robot actions over the local HTTP bridge.

The request carries the current global RGB observation, raw per-agent state and
online compact Gaussian condition.  Actions are generated in the training-time
agent-major order and returned in RoboFactory's chronological flat order.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import replace
from http.server import ThreadingHTTPServer
from typing import Any

import numpy as np
import torch

from cosmos_framework.data.generator.action.action_processing import (
    ActionProcessingRecord,
    make_batched_action_processing_fields,
)
from cosmos_framework.data.generator.action.domain_utils import get_domain_id
from cosmos_framework.data.generator.action.transforms import (
    build_sequence_plan_from_mode,
    find_closest_target_size,
    reflection_pad_to_target,
)
from cosmos_framework.inference.args import OmniSetupOverrides
from cosmos_framework.scripts.action_policy_server_libero import (
    ActionModelService,
    ActionServerArgs,
    _ActionHandler,
    _decode_base64_png_to_rgb_uint8,
    get_local_ip,
    tyro,
    tyro_cli,
)
from cosmos_framework.scripts.robofactory_action_bridge import (
    ACTION_HORIZON_PER_AGENT,
    chronological_views,
    decode_array,
)
from cosmos_framework.utils import log
from cosmos_framework.utils.generator.data_utils import get_vision_data_resolution


VIDEO_LENGTH = 17
STATE_DIM = 18
GAUSSIAN_SHAPE = (13, 28, 40)


class RoboFactoryActionServerArgs(ActionServerArgs):
    """RoboFactory serving overrides that are fixed by the evaluation contract."""

    def build_setup_overrides(self) -> OmniSetupOverrides:
        setup = super().build_setup_overrides()
        setup.guardrails = False
        # The multi-robot conditioner consumes per-request scalar metadata
        # (agent count and horizon).  Dynamo cannot safely guard those values
        # inside the fully compiled VFM forward, while eager inference handles
        # the exact same model and checkpoint without changing policy math.
        setup.use_torch_compile = False
        setup.use_cuda_graphs = False
        return setup


class RoboFactoryActionModelService(ActionModelService):
    """Cosmos3 action service with the exact variable-cardinality contract."""

    def __init__(self, args: RoboFactoryActionServerArgs) -> None:
        from cosmos_framework.data.generator.action.datasets.robofactory_multi_robot_dataset import (
            load_imagewam_zscore_normalizer,
        )

        if args.action_stats_path is None:
            raise ValueError("--action-stats-path is required for RoboFactory")
        if args.action_chunk_size not in (None, ACTION_HORIZON_PER_AGENT):
            raise ValueError("RoboFactory action horizon per agent is fixed at 32")
        if args.raw_action_dim not in (None, 8):
            raise ValueError("RoboFactory raw action width is fixed at 8")
        # The parent loader must not manually denormalize an already externalized
        # model output.  Keep its stats disabled and own the exact normalizer here.
        stats_path = args.action_stats_path
        parent_args = args.model_copy(
            update={
                "action_stats_path": None,
                "action_chunk_size": 32,
                "raw_action_dim": 8,
                "format_prompt_as_json": True,
                "http_400_on_error": True,
            }
        )
        super().__init__(parent_args)
        if self._prompt_json_formatter is None:
            raise RuntimeError("RoboFactory service requires structured JSON prompts")
        self.cfg = replace(self.cfg, action_stats_path=stats_path, raw_action_dim=8)
        self.raw_action_dim = 8
        self.action_normalizer = load_imagewam_zscore_normalizer(stats_path)

    def get_info(self) -> dict[str, Any]:
        info = super().get_info()
        info.update(
            {
                "adapter": "RoboFactoryActionModelService",
                "video_length": VIDEO_LENGTH,
                "action_horizon_per_agent": ACTION_HORIZON_PER_AGENT,
                "state_normalization": "none_raw_qpos9_qvel9",
                "action_normalization": "imagewam_global_meanstd_once",
                "gaussian_contract": "online_fp16_n13x28x40",
            }
        )
        return info

    def _prep_robofactory_item(self, req: dict[str, Any]) -> dict[str, Any]:
        prompt = req.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("'prompt' must be a non-empty string")
        agent_count = req.get("agent_count")
        if not isinstance(agent_count, int) or not 1 <= agent_count <= 4:
            raise ValueError("'agent_count' must be an integer in 1..4")
        image = _decode_base64_png_to_rgb_uint8(req.get("image", ""))
        if tuple(image.shape) != (3, 240, 320):
            raise ValueError(f"Global RGB must be [3,240,320], got {tuple(image.shape)}")
        state = decode_array(
            req.get("agent_state"),
            dtype=np.dtype("<f4"),
            shape=(agent_count, STATE_DIM),
            label="agent_state",
        ).float()
        gaussian = decode_array(
            req.get("agent_gaussian"),
            dtype=np.dtype("<f2"),
            shape=(agent_count, *GAUSSIAN_SHAPE),
            label="agent_gaussian",
        ).float()

        video = image.unsqueeze(1).repeat(1, VIDEO_LENGTH, 1, 1)
        resolution = get_vision_data_resolution((240, 320))
        target_w, target_h = find_closest_target_size(240, 320, resolution)
        padded: dict[str, Any] = {"video": video}
        reflection_pad_to_target(padded, ["video"], True, target_w, target_h)
        action_length = agent_count * ACTION_HORIZON_PER_AGENT
        sequence_plan = build_sequence_plan_from_mode(
            mode="policy", video_length=VIDEO_LENGTH, action_length=action_length, has_text=True
        )
        if self._prompt_json_formatter is None:
            raise ValueError("RoboFactory checkpoint requires format_prompt_as_json=True")
        prompt_payload: dict[str, Any] = {
            "ai_caption": prompt,
            "viewpoint": "third_person_view",
            "video": padded["video"],
            "image_size": padded["image_size"],
            "conditioning_fps": torch.tensor(10, dtype=torch.long),
            "mode": "policy",
            "action": torch.zeros((action_length, self.cfg.max_action_dim), dtype=torch.float32),
            "idle_frames": torch.tensor(0, dtype=torch.long),
        }
        formatted = self._prompt_json_formatter(prompt_payload)["ai_caption"]
        return {
            "image": image,
            "video": padded["video"],
            "image_size": padded["image_size"],
            "prompt": json.dumps(formatted) if isinstance(formatted, dict) else str(formatted),
            "sequence_plan": sequence_plan,
            "agent_count": agent_count,
            "state": state,
            "gaussian": gaussian,
            "action_length": action_length,
            "query_seed": int(req.get("query_seed", self.cfg.seed)),
        }

    def predict_policy(self, req: dict[str, Any]) -> dict[str, Any]:
        started = time.monotonic()
        item = self._prep_robofactory_item(req)
        action_length = item["action_length"]
        record = ActionProcessingRecord(raw_action_dim=8, action_normalizer=self.action_normalizer)
        batch: dict[str, Any] = {
            self._input_video_key(): [[item["video"]]],
            **make_batched_action_processing_fields(record, batch_size=1),
            "action": [[torch.zeros((action_length, self.cfg.max_action_dim), dtype=torch.float32)]],
            "mode": ["policy"],
            "ai_caption": [item["prompt"]],
            "prompt": [item["prompt"]],
            "conditioning_fps": [torch.tensor(10, dtype=torch.long)],
            "image_size": item["image_size"].unsqueeze(0).to(device="cuda"),
            "domain_id": [torch.tensor(get_domain_id("robofactory"), dtype=torch.long)],
            "sequence_plan": [item["sequence_plan"]],
            "agent_count": [torch.tensor(item["agent_count"], dtype=torch.long)],
            "action_horizon_per_agent": [torch.tensor(ACTION_HORIZON_PER_AGENT, dtype=torch.long)],
            "agent_state": [item["state"]],
            "agent_gaussian": [item["gaussian"]],
            "action_token_mask": [torch.ones(action_length, dtype=torch.bool)],
        }
        with self._lock, torch.inference_mode():
            samples = self.model.generate_samples_from_batch(
                batch,
                guidance=self.cfg.guidance,
                seed=[item["query_seed"]],
                num_steps=self.cfg.num_steps,
                has_negative_prompt=False,
            )
        # generate_samples_from_batch has already unpadded and denormalized via record.
        raw_agent_major = samples["action"][0].float().squeeze(0).cpu()
        normalized, raw = chronological_views(
            raw_agent_major, self.action_normalizer, agent_count=item["agent_count"]
        )
        return {
            "normalized": normalized.numpy().tolist(),
            "denormalized": raw.numpy().tolist(),
            "query_seed": item["query_seed"],
            "elapsed_seconds": time.monotonic() - started,
        }

    def predict_policy_batch(self, reqs: list[dict[str, Any]]) -> dict[str, Any]:
        raise ValueError("RoboFactory closed-loop service accepts one stateful request at a time")


def serve(args: RoboFactoryActionServerArgs) -> None:
    service = RoboFactoryActionModelService(args)
    log.info(
        f"[robofactory-action-server] accessible=http://{get_local_ip()}:{int(args.port)}/ "
        f"checkpoint={service.cfg.checkpoint_dir}"
    )
    httpd = ThreadingHTTPServer((args.host, int(args.port)), _ActionHandler)
    setattr(httpd, "service", service)
    httpd.serve_forever()


def main() -> None:
    args = tyro_cli(
        RoboFactoryActionServerArgs,
        description=__doc__,
        config=(
            tyro.conf.OmitArgPrefixes,
            tyro.conf.CascadeSubcommandArgs,
            tyro.conf.OmitSubcommandPrefixes,
        ),
    )
    serve(args)


if __name__ == "__main__":
    main()
