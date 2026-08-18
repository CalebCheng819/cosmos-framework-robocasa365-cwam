"""Attach ordered stage diagnostics to the pinned DeMiAn RoboCasa evaluator."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import tyro

from cosmos_framework.scripts.robocasa_stage_diagnostics import OrderedStageTracker, SUPPORTED_TASKS
from eval_utils import run_robocasa_eval as base


def reset_episode_policy(client: Any) -> None:
    """Reset both the evaluator chunk buffer and the server-side C history."""
    client.reset()
    metadata = client.client.get_server_metadata()
    launch_nonce = metadata.get("launch_nonce")
    reset_protocol = metadata.get("control_protocol", {}).get("reset")
    if not launch_nonce or reset_protocol != "nonce_bound_websocket_control_v1":
        raise RuntimeError("CWAM diagnostic server does not advertise the required episode reset protocol")
    response = client.client.infer(
        {
            "__cwam_control__": "reset",
            "launch_nonce": launch_nonce,
        }
    )
    expected = {
        "actions": [],
        "cwam_control": "reset_ack",
        "launch_nonce": launch_nonce,
    }
    if response != expected:
        raise RuntimeError(f"CWAM diagnostic reset acknowledgement drifted: {response!r}")


def run_episode_with_stages(
    env: Any,
    client: Any,
    instruction: str,
    max_steps: int,
    headless: bool,
    init_state: dict[str, Any] | None = None,
    seed: int | None = None,
    fg_caption_fn: Any = None,
    caption_suffix: str | None = None,
    caption_format: str = "reasoning",
) -> dict[str, Any]:
    reset_episode_policy(client)
    if init_state is not None:
        raw_env = env.unwrapped
        raw_env.set_ep_meta(init_state["ep_meta"])
        obs, _info = env.reset()
        instruction = init_state["ep_meta"].get("lang", instruction)
    else:
        obs, _info = env.reset(seed=seed)
        env_instruction = (obs.get("annotation.human.task_description", instruction) or instruction).strip()
        if caption_suffix:
            if caption_format == "concat":
                instruction = f"{env_instruction.rstrip('.')}. {caption_suffix}"
            else:
                instruction = f"Task: {env_instruction} Reasoning: {caption_suffix}"
        else:
            instruction = env_instruction

    tracker = OrderedStageTracker(env.unwrapped)
    subtitle_lines = [f"Task: {instruction}"]
    video: list[np.ndarray] = []
    action_log: list[list[float]] = []
    generated_texts: list[dict[str, Any]] = []
    success = False
    place_anytime = False
    place_first_step: int | None = None
    subtask_successes: list[bool] = []

    for step in range(max_steps):
        caption = fg_caption_fn(step) if fg_caption_fn else None
        ret = client.infer(obs, instruction, reasoning_text=caption)
        video.append(base._add_subtitle(ret["viz"], subtitle_lines, step))
        if ret.get("generated_text"):
            generated_texts.append({"step": step, "text": ret["generated_text"]})
        action_log.append(np.asarray(ret["raw_action"]).tolist())
        if not headless and base._HAS_CV2:
            base.cv2.imshow("CWAM C1 stage diagnostic", base.cv2.cvtColor(ret["viz"], base.cv2.COLOR_RGB2BGR))
            base.cv2.waitKey(1)

        obs, _reward, done, _truncated, info = env.step(ret["action_dict"])
        newly_reached = tracker.update(step)
        if newly_reached:
            print(f"  [stage step={step:03d}] {','.join(newly_reached)}", flush=True)
        if info.get("success", False):
            success = True
            done = True
        if not place_anytime and base._check_loose_success(env):
            place_anytime = True
            place_first_step = step
        if "subtask_successes" in info:
            subtask_successes = info["subtask_successes"]
        if done:
            break

    result: dict[str, Any] = {
        "success": success,
        "place_anytime": place_anytime,
        "place_first_step": place_first_step,
        "steps": step + 1,
        "subtask_successes": subtask_successes,
        "video": video,
        "action_log": action_log,
        "instruction": instruction,
        "diagnostic_condition": os.environ.get("CWAM_DIAGNOSTIC_CONDITION", "unspecified"),
        "diagnostic_horizon": int(os.environ.get("CWAM_DIAGNOSTIC_HORIZON", client.open_loop_horizon)),
        **tracker.summary(),
    }
    if generated_texts:
        result["generated_texts"] = generated_texts
    return result


def main() -> None:
    args = tyro.cli(base.Args)
    requested = base.select_tasks(args)
    unsupported = sorted(set(requested) - set(SUPPORTED_TASKS))
    if unsupported:
        raise ValueError(f"Stage diagnostic only supports {SUPPORTED_TASKS}, got {unsupported}")
    base.run_episode = run_episode_with_stages
    base.main(args)


if __name__ == "__main__":
    main()
