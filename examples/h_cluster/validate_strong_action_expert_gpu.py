# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Run a realistic BF16 forward/backward gate for StrongActionContextExpert."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import uuid
from pathlib import Path

import torch

from cosmos_framework.model.generator.mot.strong_action_expert import StrongActionContextExpert


def _git_head(root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def _atomic_write_json(path: Path, value: object) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to replace GPU gate receipt: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial-{uuid.uuid4().hex}")
    payload = json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.rename(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _packed_inputs(*, device: torch.device, dtype: torch.dtype) -> dict[str, object]:
    batch_size = 2
    text_length = 32
    vision_height = 30
    vision_width = 52
    vision_length = vision_height * vision_width
    action_length = 16
    sample_length = text_length + vision_length + action_length
    total_length = batch_size * sample_length

    hidden_states = torch.randn(total_length, 4096, device=device, dtype=dtype, requires_grad=True)
    position_ids = torch.zeros(3, total_length, device=device, dtype=torch.long)
    text_indexes: list[torch.Tensor] = []
    vision_indexes: list[torch.Tensor] = []
    action_indexes: list[torch.Tensor] = []
    horizontal = torch.arange(vision_width, device=device).repeat(vision_height)
    for sample_id in range(batch_size):
        start = sample_id * sample_length
        text = torch.arange(start, start + text_length, device=device)
        vision = torch.arange(start + text_length, start + text_length + vision_length, device=device)
        action = torch.arange(start + text_length + vision_length, start + sample_length, device=device)
        text_indexes.append(text)
        vision_indexes.append(vision)
        action_indexes.append(action)
        position_ids[2, vision] = horizontal

    proprio_history = [
        torch.randn(4, 16, device=device, dtype=torch.float32, requires_grad=True)
        for _ in range(batch_size)
    ]
    action_history = [
        torch.randn(16, 10, device=device, dtype=torch.float32, requires_grad=True)
        for _ in range(batch_size)
    ]
    return {
        "hidden_states": hidden_states,
        "action_indexes": torch.cat(action_indexes),
        "text_indexes": torch.cat(text_indexes),
        "vision_indexes": torch.cat(vision_indexes),
        "position_ids": position_ids,
        "sample_lens": [sample_length] * batch_size,
        "proprio_history": proprio_history,
        "proprio_history_mask": [
            torch.tensor([False, True, True, True], device=device),
            torch.tensor([True, True, True, True], device=device),
        ],
        "action_history": action_history,
        "action_history_mask": [
            torch.tensor([False] * 4 + [True] * 12, device=device),
            torch.tensor([True] * 16, device=device),
        ],
    }


def run_gate(*, source_root: Path, expected_commit: str) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the strong action expert GPU gate")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The active CUDA device does not support BF16")
    actual_commit = _git_head(source_root)
    if actual_commit != expected_commit:
        raise RuntimeError(f"Source commit mismatch: expected={expected_commit}, actual={actual_commit}")

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    device = torch.device("cuda", torch.cuda.current_device())
    dtype = torch.bfloat16
    expert = StrongActionContextExpert(
        hidden_size=4096,
        model_dim=1024,
        num_heads=16,
        num_layers=4,
        ffn_multiplier=4,
        state_dim=16,
        action_dim=10,
        proprio_history_length=4,
        proprio_history_stride=12,
        action_history_length=16,
        memory_layers=2,
        memory_tokens=4,
        num_views=3,
        view_token_width=16,
        initial_residual_gate=0.01,
    ).to(device=device, dtype=dtype)
    expert.init_weights()
    inputs = _packed_inputs(device=device, dtype=dtype)

    canvas_positions = torch.zeros(52, 3, device=device, dtype=torch.long)
    canvas_positions[:, 2] = torch.arange(52, device=device)
    view_ids = expert._vision_view_ids(canvas_positions)
    expected_view_ids = torch.tensor([0] * 18 + [1] * 16 + [2] * 18, device=device)
    torch.testing.assert_close(view_ids, expected_view_ids)

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    output, auxiliary, metrics = expert(**inputs)
    if auxiliary is not None:
        raise RuntimeError("Visual-memory auxiliary output must be disabled for the baseline GPU gate")
    action_indexes = inputs["action_indexes"]
    assert isinstance(action_indexes, torch.Tensor)
    loss = output[action_indexes].float().square().mean()
    loss.backward()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started

    if not torch.isfinite(output).all() or not torch.isfinite(loss):
        raise RuntimeError("Strong action expert produced non-finite output or loss")
    missing_gradients: list[str] = []
    nonfinite_gradients: list[str] = []
    zero_gradient_parameters: list[str] = []
    gradient_elements = 0
    parameter_elements = 0
    for name, parameter in expert.named_parameters():
        parameter_elements += parameter.numel()
        if parameter.grad is None:
            missing_gradients.append(name)
            continue
        gradient_elements += parameter.grad.numel()
        if not torch.isfinite(parameter.grad).all():
            nonfinite_gradients.append(name)
        if torch.count_nonzero(parameter.grad) == 0:
            zero_gradient_parameters.append(name)
    if missing_gradients or nonfinite_gradients or zero_gradient_parameters:
        raise RuntimeError(
            "Strong action expert gradient gate failed: "
            f"missing={missing_gradients}, nonfinite={nonfinite_gradients}, zero={zero_gradient_parameters}"
        )

    for field in ("proprio_history", "action_history"):
        for index, value in enumerate(inputs[field]):
            assert isinstance(value, torch.Tensor)
            if value.grad is None or not torch.isfinite(value.grad).all() or torch.count_nonzero(value.grad) == 0:
                raise RuntimeError(f"Missing finite nonzero context gradient for {field}[{index}]")
    hidden_states = inputs["hidden_states"]
    assert isinstance(hidden_states, torch.Tensor) and hidden_states.grad is not None
    for field in ("text_indexes", "vision_indexes", "action_indexes"):
        indexes = inputs[field]
        assert isinstance(indexes, torch.Tensor)
        if torch.count_nonzero(hidden_states.grad[indexes]) == 0:
            raise RuntimeError(f"Packed hidden-state gradient is zero for {field}")

    properties = torch.cuda.get_device_properties(device)
    return {
        "schema_version": 1,
        "status": "passed",
        "git_commit": actual_commit,
        "source_path": "cosmos_framework/model/generator/mot/strong_action_expert.py",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": properties.name,
        "device_total_memory_bytes": properties.total_memory,
        "dtype": str(dtype),
        "batch_size": 2,
        "hidden_size": 4096,
        "model_dim": 1024,
        "layers": 4,
        "vision_tokens_per_sample": 1560,
        "action_tokens_per_sample": 16,
        "view_token_counts": [18, 16, 18],
        "parameter_elements": parameter_elements,
        "gradient_elements": gradient_elements,
        "loss": float(loss.detach()),
        "metrics": {name: float(value.detach()) for name, value in metrics.items()},
        "elapsed_seconds": elapsed,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    source_root = args.source_root.resolve(strict=True)
    receipt = args.receipt.resolve(strict=False)
    result = run_gate(source_root=source_root, expected_commit=args.expected_commit)
    _atomic_write_json(receipt, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
