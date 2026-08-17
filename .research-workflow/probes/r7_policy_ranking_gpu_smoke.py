import json
import os

import torch
import torch.distributed as dist
from torch import nn

from cosmos_framework.model.generator.mot.strong_action_expert import StrongActionContextExpert
from cosmos_framework.model.generator.omni_mot_model import intervention_policy_ranking_loss


def main() -> None:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dtype = torch.bfloat16
    torch.manual_seed(20260817 + rank)

    batch_size = 2
    text_length = 32
    vision_length = 30 * 52
    action_length = 16
    sample_length = text_length + vision_length + action_length
    total_length = batch_size * sample_length
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
        visual_memory_enabled=True,
        visual_history_length=3,
        visual_latent_channels=48,
        visual_latent_height=15,
        visual_latent_width=26,
        visual_memory_layers=2,
        visual_memory_tokens=8,
        visual_memory_initial_gate=0.1,
        visual_memory_conditioning_scale=0.25,
        visual_memory_delta_main_enabled=True,
        visual_memory_delta_gate_scale=0.25,
        visual_memory_delta_residual_scale=0.25,
        visual_memory_delta_main_component="full",
        visual_memory_exclusive_dynamic=True,
        visual_memory_delta_conditioning_layers=0,
        visual_memory_delta_conditioning_scale=0.25,
        visual_memory_contrastive_temperature=0.07,
        visual_memory_intervention="none",
    ).to(device=device, dtype=dtype)
    expert.init_weights()
    decoder = nn.Linear(4096, 10, bias=False).to(device=device, dtype=dtype)

    hidden_states = torch.randn(total_length, 4096, device=device, dtype=dtype, requires_grad=True)
    position_ids = torch.zeros(3, total_length, device=device, dtype=torch.long)
    text_indexes = []
    vision_indexes = []
    action_indexes = []
    horizontal = torch.arange(52, device=device).repeat(30)
    for sample_id in range(batch_size):
        start = sample_id * sample_length
        text_indexes.append(torch.arange(start, start + text_length, device=device))
        vision_start = start + text_length
        vision_indexes.append(torch.arange(vision_start, vision_start + vision_length, device=device))
        action_indexes.append(
            torch.arange(vision_start + vision_length, start + sample_length, device=device)
        )
        position_ids[2, vision_indexes[-1]] = horizontal
    action_indexes_tensor = torch.cat(action_indexes)
    action_seed_hidden = torch.randn(
        batch_size * action_length,
        4096,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    forward_kwargs = {
        "action_indexes": action_indexes_tensor,
        "text_indexes": torch.cat(text_indexes),
        "vision_indexes": torch.cat(vision_indexes),
        "position_ids": position_ids,
        "sample_lens": [sample_length] * batch_size,
        "proprio_history": [
            torch.randn(4, 16, device=device, dtype=torch.float32) for _ in range(batch_size)
        ],
        "proprio_history_mask": [
            torch.ones(4, device=device, dtype=torch.bool) for _ in range(batch_size)
        ],
        "action_history": [
            torch.randn(16, 10, device=device, dtype=torch.float32) for _ in range(batch_size)
        ],
        "action_history_mask": [
            torch.ones(16, device=device, dtype=torch.bool) for _ in range(batch_size)
        ],
        "visual_history_latent": [
            torch.randn(3, 48, 15, 26, device=device, dtype=dtype) for _ in range(batch_size)
        ],
        "visual_history_mask": [
            torch.ones(3, device=device, dtype=torch.bool) for _ in range(batch_size)
        ],
        "episode_ids": [
            torch.tensor(rank * batch_size + sample_id, device=device)
            for sample_id in range(batch_size)
        ],
        "action_seed_hidden": action_seed_hidden,
    }

    torch.cuda.reset_peak_memory_stats(device)
    aligned_hidden, _, aligned_metrics = expert(hidden_states, **forward_kwargs)
    aligned_prediction = decoder(aligned_hidden[action_indexes_tensor]).reshape(
        batch_size, action_length, 10
    )
    target = torch.randn_like(aligned_prediction)
    aligned_loss = (aligned_prediction.float() - target.float()).square().mean(dim=(1, 2))

    counterfactual_losses = {}
    counterfactual_requires_grad = {}
    counterfactual_metrics = {}
    for intervention in ("zero", "shuffle"):
        with torch.no_grad():
            intervention_hidden, _, metrics = expert(
                hidden_states,
                visual_memory_intervention_override=intervention,
                **forward_kwargs,
            )
            prediction = decoder(intervention_hidden[action_indexes_tensor]).reshape(
                batch_size, action_length, 10
            )
            counterfactual_losses[intervention] = (
                (prediction.float() - target.float()).square().mean(dim=(1, 2))
            )
        counterfactual_requires_grad[intervention] = prediction.requires_grad
        counterfactual_metrics[intervention] = metrics

    rank_loss, rank_metrics = intervention_policy_ranking_loss(
        aligned_loss,
        counterfactual_losses["zero"],
        counterfactual_losses["shuffle"],
        relative_margin=0.01,
    )
    rank_gradient = torch.autograd.grad(
        rank_loss,
        expert.visual_memory_delta_main.delta_out.weight,
        retain_graph=True,
        allow_unused=True,
    )[0]
    total_loss = aligned_loss.mean() + rank_loss
    total_loss.backward()

    zero_metrics = counterfactual_metrics["zero"]
    local_checks = {
        "finite_total_loss": bool(torch.isfinite(total_loss).item()),
        "positive_rank_loss": bool(rank_loss.detach().item() > 0.0),
        "rank_gradient_nonzero": bool(
            rank_gradient is not None
            and torch.isfinite(rank_gradient).all().item()
            and torch.count_nonzero(rank_gradient).item() > 0
        ),
        "zero_no_grad": not counterfactual_requires_grad["zero"],
        "shuffle_no_grad": not counterfactual_requires_grad["shuffle"],
        "zero_delta_exact": bool(zero_metrics["visual_memory_object_delta_rms"].item() == 0.0),
        "zero_main_exact": bool(
            zero_metrics["visual_memory_delta_main_applied_delta_rms"].item() == 0.0
        ),
        "override_non_mutating": expert.visual_object_memory.intervention == "none",
        "aligned_delta_active": bool(
            aligned_metrics["visual_memory_object_delta_active_fraction"].item() == 1.0
        ),
    }
    local_pass = all(local_checks.values())
    pass_tensor = torch.tensor(int(local_pass), device=device)
    dist.all_reduce(pass_tensor, op=dist.ReduceOp.MIN)
    if rank == 0:
        print(
            json.dumps(
                {
                    "world_size": dist.get_world_size(),
                    "rank_loss": float(rank_loss.detach()),
                    "zero_relative_gain": float(rank_metrics["zero_relative_gain"]),
                    "shuffle_relative_gain": float(rank_metrics["shuffle_relative_gain"]),
                    "peak_memory_mib": round(torch.cuda.max_memory_allocated(device) / 2**20, 2),
                    "checks": local_checks,
                    "all_ranks_pass": bool(pass_tensor.item()),
                },
                sort_keys=True,
            )
        )
    dist.destroy_process_group()
    if not bool(pass_tensor.item()):
        raise RuntimeError("R7 aligned/zero/shuffle policy-ranking BF16 gate failed")


if __name__ == "__main__":
    main()
