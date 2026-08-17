import json
import os

import torch
import torch.distributed as dist

from cosmos_framework.model.generator.mot.strong_action_expert import (
    StrongActionContextExpert,
    action_gradient_alignment_loss,
)


def main() -> None:
    variant = os.environ.get("CWAM_R6_VARIANT", "r6a").lower()
    if variant not in {"r6a", "r6b"}:
        raise ValueError("CWAM_R6_VARIANT must be r6a or r6b")
    delta_layers = 0 if variant == "r6a" else 4

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
        visual_memory_delta_conditioning_layers=delta_layers,
        visual_memory_delta_conditioning_scale=0.25,
        visual_memory_contrastive_temperature=0.07,
    ).to(device=device, dtype=dtype)
    expert.init_weights()

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
    visual_latents = [
        torch.randn(3, 48, 15, 26, device=device, dtype=dtype, requires_grad=True)
        for _ in range(batch_size)
    ]
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
        "visual_history_latent": visual_latents,
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
    output, auxiliary, metrics = expert(hidden_states, **forward_kwargs)
    if auxiliary is None:
        raise RuntimeError("R6 auxiliary branch did not produce action hidden states")
    main_action_loss = output[action_indexes_tensor].float().square().mean()
    gradient_loss, gradient_valid_fraction, _ = action_gradient_alignment_loss(
        main_action_loss,
        metrics["visual_memory_object_delta"],
        metrics["visual_memory_action_attention"],
    )
    loss = (
        main_action_loss
        + 0.25 * auxiliary.float().square().mean()
        + 0.20 * metrics["visual_memory_diversity_loss"]
        + 0.05 * metrics["visual_memory_assignment_entropy_loss"]
        + 0.05 * metrics["visual_memory_assignment_balance_loss"]
        + 0.05 * metrics["visual_memory_spatial_compactness_loss"]
        + 0.02 * metrics["visual_memory_temporal_correspondence_loss"]
        + 0.10 * metrics["visual_memory_object_contrastive_loss"]
        + 0.10 * metrics["visual_memory_episode_contrastive_loss"]
        + 0.05 * gradient_loss
    )
    loss.backward()

    required_prefixes = [
        "visual_object_memory.slot_q.",
        "visual_object_memory.token_k.",
        "visual_memory_aux_out.",
        "visual_memory_delta_main.cross_attention.",
        "visual_memory_delta_main.delta_out.",
    ]
    if variant == "r6b":
        required_prefixes.append("visual_memory_delta_conditioners.")
    required_gradients = {}
    for prefix in required_prefixes:
        parameters = [parameter for name, parameter in expert.named_parameters() if name.startswith(prefix)]
        required_gradients[prefix] = bool(parameters) and all(
            parameter.grad is not None
            and torch.isfinite(parameter.grad).all().item()
            and torch.count_nonzero(parameter.grad).item() > 0
            for parameter in parameters
        )

    aligned_checks = (
        metrics["visual_memory_main_context_tokens"].item() == 0.0,
        metrics["visual_memory_conditioned_residual_rms"].item() == 0.0,
        metrics["visual_memory_aux_conditioned_residual_rms"].item() > 0.0,
        metrics["visual_memory_delta_conditioning_layers"].item() == float(delta_layers),
        metrics["visual_memory_delta_conditioned_residual_rms"].item()
        > (0.0 if variant == "r6b" else -1.0),
        metrics["visual_memory_object_delta_active_fraction"].item() == 1.0,
        metrics["visual_memory_delta_main_applied_delta_rms"].item() > 0.0,
    )
    if variant == "r6a":
        aligned_checks += (metrics["visual_memory_delta_conditioned_residual_rms"].item() == 0.0,)

    expert.visual_object_memory.intervention = "zero"
    with torch.no_grad():
        _, _, zero_metrics = expert(hidden_states.detach(), **forward_kwargs)
    zero_checks = (
        zero_metrics["visual_memory_object_delta_rms"].item() == 0.0,
        zero_metrics["visual_memory_object_delta_active_fraction"].item() == 0.0,
        zero_metrics["visual_memory_delta_conditioned_residual_rms"].item() == 0.0,
        zero_metrics["visual_memory_delta_main_gate_deviation"].item() == 0.0,
        zero_metrics["visual_memory_delta_main_residual_rms"].item() == 0.0,
        zero_metrics["visual_memory_delta_main_applied_delta_rms"].item() == 0.0,
        torch.count_nonzero(zero_metrics["visual_memory_object_delta"]).item() == 0,
    )
    local_pass = all(
        (
            torch.isfinite(loss).item(),
            torch.isfinite(gradient_loss).item(),
            gradient_valid_fraction.item() > 0.0,
            all(required_gradients.values()),
            all(aligned_checks),
            all(zero_checks),
        )
    )
    if not local_pass:
        print(
            json.dumps(
                {
                    "rank": rank,
                    "finite_loss": bool(torch.isfinite(loss).item()),
                    "finite_gradient_loss": bool(torch.isfinite(gradient_loss).item()),
                    "gradient_valid_fraction": float(gradient_valid_fraction.detach()),
                    "required_gradients": required_gradients,
                    "aligned_checks": aligned_checks,
                    "zero_checks": zero_checks,
                },
                sort_keys=True,
            )
        )
    pass_tensor = torch.tensor(int(local_pass), device=device)
    dist.all_reduce(pass_tensor, op=dist.ReduceOp.MIN)
    if rank == 0:
        print(
            json.dumps(
                {
                    "variant": variant,
                    "world_size": dist.get_world_size(),
                    "finite_loss": bool(torch.isfinite(loss).item()),
                    "gradient_valid_fraction": float(gradient_valid_fraction.detach()),
                    "loss": float(loss.detach()),
                    "peak_memory_mib": round(torch.cuda.max_memory_allocated(device) / 2**20, 2),
                    "main_context_tokens": float(
                        metrics["visual_memory_main_context_tokens"].detach()
                    ),
                    "delta_conditioning_layers": float(
                        metrics["visual_memory_delta_conditioning_layers"].detach()
                    ),
                    "delta_conditioned_residual_rms": float(
                        metrics["visual_memory_delta_conditioned_residual_rms"].detach()
                    ),
                    "applied_delta_rms": float(
                        metrics["visual_memory_delta_main_applied_delta_rms"].detach()
                    ),
                    "zero_checks": bool(all(zero_checks)),
                    "required_gradients": required_gradients,
                    "all_ranks_pass": bool(pass_tensor.item()),
                },
                sort_keys=True,
            )
        )
    dist.destroy_process_group()
    if not bool(pass_tensor.item()):
        raise RuntimeError(f"{variant} dynamic-exclusive BF16 distributed smoke gate failed")


if __name__ == "__main__":
    main()
