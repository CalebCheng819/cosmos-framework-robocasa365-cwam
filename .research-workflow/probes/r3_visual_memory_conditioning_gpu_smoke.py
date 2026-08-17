import json

import torch

from cosmos_framework.model.generator.mot.strong_action_expert import StrongActionContextExpert


def main() -> None:
    torch.manual_seed(20260817)
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
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

    torch.cuda.reset_peak_memory_stats(device)
    output, auxiliary, metrics = expert(
        hidden_states,
        action_indexes=action_indexes_tensor,
        text_indexes=torch.cat(text_indexes),
        vision_indexes=torch.cat(vision_indexes),
        position_ids=position_ids,
        sample_lens=[sample_length] * batch_size,
        proprio_history=[
            torch.randn(4, 16, device=device, dtype=torch.float32) for _ in range(batch_size)
        ],
        proprio_history_mask=[torch.ones(4, device=device, dtype=torch.bool) for _ in range(batch_size)],
        action_history=[
            torch.randn(16, 10, device=device, dtype=torch.float32) for _ in range(batch_size)
        ],
        action_history_mask=[torch.ones(16, device=device, dtype=torch.bool) for _ in range(batch_size)],
        visual_history_latent=visual_latents,
        visual_history_mask=[torch.ones(3, device=device, dtype=torch.bool) for _ in range(batch_size)],
        action_seed_hidden=action_seed_hidden,
    )
    if auxiliary is None:
        raise RuntimeError("R3 visual-memory auxiliary branch did not produce action hidden states")
    loss = (
        output[action_indexes_tensor].float().square().mean()
        + auxiliary.float().square().mean()
        + 0.05 * metrics["visual_memory_diversity_loss"]
        + 0.05 * metrics["visual_memory_temporal_correspondence_loss"]
    )
    loss.backward()

    required_gradient_prefixes = (
        "visual_object_memory.temporal_predictor.",
        "visual_memory_conditioners.",
        "visual_memory_aux_norm.",
        "visual_memory_aux_out.",
    )
    required_gradients = {}
    for prefix in required_gradient_prefixes:
        parameters = [
            parameter
            for name, parameter in expert.named_parameters()
            if name.startswith(prefix)
        ]
        required_gradients[prefix] = bool(parameters) and all(
            parameter.grad is not None
            and torch.isfinite(parameter.grad).all().item()
            and torch.count_nonzero(parameter.grad).item() > 0
            for parameter in parameters
        )

    result = {
        "cuda_device": torch.cuda.get_device_name(device),
        "dtype": str(output.dtype),
        "finite_loss": bool(torch.isfinite(loss).item()),
        "loss": float(loss.detach()),
        "peak_memory_mib": round(torch.cuda.max_memory_allocated(device) / 2**20, 2),
        "slot_abs_cosine": float(metrics["visual_memory_slot_abs_cosine"].detach()),
        "temporal_correspondence_loss": float(
            metrics["visual_memory_temporal_correspondence_loss"].detach()
        ),
        "conditioned_residual_rms": float(
            metrics["visual_memory_conditioned_residual_rms"].detach()
        ),
        "auxiliary_shape": list(auxiliary.shape),
        "required_gradients": required_gradients,
    }
    print(json.dumps(result, sort_keys=True))
    if not result["finite_loss"] or not all(required_gradients.values()):
        raise RuntimeError("R3 visual-memory BF16 smoke gate failed")


if __name__ == "__main__":
    main()
