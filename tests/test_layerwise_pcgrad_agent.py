from __future__ import annotations

import numpy as np
import pytest
import torch

from agents import FullBehaviorCloningSACAgent, LayerwiseAdaptivePCGradAgent
from methods import get_method


def make_agent(agent_type: type[FullBehaviorCloningSACAgent]):
    return agent_type(
        observation_dim=6,
        action_dim=2,
        action_low=np.full(2, -1.0, dtype=np.float32),
        action_high=np.full(2, 1.0, dtype=np.float32),
        num_tasks=3,
        task_id_dim=3,
        hide_task_id=True,
        device="cpu",
        episodic_batch_size=4,
        actor_cloning_coefficient=100.0,
        bc_gradient_strategy="pcgrad_sac_priority",
        bc_max_norm_ratio=1.0,
        bc_combination_strategy="adaptive_additive",
        bc_adaptive_target_ratio=0.2,
        bc_adaptive_conflict_ratio=0.05,
    )


def synthetic_gradients(
    agent: FullBehaviorCloningSACAgent,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    parameters = tuple(agent.actor.parameters())
    sac = [torch.zeros_like(parameter) for parameter in parameters]
    bc = [torch.zeros_like(parameter) for parameter in parameters]
    layers = agent.gradient_diagnostics.layer_indices
    for layer_position, indices in enumerate(layers.values()):
        for index in indices:
            sac[index] = torch.ones_like(parameters[index])
            if layer_position == 1:
                values = torch.arange(
                    parameters[index].numel(), dtype=parameters[index].dtype
                ).reshape_as(parameters[index])
                values = values - values.mean()
                bc[index] = values - torch.ones_like(values)
            else:
                bc[index] = 0.3 * torch.ones_like(parameters[index])

    shared = set(agent.gradient_diagnostics.shared_indices)
    for index, parameter in enumerate(parameters):
        if index not in shared:
            sac[index] = 0.1 * torch.ones_like(parameter)
            bc[index] = 0.7 * torch.ones_like(parameter)
    return tuple(sac), tuple(bc)


def gradient_norm(
    gradients: tuple[torch.Tensor, ...], indices: tuple[int, ...]
) -> torch.Tensor:
    return sum(gradients[index].square().sum() for index in indices).sqrt()


def gradient_dot(
    first: tuple[torch.Tensor, ...],
    second: tuple[torch.Tensor, ...],
    indices: tuple[int, ...],
) -> torch.Tensor:
    return sum((first[index] * second[index]).sum() for index in indices)


def apply_and_combine(
    agent: FullBehaviorCloningSACAgent,
    sac: tuple[torch.Tensor, ...],
    bc: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, ...]:
    adjusted, metadata = agent._apply_bc_gradient_strategy(
        sac_gradients=sac,
        cloning_gradients=bc,
    )
    combined, _ = agent._combine_actor_gradients(
        sac_gradients=sac,
        applied_cloning_gradients=adjusted,
        conflict=bool(metadata["conflict"]),
    )
    return combined


def test_layerwise_projection_uses_each_backbone_layer_geometry() -> None:
    agent = make_agent(LayerwiseAdaptivePCGradAgent)
    sac, bc = synthetic_gradients(agent)
    combined = apply_and_combine(agent, sac, bc)
    effective_bc = tuple(
        final - sac_gradient
        for final, sac_gradient in zip(combined, sac, strict=True)
    )
    layers = list(agent.gradient_diagnostics.layer_indices.values())

    compatible_ratio = gradient_norm(effective_bc, layers[0]) / gradient_norm(
        sac, layers[0]
    )
    conflicting_ratio = gradient_norm(effective_bc, layers[1]) / gradient_norm(
        sac, layers[1]
    )
    assert compatible_ratio.item() == pytest.approx(0.2, abs=1e-6)
    assert conflicting_ratio.item() == pytest.approx(0.05, abs=1e-6)
    assert gradient_dot(sac, effective_bc, layers[1]).item() >= -1e-3


def test_layerwise_method_preserves_original_global_head_scaling() -> None:
    baseline = make_agent(FullBehaviorCloningSACAgent)
    layerwise = make_agent(LayerwiseAdaptivePCGradAgent)
    layerwise.load_state_dict(baseline.state_dict())
    sac, bc = synthetic_gradients(baseline)

    baseline_combined = apply_and_combine(baseline, sac, bc)
    layerwise_combined = apply_and_combine(layerwise, sac, bc)
    shared = set(baseline.gradient_diagnostics.shared_indices)
    for index in range(len(sac)):
        if index not in shared:
            torch.testing.assert_close(
                layerwise_combined[index], baseline_combined[index]
            )


def test_layerwise_agent_is_checkpoint_compatible_but_separately_registered() -> None:
    baseline = make_agent(FullBehaviorCloningSACAgent)
    layerwise = make_agent(LayerwiseAdaptivePCGradAgent)
    assert baseline.state_dict().keys() == layerwise.state_dict().keys()
    layerwise.load_state_dict(baseline.state_dict(), strict=True)

    original = get_method("success_replay_best_adaptive_pcgrad")
    candidate = get_method("success_replay_best_layerwise_adaptive_pcgrad")
    assert original.method_id != candidate.method_id
    assert original.agent_factory is not candidate.agent_factory
