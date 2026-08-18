import copy

import numpy as np
import torch

from methods import get_method
from agents import (
    EWCSACAgent,
    L2SACAgent,
    PackNetSACAgent,
    SSDEFullSACAgent,
    SSDESACAgent,
)
from agents.replay_buffer import ReplayBuffer


def make_agent(agent_type, **kwargs):
    return agent_type(
        observation_dim=7,
        action_dim=2,
        action_low=np.full(2, -1.0, dtype=np.float32),
        action_high=np.full(2, 1.0, dtype=np.float32),
        num_tasks=3,
        task_id_dim=3,
        hide_task_id=True,
        device="cpu",
        gradient_clip_norm=None,
        **kwargs,
    )


def make_full_ssde(**kwargs):
    def encoder(description: str) -> np.ndarray:
        offset = sum(description.encode("utf-8")) % 7
        return np.linspace(-0.8, 0.8, 8, dtype=np.float32) + offset * 0.01

    settings = {
        "ssde_seed": 11,
        "hidden_size": 16,
        "descriptor_dim": 8,
        "fixed_lasso_alpha": 1e-4,
        "random_lasso_alpha_start": 1e-4,
        "random_lasso_alpha_end": 1e-5,
        "sensitivity_interval": 10,
        "sensitivity_batch_size": 8,
        "random_distillation": False,
        "descriptor_encoder": encoder,
    }
    settings.update(kwargs)
    return make_agent(SSDEFullSACAgent, **settings)


def task_observations(task_index: int, count: int = 8) -> torch.Tensor:
    observations = torch.randn(count, 7)
    observations[:, -3:] = 0.0
    observations[:, 4 + task_index] = 1.0
    return observations


def replay_for_task(task_index: int, count: int = 16) -> ReplayBuffer:
    replay = ReplayBuffer(observation_dim=7, action_dim=2, capacity=count, seed=3)
    observations = task_observations(task_index, count).numpy()
    for observation in observations:
        replay.add(
            observation=observation,
            action=np.zeros(2, dtype=np.float32),
            reward=0.0,
            next_observation=observation,
            terminated=False,
        )
    return replay


def update_task(agent, task_index: int) -> None:
    observations = task_observations(task_index)
    agent.update_batch(
        observations=observations,
        actions=torch.zeros(8, 2),
        rewards=torch.zeros(8, 1),
        next_observations=task_observations(task_index),
        dones=torch.zeros(8, 1),
        collect_metrics=False,
    )


def test_packnet_retains_largest_weights_and_releases_the_rest() -> None:
    agent = make_agent(PackNetSACAgent, total_tasks=3)
    managed = agent._managed_actor_parameters[0]
    with torch.no_grad():
        values = torch.arange(
            1,
            managed.parameter.numel() + 1,
            dtype=managed.parameter.dtype,
        ).reshape_as(managed.parameter)
        managed.parameter.copy_(values)

    agent._allocate_actor_weights(task_index=0, allocation_fraction=0.25)

    expected_count = round(0.25 * managed.parameter.numel())
    assert int((managed.owner == 0).sum()) == expected_count
    retained = managed.parameter[managed.owner == 0]
    released = managed.parameter[managed.owner == -1]
    assert torch.all(retained >= managed.parameter.numel() - expected_count + 1)
    assert torch.count_nonzero(released) == 0


def test_packnet_old_weights_do_not_move_on_new_task() -> None:
    agent = make_agent(PackNetSACAgent, total_tasks=3)
    agent.on_task_end(
        task_index=0,
        replay_buffer=replay_for_task(0),
        batch_size=8,
    )
    before = {
        managed.name: managed.parameter[managed.owner == 0].detach().clone()
        for managed in agent._managed_actor_parameters
    }

    update_task(agent, 1)

    for managed in agent._managed_actor_parameters:
        assert torch.equal(
            managed.parameter[managed.owner == 0],
            before[managed.name],
        )


def test_packnet_evaluation_masks_future_weights_and_restores_training_view() -> None:
    agent = make_agent(PackNetSACAgent, total_tasks=3)
    agent.on_task_end(
        task_index=0,
        replay_buffer=replay_for_task(0),
        batch_size=8,
    )
    agent.on_task_start(task_index=1, replay_buffer=object())
    managed = agent._managed_actor_parameters[0]
    with torch.no_grad():
        managed.parameter[managed.owner == -1] = 7.0
    training_view = managed.parameter.detach().clone()

    agent.on_evaluation_start(0)
    assert torch.count_nonzero(managed.parameter[managed.owner == -1]) == 0
    agent.on_evaluation_end(0)

    assert torch.equal(managed.parameter, training_view)


def test_packnet_masks_survive_state_dict_round_trip() -> None:
    source = make_agent(PackNetSACAgent, total_tasks=3)
    source.on_task_end(
        task_index=0,
        replay_buffer=replay_for_task(0),
        batch_size=8,
    )
    restored = make_agent(PackNetSACAgent, total_tasks=3)
    restored.load_state_dict(copy.deepcopy(source.state_dict()), strict=True)

    for source_weight, restored_weight in zip(
        source._managed_actor_parameters,
        restored._managed_actor_parameters,
        strict=True,
    ):
        assert torch.equal(source_weight.owner, restored_weight.owner)


def test_l2_regularization_changes_only_backbone_gradients() -> None:
    agent = make_agent(L2SACAgent, cl_reg_coef=2.0)
    agent.on_task_end(
        task_index=0,
        replay_buffer=replay_for_task(0),
        batch_size=8,
    )
    backbone_parameter = next(agent.actor.backbone.parameters())
    head_parameter = agent.actor.mean_head.weight
    with torch.no_grad():
        backbone_parameter.add_(0.5)
        head_parameter.add_(0.5)
    gradients = (
        torch.zeros_like(backbone_parameter),
        torch.zeros_like(head_parameter),
    )

    adjusted = agent.adjust_actor_gradients(
        gradients=gradients,
        parameters=(backbone_parameter, head_parameter),
        task_index=1,
    )

    assert torch.allclose(adjusted[0], torch.full_like(adjusted[0], 2.0))
    assert torch.count_nonzero(adjusted[1]) == 0


def test_regularization_state_survives_state_dict_round_trip() -> None:
    source = make_agent(EWCSACAgent, fisher_batches=1)
    source.on_task_end(
        task_index=0,
        replay_buffer=replay_for_task(0),
        batch_size=8,
    )
    restored = make_agent(EWCSACAgent, fisher_batches=1)
    restored.load_state_dict(copy.deepcopy(source.state_dict()), strict=True)

    assert restored._has_reference
    assert restored._reference_parameters.keys() == source._reference_parameters.keys()
    for name in source._reference_parameters:
        assert torch.equal(
            restored._reference_parameters[name],
            source._reference_parameters[name],
        )
        assert torch.equal(
            restored._importance_weights[name],
            source._importance_weights[name],
        )


def test_ewc_fisher_is_finite_positive_and_does_not_consume_torch_rng() -> None:
    agent = make_agent(EWCSACAgent, fisher_batches=2)
    replay = replay_for_task(0)
    rng_before = torch.random.get_rng_state().clone()

    fisher = agent._compute_importance_weights(
        replay_buffer=replay,
        task_index=0,
        batch_size=8,
    )

    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert fisher
    for weight in fisher.values():
        assert torch.isfinite(weight).all()
        assert torch.all(weight >= 1e-5)
        assert torch.count_nonzero(weight > 1e-5) > 0


def test_ssde_dormant_reset_never_changes_owned_weights() -> None:
    agent = make_agent(
        SSDESACAgent,
        dormancy_threshold=1.0,
        dormancy_check_every=10,
    )
    managed = agent._managed_weights[0]
    with torch.no_grad():
        managed.owner[:, : managed.owner.shape[1] // 2] = 0
    before = managed.parameter.detach().clone()
    output_size = managed.parameter.shape[0]
    agent._activation_tracker._sum["hidden_0"] = torch.zeros(output_size)
    agent._activation_tracker._count["hidden_0"] = 1

    reactivated = agent._reactivate_dormant(task_index=1)

    owned = managed.owner == 0
    free = managed.owner == -1
    assert reactivated == output_size
    assert torch.equal(managed.parameter[owned], before[owned])
    assert not torch.equal(managed.parameter[free], before[free])


def test_ssde_tracker_ignores_inference_mode_activations() -> None:
    agent = make_agent(SSDESACAgent, dormancy_check_every=10)
    observations = task_observations(0)

    with torch.inference_mode():
        agent.actor.sample(observations)

    assert agent._activation_tracker._sum == {}
    assert agent._activation_tracker._count == {}

    agent.actor.sample(observations)
    assert agent._activation_tracker._sum


def test_ssde_masks_survive_state_dict_round_trip() -> None:
    source = make_agent(SSDESACAgent, dormancy_check_every=10)
    source.on_task_end(
        task_index=0,
        replay_buffer=replay_for_task(0),
        batch_size=8,
    )
    restored = make_agent(SSDESACAgent, dormancy_check_every=10)
    restored.load_state_dict(copy.deepcopy(source.state_dict()), strict=True)

    for source_weight, restored_weight in zip(
        source._managed_weights,
        restored._managed_weights,
        strict=True,
    ):
        assert torch.equal(source_weight.owner, restored_weight.owner)


def test_full_ssde_initializes_sparse_task_gates_and_overlap() -> None:
    agent = make_full_ssde()
    before = tuple(mask.clone() for mask in agent._cumulative_masks)

    agent.on_task_start(task_index=0, replay_buffer=object())

    assert 0 in agent._task_descriptors
    assert agent.actor._overlap_masks[0]
    assert all(not mask.any() for mask in agent.actor._overlap_masks[0])
    assert all(torch.equal(mask, old) for mask, old in zip(agent._cumulative_masks, before))
    assert any(mask.any() for mask in agent.actor.task_masks(0))


def test_full_ssde_freezes_completed_backbone_connections() -> None:
    agent = make_full_ssde()
    agent.on_task_start(task_index=0, replay_buffer=object())
    agent.on_task_end(task_index=0, replay_buffer=object(), batch_size=8)
    agent.on_task_start(task_index=1, replay_buffer=object())

    names = {id(parameter): name for name, parameter in agent.actor.named_parameters()}
    parameters = tuple(agent.actor.parameters())
    gradients = tuple(torch.ones_like(parameter) for parameter in parameters)
    adjusted = agent.adjust_actor_gradients(
        gradients=gradients,
        parameters=parameters,
        task_index=1,
    )

    saw_frozen = False
    for parameter, gradient in zip(parameters, adjusted, strict=True):
        name = names[id(parameter)]
        if name in agent._parameter_masks:
            expected = agent._parameter_masks[name].to(gradient.dtype)
            assert torch.equal(gradient.cpu(), expected)
            saw_frozen |= bool((~agent._parameter_masks[name]).any())
        elif name.startswith("task_embeddings") or name.startswith("random_task_embeddings"):
            assert torch.count_nonzero(gradient) == 0
    assert saw_frozen


def test_full_ssde_multi_head_output_parameters_remain_available() -> None:
    agent = make_full_ssde()
    agent.on_task_start(task_index=0, replay_buffer=object())
    agent.on_task_end(task_index=0, replay_buffer=object(), batch_size=8)

    assert agent._parameter_masks["mean_head.weight"].all()


def test_full_ssde_state_survives_checkpoint_round_trip() -> None:
    source = make_full_ssde()
    source.on_task_start(task_index=0, replay_buffer=object())
    source.on_task_end(task_index=0, replay_buffer=object(), batch_size=8)
    restored = make_full_ssde()
    restored.load_state_dict(copy.deepcopy(source.state_dict()), strict=True)

    assert restored._task_descriptors.keys() == source._task_descriptors.keys()
    for left, right in zip(source._cumulative_masks, restored._cumulative_masks, strict=True):
        assert torch.equal(left, right)
    for key in source._parameter_masks:
        assert torch.equal(source._parameter_masks[key], restored._parameter_masks[key])
    assert restored.actor._beta_by_task == source.actor._beta_by_task


def test_full_ssde_real_update_is_finite() -> None:
    agent = make_full_ssde(sensitivity_interval=100)
    agent.on_task_start(task_index=0, replay_buffer=object())

    result = agent.update_batch(
        observations=task_observations(0),
        actions=torch.zeros(8, 2),
        rewards=torch.zeros(8, 1),
        next_observations=task_observations(0),
        dones=torch.zeros(8, 1),
    )

    assert result is not None
    assert all(np.isfinite(value) for value in result.values())


def test_full_ssde_random_distillation_updates_actor_only() -> None:
    agent = make_full_ssde(
        random_distillation=True,
        distillation_steps=2,
        distillation_batch_size=8,
    )
    agent.on_task_start(task_index=0, replay_buffer=object())
    agent.on_task_end(task_index=0, replay_buffer=object(), batch_size=8)
    replay = replay_for_task(0)
    actor_before = tuple(parameter.detach().clone() for parameter in agent.actor.parameters())
    critic_before = tuple(parameter.detach().clone() for parameter in agent.critic1.parameters())

    agent.on_task_start(task_index=1, replay_buffer=replay)

    assert any(
        not torch.equal(before, after)
        for before, after in zip(actor_before, agent.actor.parameters(), strict=True)
    )
    assert all(
        torch.equal(before, after)
        for before, after in zip(critic_before, agent.critic1.parameters(), strict=True)
    )


def test_formal_ssde_registration_is_distinct_from_approximation() -> None:
    formal = get_method("ssde")
    approximation = get_method("ssde_approx")

    assert formal.method_id == "ssde"
    assert approximation.method_id == "ssde_approx"
    assert formal.defaults["batch_size"] == 256
    assert formal.defaults["reset_critic_on_task_change"] is True
    assert formal.defaults["exploration_strategy"] == "uniform_previous"
