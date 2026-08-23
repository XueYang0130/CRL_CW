from __future__ import annotations

import numpy as np
import torch

from agents import DemonstrationGuidedFullBCAgent
from methods import get_method


def make_agent() -> DemonstrationGuidedFullBCAgent:
    return DemonstrationGuidedFullBCAgent(
        observation_dim=5,
        action_dim=2,
        action_low=np.full(2, -1.0, dtype=np.float32),
        action_high=np.full(2, 1.0, dtype=np.float32),
        num_tasks=2,
        task_id_dim=2,
        hide_task_id=True,
        device="cpu",
        episodic_batch_size=2,
        actor_cloning_coefficient=100.0,
        bc_gradient_strategy="pcgrad_sac_priority",
        bc_combination_strategy="adaptive_additive",
    )


def task_observation(task_index: int) -> np.ndarray:
    observation = np.zeros(5, dtype=np.float32)
    observation[3 + task_index] = 1.0
    return observation


def perturb_actor(agent: DemonstrationGuidedFullBCAgent) -> None:
    with torch.no_grad():
        for parameter in agent.actor.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.5)


def test_old_task_guide_is_frozen_across_live_actor_updates() -> None:
    torch.manual_seed(4)
    agent = make_agent()
    agent.on_task_start(task_index=0, replay_buffer=object())
    agent.on_task_end(task_index=0, replay_buffer=object(), batch_size=2)
    agent.on_task_start(task_index=1, replay_buffer=object())
    agent.initialize_task_from_guide(task_index=1, guide_task_index=0)

    observation = task_observation(1)
    before = agent.select_guide_action(
        observation,
        guide_task_index=0,
        deterministic=True,
    )
    perturb_actor(agent)
    after = agent.select_guide_action(
        observation,
        guide_task_index=0,
        deterministic=True,
    )
    np.testing.assert_allclose(before, after, atol=0.0, rtol=0.0)


def test_promoted_current_policy_is_frozen_and_uses_current_head() -> None:
    torch.manual_seed(7)
    agent = make_agent()
    agent.on_task_start(task_index=0, replay_buffer=object())
    agent.on_task_end(task_index=0, replay_buffer=object(), batch_size=2)
    agent.on_task_start(task_index=1, replay_buffer=object())
    agent.initialize_task_from_guide(task_index=1, guide_task_index=0)
    observation = task_observation(1)
    expected = agent.select_action(observation, deterministic=True)

    assert agent.promote_current_policy_to_guide(
        task_index=1,
        success_rate=0.2,
        mean_return=10.0,
    )
    promoted = agent.select_guide_action(
        observation,
        guide_task_index=0,
        deterministic=True,
    )
    np.testing.assert_allclose(expected, promoted, atol=0.0, rtol=0.0)

    perturb_actor(agent)
    still_frozen = agent.select_guide_action(
        observation,
        guide_task_index=0,
        deterministic=True,
    )
    np.testing.assert_allclose(promoted, still_frozen, atol=0.0, rtol=0.0)
    assert not agent.promote_current_policy_to_guide(
        task_index=1,
        success_rate=1.0,
        mean_return=100.0,
    )


def test_best_actor_state_can_replace_end_of_task_guide() -> None:
    torch.manual_seed(11)
    agent = make_agent()
    best_state = {
        name: value.detach().cpu().clone()
        for name, value in agent.actor.state_dict().items()
    }
    observation = task_observation(0)
    expected = agent.select_action(observation, deterministic=True)
    perturb_actor(agent)
    agent.set_task_guide_actor_state(task_index=0, actor_state=best_state)
    actual = agent.select_guide_action(
        task_observation(1),
        guide_task_index=0,
        deterministic=True,
    )
    np.testing.assert_allclose(expected, actual, atol=0.0, rtol=0.0)


def test_new_method_does_not_change_adaptive_pcgrad_method_spec() -> None:
    original = get_method("success_replay_best_adaptive_pcgrad")
    jumpstart = get_method("success_replay_best_jumpstart_adaptive_pcgrad")

    assert original.guide_mode == "none"
    assert jumpstart.guide_mode == "demonstration_curriculum"
    for key in (
        "bc_combination_strategy",
        "bc_adaptive_target_ratio",
        "bc_adaptive_conflict_ratio",
        "episodic_memory_per_task",
        "actor_cloning_coefficient",
    ):
        assert jumpstart.defaults[key] == original.defaults[key]
