from __future__ import annotations

import numpy as np
import torch
from unittest.mock import patch
import sys

from agents import RECALLAgent, ReplayBuffer
from scripts.run_copy import parse_args


def make_agent(*, gradient_diagnostics: bool = False) -> RECALLAgent:
    return RECALLAgent(
        observation_dim=5,
        action_dim=2,
        action_low=np.full(2, -1.0, dtype=np.float32),
        action_high=np.full(2, 1.0, dtype=np.float32),
        num_tasks=2,
        task_id_dim=2,
        hide_task_id=True,
        initial_log_alpha=0.0,
        gradient_clip_norm=0.1,
        episodic_memory_per_task=10,
        episodic_batch_size=4,
        policy_reg_coef=10.0,
        recall_seed=7,
        gradient_diagnostics=gradient_diagnostics,
        gradient_diagnostics_interval=1,
        gradient_diagnostics_source_batch_size=4,
    )


def fill_replay(replay: ReplayBuffer, *, task_index: int, count: int) -> None:
    for index in range(count):
        observation = np.asarray(
            [index / 10.0, 0.2, -0.1, float(task_index == 0), float(task_index == 1)],
            dtype=np.float32,
        )
        next_observation = observation.copy()
        next_observation[0] += 0.01
        replay.add(
            observation=observation,
            action=np.asarray([0.1, -0.2], dtype=np.float32),
            reward=float(index),
            next_observation=next_observation,
            terminated=False,
        )


def test_recall_archives_full_replay_and_builds_fixed_expert_memory() -> None:
    agent = make_agent()
    replay = ReplayBuffer(observation_dim=5, action_dim=2, capacity=20, seed=3)
    fill_replay(replay, task_index=0, count=8)

    agent.on_task_end(task_index=0, replay_buffer=replay, batch_size=4)
    replay.clear()

    assert agent.historical_transition_count == 8
    assert agent.expert_state_count == 10
    assert len(replay) == 0
    assert torch.all(agent._expert_observations[:, -2] == 1.0)
    assert torch.all(agent._expert_observations[:, -1] == 0.0)


def test_recall_task_one_update_uses_historical_replay_and_distillation() -> None:
    agent = make_agent()
    replay = ReplayBuffer(observation_dim=5, action_dim=2, capacity=20, seed=3)
    fill_replay(replay, task_index=0, count=8)
    agent.on_task_end(task_index=0, replay_buffer=replay, batch_size=4)
    replay.clear()
    fill_replay(replay, task_index=1, count=8)

    metrics = agent.update_from_replay_buffer(
        replay_buffer=replay,
        batch_size=4,
        collect_metrics=True,
    )

    assert metrics is not None
    assert metrics["recall_historical_transitions"] == 8.0
    assert metrics["recall_expert_states"] == 10.0
    assert np.isfinite(metrics["recall_policy_kl"])
    assert metrics["recall_policy_kl"] >= -1e-6


def test_recall_copies_actor_and_carried_critic_heads() -> None:
    agent = make_agent()
    with torch.no_grad():
        agent.actor.mean_head.weight[:2].fill_(0.25)
        agent.actor.mean_head.bias[:2].fill_(0.5)
        for parameter in agent.critic1.q_heads[0].parameters():
            parameter.fill_(0.125)

    agent.initialize_task_from_guide(task_index=1, guide_task_index=0)

    assert torch.equal(agent.actor.mean_head.weight[2:4], agent.actor.mean_head.weight[:2])
    assert torch.equal(agent.actor.mean_head.bias[2:4], agent.actor.mean_head.bias[:2])
    for source, target in zip(
        agent.critic1.q_heads[0].parameters(),
        agent.critic1.q_heads[1].parameters(),
        strict=True,
    ):
        assert torch.equal(source, target)


def test_recall_kl_direction_matches_current_to_expert() -> None:
    current_mean = torch.tensor([[0.2, -0.1]])
    current_log_std = torch.log(torch.tensor([[0.5, 1.2]]))
    expert_mean = torch.tensor([[0.0, 0.3]])
    expert_log_std = torch.log(torch.tensor([[1.0, 0.7]]))

    actual = RECALLAgent._current_to_expert_kl(
        current_mean,
        current_log_std,
        expert_mean,
        expert_log_std,
    )
    expected = torch.distributions.kl_divergence(
        torch.distributions.Normal(current_mean, current_log_std.exp()),
        torch.distributions.Normal(expert_mean, expert_log_std.exp()),
    ).sum(dim=-1)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_recall_cli_uses_official_run_defaults() -> None:
    with patch.object(
        sys,
        "argv",
        ["run.py", "--mode", "continual", "--method", "recall"],
    ):
        args = parse_args()

    assert args.batch_size == 128
    assert args.episodic_memory_per_task == 10_000
    assert args.episodic_batch_size == 128
    assert args.actor_cloning_coefficient == 10.0
    assert args.gradient_clip_norm == 0.1
    assert args.best_return_eval_episodes == 10
    assert args.reset_buffer_on_task_change
    assert args.reset_optimizer_on_task_change
    assert args.gradient_diagnostics


def test_recall_gradient_diagnostics_are_read_only_and_source_attributed() -> None:
    agent = make_agent(gradient_diagnostics=True)
    replay = ReplayBuffer(observation_dim=5, action_dim=2, capacity=20, seed=3)
    fill_replay(replay, task_index=0, count=8)
    agent.on_task_end(task_index=0, replay_buffer=replay, batch_size=4)
    replay.clear()
    fill_replay(replay, task_index=1, count=8)

    agent.update_from_replay_buffer(
        replay_buffer=replay,
        batch_size=4,
        collect_metrics=False,
    )
    rows = agent.drain_gradient_diagnostics()

    assert len(rows["windows"]) == 2
    assert len(rows["layers"]) > 0
    assert len(rows["task_pairs"]) == 1
    assert rows["windows"][0]["gradient_strategy"] == "standard"
    assert rows["windows"][0]["bc_combination_strategy"] == "additive"
    assert rows["windows"][0]["projection_applied"] is False
    assert rows["task_pairs"][0]["source_task_index"] == 0


def test_enabling_recall_diagnostics_does_not_change_the_update() -> None:
    plain = make_agent(gradient_diagnostics=False)
    diagnosed = make_agent(gradient_diagnostics=True)
    diagnosed.load_state_dict(plain.state_dict())
    plain_replay = ReplayBuffer(observation_dim=5, action_dim=2, capacity=20, seed=3)
    diagnosed_replay = ReplayBuffer(observation_dim=5, action_dim=2, capacity=20, seed=3)
    fill_replay(plain_replay, task_index=0, count=8)
    fill_replay(diagnosed_replay, task_index=0, count=8)
    plain.on_task_end(task_index=0, replay_buffer=plain_replay, batch_size=4)
    diagnosed.on_task_end(task_index=0, replay_buffer=diagnosed_replay, batch_size=4)
    plain_replay.clear()
    diagnosed_replay.clear()
    fill_replay(plain_replay, task_index=1, count=8)
    fill_replay(diagnosed_replay, task_index=1, count=8)

    torch.manual_seed(1234)
    plain.update_from_replay_buffer(
        replay_buffer=plain_replay,
        batch_size=4,
        collect_metrics=False,
    )
    torch.manual_seed(1234)
    diagnosed.update_from_replay_buffer(
        replay_buffer=diagnosed_replay,
        batch_size=4,
        collect_metrics=False,
    )

    for name, expected in plain.state_dict().items():
        assert torch.equal(expected, diagnosed.state_dict()[name]), name
