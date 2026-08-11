from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from agents import LOG_STD_MAX, LOG_STD_MIN, SACAgent


@dataclass(frozen=True)
class PriorInitializationResult:
    initial_kl: float
    final_kl: float
    updates: int
    reset_states: int


def collect_reset_observations(env: object, *, count: int, seed: int) -> np.ndarray:
    if count <= 0:
        raise ValueError("reset observation count must be positive.")
    observations: list[np.ndarray] = []
    for index in range(count):
        reset_result = env.reset(seed=seed + index)
        observation = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        observations.append(np.asarray(observation, dtype=np.float32).copy())
    return np.stack(observations)


def _gaussian_kl(
    teacher_mean: torch.Tensor,
    teacher_log_std: torch.Tensor,
    student_mean: torch.Tensor,
    student_log_std: torch.Tensor,
) -> torch.Tensor:
    epsilon = 1e-6
    teacher_variance = teacher_log_std.exp().square() + epsilon
    student_variance = student_log_std.exp().square() + epsilon
    return (
        student_log_std
        - teacher_log_std
        + (teacher_variance + (teacher_mean - student_mean).square())
        / (2.0 * student_variance)
        - 0.5
    ).sum(dim=-1)


def initialize_actor_head_from_source(
    agent: SACAgent,
    *,
    observations: np.ndarray,
    source_task_index: int,
    target_task_index: int,
    updates: int,
    learning_rate: float,
) -> PriorInitializationResult:
    if agent.task_id_dim != agent.num_tasks:
        raise ValueError("implicit prior initialization requires task one-hot heads.")
    if not 0 <= source_task_index < target_task_index < agent.num_tasks:
        raise ValueError("prior initialization requires source < target task index.")
    if updates <= 0 or learning_rate <= 0.0:
        raise ValueError("prior updates and learning rate must be positive.")
    observation_tensor = torch.as_tensor(
        observations, dtype=torch.float32, device=agent.device
    )
    if observation_tensor.ndim != 2 or observation_tensor.shape[1] != agent.observation_dim:
        raise ValueError("reset observations have the wrong shape.")

    source_observations = observation_tensor.clone()
    source_observations[:, -agent.task_id_dim :] = 0.0
    source_observations[:, -agent.task_id_dim + source_task_index] = 1.0
    target_observations = observation_tensor.clone()
    target_observations[:, -agent.task_id_dim :] = 0.0
    target_observations[:, -agent.task_id_dim + target_task_index] = 1.0

    actor = agent.actor
    with torch.no_grad():
        features_input, _ = actor._split_observation(target_observations)
        features = actor.backbone(features_input).detach()
        teacher_mean, teacher_log_std = actor.distribution_parameters(source_observations)

    student_mean = nn.Linear(actor.backbone.output_dim, agent.action_dim).to(agent.device)
    student_log_std = nn.Linear(actor.backbone.output_dim, agent.action_dim).to(agent.device)
    with torch.no_grad():
        student_mean.weight.copy_(actor.mean_head.weight[target_task_index::agent.num_tasks])
        student_mean.bias.copy_(actor.mean_head.bias[target_task_index::agent.num_tasks])
        student_log_std.weight.copy_(actor.log_std_head.weight[target_task_index::agent.num_tasks])
        student_log_std.bias.copy_(actor.log_std_head.bias[target_task_index::agent.num_tasks])

    optimizer = torch.optim.Adam(
        [*student_mean.parameters(), *student_log_std.parameters()],
        lr=learning_rate,
    )

    def loss_value() -> torch.Tensor:
        mean = student_mean(features)
        log_std = torch.clamp(student_log_std(features), LOG_STD_MIN, LOG_STD_MAX)
        return _gaussian_kl(teacher_mean, teacher_log_std, mean, log_std).mean()

    initial_kl = float(loss_value().detach().item())
    for _ in range(updates):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_value()
        loss.backward()
        optimizer.step()
    final_kl = float(loss_value().detach().item())

    with torch.no_grad():
        actor.mean_head.weight[target_task_index::agent.num_tasks].copy_(student_mean.weight)
        actor.mean_head.bias[target_task_index::agent.num_tasks].copy_(student_mean.bias)
        actor.log_std_head.weight[target_task_index::agent.num_tasks].copy_(student_log_std.weight)
        actor.log_std_head.bias[target_task_index::agent.num_tasks].copy_(student_log_std.bias)

    return PriorInitializationResult(
        initial_kl=initial_kl,
        final_kl=final_kl,
        updates=updates,
        reset_states=int(observation_tensor.shape[0]),
    )
