"""PyTorch reproduction of the authors' official RECALL implementation."""

from __future__ import annotations

import math
from itertools import chain

import numpy as np
import torch
from torch import nn

from agents.gradient_diagnostics import GradientConflictDiagnostics
from agents.networks import (
    DEFAULT_HIDDEN_SIZES,
    LEAKY_RELU_SLOPE,
    SharedMLP,
    initialize_linear_layer,
)
from agents.replay_buffer import ReplayBatch, ReplayBuffer
from agents.sac_agent import SACAgent
from agents.sac_losses import compute_alpha_loss


class RecallPopArtCritic(nn.Module):
    """Official RECALL multi-layer critic head with task-wise PopArt state."""

    def __init__(
        self,
        *,
        observation_dim: int,
        action_dim: int,
        task_id_dim: int,
        num_heads: int,
        hide_task_id: bool,
        hidden_sizes: tuple[int, ...] = DEFAULT_HIDDEN_SIZES,
        beta: float = 3e-4,
    ) -> None:
        super().__init__()
        if len(hidden_sizes) < 2:
            raise ValueError("RECALL critic requires at least two hidden layers.")
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self.task_id_dim = int(task_id_dim)
        self.num_heads = int(num_heads)
        self.hide_task_id = bool(hide_task_id)
        self.beta = float(beta)
        backbone_observation_dim = (
            self.observation_dim - self.task_id_dim
            if self.hide_task_id
            else self.observation_dim
        )
        self.backbone = SharedMLP(
            input_dim=backbone_observation_dim + self.action_dim,
            hidden_sizes=hidden_sizes[:1],
            use_layer_norm=True,
        )
        self.q_heads = nn.ModuleList(
            self._make_head(hidden_sizes[0], hidden_sizes[1:])
            for _ in range(self.num_heads)
        )
        self.register_buffer("moment1", torch.zeros(self.num_heads))
        self.register_buffer("moment2", torch.ones(self.num_heads))
        self.register_buffer("sigma", torch.ones(self.num_heads))

    @staticmethod
    def _make_head(input_dim: int, hidden_sizes: tuple[int, ...]) -> nn.Sequential:
        layers: list[nn.Module] = []
        previous = input_dim
        for hidden in hidden_sizes:
            layer = nn.Linear(previous, hidden)
            initialize_linear_layer(layer)
            layers.extend((layer, nn.LeakyReLU(LEAKY_RELU_SLOPE)))
            previous = hidden
        output = nn.Linear(previous, 1)
        initialize_linear_layer(output)
        layers.append(output)
        return nn.Sequential(*layers)

    def _task_indices(self, observations: torch.Tensor) -> torch.Tensor:
        if self.num_heads == 1:
            return torch.zeros(
                observations.shape[0], dtype=torch.long, device=observations.device
            )
        return observations[:, -self.task_id_dim :].argmax(dim=1)

    def forward(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        task_indices = self._task_indices(observations)
        backbone_observations = (
            observations[:, : -self.task_id_dim]
            if self.hide_task_id and self.task_id_dim > 0
            else observations
        )
        features = self.backbone(torch.cat((backbone_observations, actions), dim=-1))
        values = features.new_zeros((features.shape[0], 1))
        for task_index_tensor in task_indices.unique():
            task_index = int(task_index_tensor.item())
            mask = task_indices == task_index_tensor
            values[mask] = self.q_heads[task_index](features[mask])
        zero_anchor = features.new_zeros(())
        for parameter in self.q_heads.parameters():
            zero_anchor = zero_anchor + parameter.reshape(-1)[0]
        return values + zero_anchor * 0.0

    def normalize(self, values: torch.Tensor, observations: torch.Tensor) -> torch.Tensor:
        indices = self._task_indices(observations)
        return (values.reshape(-1) - self.moment1[indices]) / self.sigma[indices]

    def unnormalize(self, values: torch.Tensor, observations: torch.Tensor) -> torch.Tensor:
        indices = self._task_indices(observations)
        return values.reshape(-1) * self.sigma[indices] + self.moment1[indices]

    @torch.no_grad()
    def update_stats_official(
        self,
        returns: torch.Tensor,
        observations: torch.Tensor,
    ) -> None:
        """Match the official code, including its update input and head rescaling."""
        indices = self._task_indices(observations)
        return_values = returns.reshape(-1)
        old_mean = self.moment1.clone()
        old_sigma = self.sigma.clone()
        new_mean = self.moment1.clone()
        new_second = self.moment2.clone()
        for task_index_tensor in indices.unique():
            task_index = int(task_index_tensor.item())
            task_returns = return_values[indices == task_index_tensor]
            batch_mean = task_returns.mean()
            batch_second = task_returns.square().mean()
            new_mean[task_index] += self.beta * (
                batch_mean - new_mean[task_index]
            )
            new_second[task_index] += self.beta * (
                batch_second - new_second[task_index]
            )
        new_sigma = torch.sqrt(new_second - new_mean.square()).clamp(1e-4, 1e6)
        for task_index_tensor in indices.unique():
            task_index = int(task_index_tensor.item())
            output = self.q_heads[task_index][-1]
            output.weight.mul_(old_sigma[task_index] / new_sigma[task_index])
            output.bias.copy_(
                (output.bias * old_sigma[task_index]
                 + old_mean[task_index] - new_mean[task_index])
                / new_sigma[task_index]
            )
        self.moment1.copy_(new_mean)
        self.moment2.copy_(new_second)
        self.sigma.copy_(new_sigma)


class RECALLAgent(SACAgent):
    """Official-code RECALL: perfect replay, PopArt, and policy distillation."""

    def __init__(
        self,
        *args,
        episodic_memory_per_task: int = 10_000,
        episodic_batch_size: int = 128,
        policy_reg_coef: float = 10.0,
        value_reg_coef: float = 1.0,
        regularize_critic: bool = False,
        popart_beta: float = 3e-4,
        recall_seed: int = 0,
        gradient_diagnostics: bool = False,
        gradient_diagnostics_interval: int = 500,
        gradient_diagnostics_source_batch_size: int = 128,
        gradient_diagnostics_seed: int = 0,
        **kwargs,
    ) -> None:
        self.popart_beta = float(popart_beta)
        super().__init__(*args, **kwargs)
        if self.num_tasks <= 1 or self.task_id_dim != self.num_tasks:
            raise ValueError("RECALL requires task-specific actor and critic heads.")
        if episodic_memory_per_task <= 0 or episodic_batch_size <= 0:
            raise ValueError("RECALL memory and batch sizes must be positive.")
        if not math.isfinite(policy_reg_coef) or policy_reg_coef < 0.0:
            raise ValueError("policy_reg_coef must be finite and non-negative.")
        if not math.isfinite(value_reg_coef) or value_reg_coef < 0.0:
            raise ValueError("value_reg_coef must be finite and non-negative.")
        self.episodic_memory_per_task = int(episodic_memory_per_task)
        self.episodic_batch_size = int(episodic_batch_size)
        self.policy_reg_coef = float(policy_reg_coef)
        self.value_reg_coef = float(value_reg_coef)
        self.regularize_critic = bool(regularize_critic)
        self._historical_replay: list[ReplayBatch] = []
        self._historical_sizes: list[int] = []
        self._expert_observations = torch.empty((0, self.observation_dim))
        self._expert_actions = torch.empty((0, self.action_dim))
        self._expert_means = torch.empty((0, self.action_dim))
        self._expert_log_stds = torch.empty((0, self.action_dim))
        self._expert_q1 = torch.empty((0, 1))
        self._expert_q2 = torch.empty((0, 1))
        self._rng = np.random.default_rng(int(recall_seed))
        self.gradient_diagnostics = GradientConflictDiagnostics(
            actor=self.actor,
            enabled=gradient_diagnostics,
            interval=gradient_diagnostics_interval,
            source_batch_size=gradient_diagnostics_source_batch_size,
            gradient_clip_norm=self.gradient_clip_norm,
            seed=gradient_diagnostics_seed,
        )

    def _make_critic(self) -> nn.Module:
        return RecallPopArtCritic(
            observation_dim=self.observation_dim,
            action_dim=self.action_dim,
            task_id_dim=self.task_id_dim,
            num_heads=self.num_tasks,
            hide_task_id=self.hide_task_id,
            beta=self.popart_beta,
        )

    @property
    def historical_transition_count(self) -> int:
        return int(sum(self._historical_sizes))

    @property
    def expert_state_count(self) -> int:
        return int(self._expert_observations.shape[0])

    # The official get_best_return_head implementation reuses one one-hot
    # array and leaves every previously visited entry enabled.
    official_recall_cumulative_guide_mask = True

    @torch.inference_mode()
    def select_official_recall_guide_action(
        self,
        observation: np.ndarray,
        *,
        head_index: int,
    ) -> np.ndarray:
        observation_array = np.asarray(observation, dtype=np.float32).copy()
        task_id_start = observation_array.shape[0] - self.task_id_dim
        observation_array[task_id_start:] = 0.0
        observation_array[task_id_start : task_id_start + head_index + 1] = 1.0
        return self.actor.act(
            observation_array,
            deterministic=False,
            device=self.device,
        )

    def requires_guide_selection(self, task_index: int) -> bool:
        return task_index > 0

    @torch.no_grad()
    def initialize_task_from_guide(
        self,
        *,
        task_index: int,
        guide_task_index: int,
    ) -> None:
        self._validate_task_index(task_index)
        self._validate_task_index(guide_task_index)
        if guide_task_index >= task_index:
            raise ValueError("RECALL guide must be a previously learned task.")
        action_start = task_index * self.action_dim
        action_end = action_start + self.action_dim
        guide_start = guide_task_index * self.action_dim
        guide_end = guide_start + self.action_dim
        for head in (self.actor.mean_head, self.actor.log_std_head):
            head.weight[action_start:action_end].copy_(head.weight[guide_start:guide_end])
            head.bias[action_start:action_end].copy_(head.bias[guide_start:guide_end])
        for critic in (self.critic1, self.critic2):
            critic.q_heads[task_index].load_state_dict(
                critic.q_heads[guide_task_index].state_dict()
            )
        self.hard_update_targets()

    @torch.no_grad()
    def on_task_end(
        self,
        *,
        task_index: int,
        replay_buffer: ReplayBuffer,
        batch_size: int,
    ) -> None:
        del batch_size
        snapshot = replay_buffer.snapshot(device="cpu")
        observed_indices = snapshot.observations[:, -self.task_id_dim :].argmax(dim=1)
        if not bool(torch.all(observed_indices == task_index)):
            raise RuntimeError("RECALL current replay contains another task's data.")
        self._historical_replay.append(snapshot)
        self._historical_sizes.append(int(snapshot.observations.shape[0]))

        indices = self._rng.integers(
            0,
            snapshot.observations.shape[0],
            size=self.episodic_memory_per_task,
        )
        observations = snapshot.observations[indices].to(self.device)
        actions = snapshot.actions[indices].to(self.device)
        means, log_stds = self.actor.distribution_parameters(observations)
        q1 = self.critic1(observations, actions)
        q2 = self.critic2(observations, actions)
        self._expert_observations = torch.cat(
            (self._expert_observations, observations.cpu())
        )
        self._expert_actions = torch.cat((self._expert_actions, actions.cpu()))
        self._expert_means = torch.cat((self._expert_means, means.cpu()))
        self._expert_log_stds = torch.cat((self._expert_log_stds, log_stds.cpu()))
        self._expert_q1 = torch.cat((self._expert_q1, q1.cpu()))
        self._expert_q2 = torch.cat((self._expert_q2, q2.cpu()))

    def _sample_historical(self, batch_size: int) -> ReplayBatch:
        total = self.historical_transition_count
        if total == 0:
            raise RuntimeError("Cannot sample empty RECALL historical replay.")
        global_indices = self._rng.integers(0, total, size=batch_size)
        boundaries = np.cumsum(self._historical_sizes)
        task_slots = np.searchsorted(boundaries, global_indices, side="right")
        fields: dict[str, list[torch.Tensor]] = {
            name: [] for name in ReplayBatch.__dataclass_fields__
        }
        for slot in np.unique(task_slots):
            positions = np.flatnonzero(task_slots == slot)
            lower = 0 if slot == 0 else int(boundaries[slot - 1])
            local = global_indices[positions] - lower
            batch = self._historical_replay[int(slot)]
            for name in fields:
                fields[name].append(getattr(batch, name)[local])
        # Grouping by task changes order only; all SAC losses are permutation invariant.
        return ReplayBatch(**{name: torch.cat(parts).to(self.device) for name, parts in fields.items()})

    def _sample_expert(self) -> tuple[torch.Tensor, ...]:
        size = self.expert_state_count
        indices = self._rng.integers(0, size, size=self.episodic_batch_size)
        return tuple(
            tensor[indices].to(self.device)
            for tensor in (
                self._expert_observations,
                self._expert_actions,
                self._expert_means,
                self._expert_log_stds,
                self._expert_q1,
                self._expert_q2,
            )
        )

    def drain_gradient_diagnostics(self) -> dict[str, list[dict[str, object]]]:
        return self.gradient_diagnostics.drain()

    def gradient_diagnostics_summary(self) -> dict[str, float | int | bool]:
        return self.gradient_diagnostics.summary()

    def _record_source_task_diagnostics(
        self,
        *,
        sac_gradients: tuple[torch.Tensor, ...],
        actor_parameters: tuple[nn.Parameter, ...],
        current_task: int,
        sac_actor_loss: float,
    ) -> None:
        source_indices = self._expert_observations[:, -self.task_id_dim :].argmax(dim=1)
        for source_task in sorted(int(value) for value in source_indices.unique().tolist()):
            if source_task >= current_task:
                raise RuntimeError("RECALL diagnostics detected current/future memory.")
            candidates = torch.nonzero(
                source_indices == source_task,
                as_tuple=False,
            ).flatten()
            indices = self.gradient_diagnostics.sample_indices(candidates)
            observations = self._expert_observations[indices].to(self.device)
            expert_means = self._expert_means[indices].to(self.device)
            expert_log_stds = self._expert_log_stds[indices].to(self.device)
            current_means, current_log_stds = self.actor.distribution_parameters(observations)
            raw_loss = self._current_to_expert_kl(
                current_means,
                current_log_stds,
                expert_means,
                expert_log_stds,
            ).mean()
            source_gradients = torch.autograd.grad(raw_loss, actor_parameters)
            weighted_gradients = tuple(
                self.policy_reg_coef * gradient for gradient in source_gradients
            )
            self.gradient_diagnostics.record_task_pair(
                sac_gradients=sac_gradients,
                bc_gradients=weighted_gradients,
                current_task_index=current_task,
                source_task_index=source_task,
                raw_bc_loss=float(raw_loss.detach().item()),
                bc_coefficient=self.policy_reg_coef,
                sac_actor_loss=sac_actor_loss,
                reference_memory_states=self.expert_state_count,
                source_memory_states=int(candidates.numel()),
            )

    def _alpha_for_observations(self, observations: torch.Tensor) -> torch.Tensor:
        indices = observations[:, -self.task_id_dim :].argmax(dim=1)
        return self.log_alpha[indices].exp()

    @staticmethod
    def _current_to_expert_kl(
        current_mean: torch.Tensor,
        current_log_std: torch.Tensor,
        expert_mean: torch.Tensor,
        expert_log_std: torch.Tensor,
    ) -> torch.Tensor:
        current_variance = current_log_std.exp().square()
        expert_variance = expert_log_std.exp().square()
        return (
            expert_log_std
            - current_log_std
            + (current_variance + (current_mean - expert_mean).square())
            / (2.0 * expert_variance)
            - 0.5
        ).sum(dim=-1)

    def update_from_replay_buffer(
        self,
        *,
        replay_buffer: ReplayBuffer,
        batch_size: int,
        collect_metrics: bool = True,
    ) -> dict[str, float] | None:
        current = replay_buffer.sample(batch_size, self.device)
        current_task = self._task_index_from_observations(current.observations)
        batches = [current]
        if current_task > 0:
            if self.historical_transition_count == 0:
                raise RuntimeError("RECALL has no historical replay after task 0.")
            batches.append(self._sample_historical(batch_size))
        observations = torch.cat([batch.observations for batch in batches])
        actions = torch.cat([batch.actions for batch in batches])
        rewards = torch.cat([batch.rewards for batch in batches]).reshape(-1)
        next_observations = torch.cat([batch.next_observations for batch in batches])
        dones = torch.cat([batch.terminated for batch in batches]).reshape(-1)
        split = batch_size
        alphas = self._alpha_for_observations(observations)

        with torch.no_grad():
            next_output = self.actor.sample(next_observations)
            target_q = torch.minimum(
                self.target_critic1(next_observations, next_output.sampled_action),
                self.target_critic2(next_observations, next_output.sampled_action),
            ).reshape(-1)
            unnormalized_target_q = self.critic1.unnormalize(
                target_q, next_observations
            )
            raw_backup = rewards + self.gamma * (1.0 - dones) * (
                unnormalized_target_q
                - alphas.detach() * next_output.log_probability.reshape(-1)
            )
            q_backup = self.critic1.normalize(raw_backup, observations).detach()

        q1 = self.critic1(observations, actions).reshape(-1)
        q2 = self.critic2(observations, actions).reshape(-1)
        actor_output = self.actor.sample(observations)
        min_q_pi = torch.minimum(
            self.critic1(observations, actor_output.sampled_action),
            self.critic2(observations, actor_output.sampled_action),
        ).reshape(-1)
        actor_terms = alphas.detach() * actor_output.log_probability.reshape(-1) - min_q_pi
        q1_terms = 0.5 * (q_backup - q1).square()
        q2_terms = 0.5 * (q_backup - q2).square()
        actor_loss = actor_terms[:split].mean()
        q1_loss = q1_terms[:split].mean()
        q2_loss = q2_terms[:split].mean()
        if len(batches) == 2:
            actor_loss = actor_loss + actor_terms[split:].mean()
            q1_loss = q1_loss + q1_terms[split:].mean()
            q2_loss = q2_loss + q2_terms[split:].mean()

        raw_policy_kl = actor_loss.new_zeros(())
        auxiliary_loss = actor_loss.new_zeros(())
        critic_auxiliary = actor_loss.new_zeros(())
        if current_task > 0:
            expert_obs, expert_actions, expert_mean, expert_log_std, expert_q1, expert_q2 = (
                self._sample_expert()
            )
            current_mean, current_log_std = self.actor.distribution_parameters(expert_obs)
            raw_policy_kl = self._current_to_expert_kl(
                current_mean,
                current_log_std,
                expert_mean,
                expert_log_std,
            ).mean()
            auxiliary_loss = self.policy_reg_coef * raw_policy_kl
            if self.regularize_critic:
                critic_auxiliary = self.value_reg_coef * (
                    0.5 * (self.critic1(expert_obs, expert_actions) - expert_q1).square().mean()
                    + 0.5 * (self.critic2(expert_obs, expert_actions) - expert_q2).square().mean()
                )

        total_actor_loss = actor_loss + auxiliary_loss
        total_critic_loss = q1_loss + q2_loss + critic_auxiliary
        active_log_alpha = self.log_alpha[current_task]
        alpha_loss = compute_alpha_loss(
            log_alpha=active_log_alpha,
            log_prob=actor_output.log_probability[:split],
            target_entropy=self.target_entropy,
        )
        actor_parameters = tuple(self.actor.parameters())
        critic_parameters = tuple(chain(self.critic1.parameters(), self.critic2.parameters()))
        collect_diagnostics = bool(
            current_task > 0 and self.gradient_diagnostics.begin_bc_update()
        )
        if collect_diagnostics:
            sac_actor_gradients = torch.autograd.grad(
                actor_loss,
                actor_parameters,
                retain_graph=True,
            )
            policy_kl_gradients = torch.autograd.grad(
                auxiliary_loss,
                actor_parameters,
                retain_graph=True,
            )
        actor_gradients = torch.autograd.grad(total_actor_loss, actor_parameters)
        critic_gradients = torch.autograd.grad(total_critic_loss, critic_parameters)
        alpha_gradients = torch.autograd.grad(alpha_loss, (self.log_alpha,))
        if collect_diagnostics:
            self.gradient_diagnostics.record(
                sac_gradients=sac_actor_gradients,
                bc_gradients=policy_kl_gradients,
                current_task_index=current_task,
                raw_bc_loss=float(raw_policy_kl.detach().item()),
                bc_coefficient=self.policy_reg_coef,
                sac_actor_loss=float(actor_loss.detach().item()),
                reference_memory_states=self.expert_state_count,
                applied_bc_gradients=policy_kl_gradients,
                final_actor_gradients=actor_gradients,
                gradient_strategy="standard",
                projection_applied=False,
                bc_norm_scale=1.0,
                bc_combination_strategy="additive",
                bc_combination_scale=1.0,
            )
            self._record_source_task_diagnostics(
                sac_gradients=sac_actor_gradients,
                actor_parameters=actor_parameters,
                current_task=current_task,
                sac_actor_loss=float(actor_loss.detach().item()),
            )

        # The official repository calls update_stats with the normalized backup
        # after gradients are formed and only updates critic1's PopArt state.
        self.critic1.update_stats_official(q_backup, observations)
        self._apply_gradients(parameters=actor_parameters, gradients=actor_gradients, group_name="actor")
        self._apply_gradients(parameters=critic_parameters, gradients=critic_gradients, group_name="critic")
        self._apply_gradients(parameters=(self.log_alpha,), gradients=alpha_gradients, group_name="alpha")
        self.soft_update_targets()
        if not collect_metrics:
            return None
        return {
            "actor_loss": float(actor_loss.detach().item()),
            "q1_loss": float(q1_loss.detach().item()),
            "q2_loss": float(q2_loss.detach().item()),
            "alpha_loss": float(alpha_loss.detach().item()),
            "alpha": float(active_log_alpha.detach().exp().item()),
            "q1_mean": float(q1.detach().mean().item()),
            "q2_mean": float(q2.detach().mean().item()),
            "q_target_mean": float(q_backup.detach().mean().item()),
            "log_prob_mean": float(actor_output.log_probability.detach().mean().item()),
            "recall_policy_kl": float(auxiliary_loss.detach().item()),
            "recall_historical_transitions": float(self.historical_transition_count),
            "recall_expert_states": float(self.expert_state_count),
        }
