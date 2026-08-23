"""Discovery-focused SAC agent with BC-preserving actor updates.

The actor side is intentionally inherited from Adaptive PCGrad.  This class
only changes current-task discovery and critic consolidation:

* a bootstrap critic ensemble estimates epistemic disagreement;
* an upper-confidence score selects among stochastic actor candidates during
  training interaction only;
* prioritized n-step replay updates all critics and writes TD priorities back.

Deterministic and stochastic evaluations use the ordinary actor policy, so the
reported policy is never the UCB search policy.
"""

from __future__ import annotations

import copy
from itertools import chain
import math
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from agents.full_bc_agent import FullBehaviorCloningSACAgent
from agents.prioritized_nstep_replay import PrioritizedNStepReplayBuffer
from agents.sac_losses import compute_actor_loss, compute_alpha_loss


class OptimisticEnsembleFullBCAgent(FullBehaviorCloningSACAgent):
    """Adaptive-PCGrad actor with ensemble-UCB exploration and PER n-step TD."""

    def __init__(
        self,
        *args,
        critic_ensemble_size: int = 4,
        critic_bootstrap_probability: float = 0.8,
        optimistic_ucb_beta: float = 0.5,
        optimistic_action_candidates: int = 8,
        ensemble_seed: int = 0,
        **kwargs,
    ) -> None:
        if critic_ensemble_size < 2:
            raise ValueError("critic_ensemble_size must be at least two.")
        if (
            not math.isfinite(critic_bootstrap_probability)
            or not 0.0 < critic_bootstrap_probability <= 1.0
        ):
            raise ValueError(
                "critic_bootstrap_probability must be finite and in (0, 1]."
            )
        if not math.isfinite(optimistic_ucb_beta) or optimistic_ucb_beta < 0.0:
            raise ValueError("optimistic_ucb_beta must be finite and non-negative.")
        if optimistic_action_candidates <= 0:
            raise ValueError("optimistic_action_candidates must be positive.")

        super().__init__(*args, **kwargs)
        self.critic_ensemble_size = int(critic_ensemble_size)
        self.critic_bootstrap_probability = float(critic_bootstrap_probability)
        self.optimistic_ucb_beta = float(optimistic_ucb_beta)
        self.optimistic_action_candidates = int(optimistic_action_candidates)
        self._ensemble_rng = np.random.default_rng(ensemble_seed)

        # Start auxiliary members from the two standard SAC critics.  They are
        # identical at initialization and become diverse only through bootstrap
        # masks, avoiding arbitrary random-target transients.
        self.extra_critics = nn.ModuleList(
            [
                copy.deepcopy(self.critic1 if index % 2 == 0 else self.critic2)
                for index in range(self.critic_ensemble_size - 2)
            ]
        ).to(self.device)
        self.extra_target_critics = nn.ModuleList(
            [
                copy.deepcopy(
                    self.target_critic1 if index % 2 == 0 else self.target_critic2
                )
                for index in range(self.critic_ensemble_size - 2)
            ]
        ).to(self.device)
        for target in self.extra_target_critics:
            target.requires_grad_(False)
            target.eval()

        self._ucb_action_calls = 0
        self._ucb_uncertainty_sum = 0.0
        self._last_replay_diagnostics: dict[str, float | int] = {}
        self.rebuild_optimizer()

    @property
    def ensemble_critics(self) -> tuple[nn.Module, ...]:
        return (self.critic1, self.critic2, *tuple(self.extra_critics))

    @property
    def ensemble_target_critics(self) -> tuple[nn.Module, ...]:
        return (
            self.target_critic1,
            self.target_critic2,
            *tuple(self.extra_target_critics),
        )

    def _build_optimizer(self) -> Adam:
        # SACAgent calls this during its constructor, before auxiliary critics
        # exist.  The second call at the end of this class's constructor adds
        # every ensemble member.
        if not hasattr(self, "extra_critics"):
            return super()._build_optimizer()
        return Adam(
            chain(
                self.actor.parameters(),
                *(critic.parameters() for critic in self.ensemble_critics),
                [self.log_alpha],
            ),
            lr=self.learning_rate,
            eps=1e-7,
        )

    @torch.inference_mode()
    def select_action(
        self,
        observation: np.ndarray | Sequence[float],
        *,
        deterministic: bool = False,
    ) -> np.ndarray:
        # Evaluation calls Module.eval().  Keeping UCB out of evaluation makes
        # success/return directly comparable with all existing baselines.
        if (
            deterministic
            or not self.training
            or self.optimistic_action_candidates == 1
            or self.optimistic_ucb_beta == 0.0
        ):
            return super().select_action(
                observation,
                deterministic=deterministic,
            )

        observation_array = np.asarray(observation, dtype=np.float32)
        if observation_array.shape != (self.observation_dim,):
            raise ValueError(
                "Expected observation shape "
                f"{(self.observation_dim,)}, received {observation_array.shape}."
            )
        observations = torch.as_tensor(
            observation_array,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0).repeat(self.optimistic_action_candidates, 1)
        actions = self.actor.sample(observations).sampled_action
        q_values = torch.stack(
            [critic(observations, actions).reshape(-1) for critic in self.ensemble_critics],
            dim=0,
        )
        q_mean = q_values.mean(dim=0)
        q_std = q_values.std(dim=0, unbiased=False)
        scores = q_mean + self.optimistic_ucb_beta * q_std
        selected = int(torch.argmax(scores).item())
        self._ucb_action_calls += 1
        self._ucb_uncertainty_sum += float(q_std[selected].item())
        return actions[selected].cpu().numpy()

    def update_from_replay_buffer(
        self,
        *,
        replay_buffer: object,
        batch_size: int,
        collect_metrics: bool,
    ) -> dict[str, float] | None:
        """Sample, update, and feed fresh TD errors back into PER."""
        if not isinstance(replay_buffer, PrioritizedNStepReplayBuffer):
            raise TypeError(
                "OptimisticEnsembleFullBCAgent requires "
                "PrioritizedNStepReplayBuffer."
            )
        batch = replay_buffer.sample(batch_size=batch_size, device=self.device)
        metrics, priorities = self._update_prioritized_batch(
            observations=batch.observations,
            actions=batch.actions,
            rewards=batch.rewards,
            next_observations=batch.next_observations,
            dones=batch.terminated,
            discounts=batch.discounts,
            importance_weights=batch.importance_weights,
            priority_beta=batch.priority_beta,
            collect_metrics=collect_metrics,
        )
        replay_buffer.update_priorities(batch.indices, priorities)
        self._last_replay_diagnostics = replay_buffer.diagnostics()
        return metrics

    def on_task_start(self, *, task_index: int, replay_buffer: object) -> None:
        super().on_task_start(task_index=task_index, replay_buffer=replay_buffer)
        self._ucb_action_calls = 0
        self._ucb_uncertainty_sum = 0.0
        self._last_replay_diagnostics = {}

    def _update_prioritized_batch(
        self,
        *,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
        discounts: torch.Tensor,
        importance_weights: torch.Tensor,
        priority_beta: float,
        collect_metrics: bool,
    ) -> tuple[dict[str, float] | None, np.ndarray]:
        (
            observations,
            actions,
            rewards,
            next_observations,
            dones,
        ) = self._prepare_update_batch(
            observations=observations,
            actions=actions,
            rewards=rewards,
            next_observations=next_observations,
            dones=dones,
        )
        discounts = discounts.to(self.device, dtype=torch.float32).reshape(-1)
        importance_weights = importance_weights.to(
            self.device, dtype=torch.float32
        ).reshape(-1)
        batch_size = observations.shape[0]
        if discounts.shape != (batch_size,):
            raise ValueError("discounts must contain one scalar per replay item.")
        if importance_weights.shape != (batch_size,):
            raise ValueError(
                "importance_weights must contain one scalar per replay item."
            )
        if not bool(torch.isfinite(discounts).all()) or torch.any(discounts < 0.0):
            raise ValueError("discounts must be finite and non-negative.")
        if (
            not bool(torch.isfinite(importance_weights).all())
            or torch.any(importance_weights <= 0.0)
            or torch.any(importance_weights > 1.0 + 1e-6)
        ):
            raise ValueError("importance weights must be finite and in (0, 1].")

        task_index = self._task_index_from_observations(observations)
        if self._task_index_from_observations(next_observations) != task_index:
            raise RuntimeError(
                "Current and next observations contain different task IDs."
            )
        active_log_alpha = self.log_alpha[task_index]
        active_alpha = active_log_alpha.exp()

        with torch.no_grad():
            next_actor_output = self.actor.sample(next_observations)
            # Keep the Bellman target anchored to the original twin-Q SAC
            # estimate.  Auxiliary critics estimate uncertainty without
            # changing the target definition used by the baseline.
            next_q1 = self.target_critic1(
                next_observations, next_actor_output.sampled_action
            ).reshape(-1)
            next_q2 = self.target_critic2(
                next_observations, next_actor_output.sampled_action
            ).reshape(-1)
            q_targets = (
                rewards.reshape(-1)
                + discounts
                * (1.0 - dones.reshape(-1))
                * (
                    torch.minimum(next_q1, next_q2)
                    - active_alpha.detach()
                    * next_actor_output.log_probability.reshape(-1)
                )
            ).detach()

        q_predictions = tuple(
            critic(observations, actions).reshape(-1)
            for critic in self.ensemble_critics
        )
        critic_losses: list[torch.Tensor] = []
        for critic_index, prediction in enumerate(q_predictions):
            per_item = 0.5 * (prediction - q_targets).square()
            if critic_index < 2:
                mask = torch.ones_like(per_item)
            else:
                mask_values = self._ensemble_rng.random(batch_size) < (
                    self.critic_bootstrap_probability
                )
                if not bool(mask_values.any()):
                    mask_values[int(self._ensemble_rng.integers(batch_size))] = True
                mask = torch.as_tensor(
                    mask_values,
                    dtype=torch.float32,
                    device=self.device,
                )
            weighted_mask = importance_weights * mask
            critic_losses.append(
                (per_item * weighted_mask).sum()
                / weighted_mask.sum().clamp_min(1e-12)
            )
        critic_loss = torch.stack(critic_losses).sum()

        actor_output = self.actor.sample(observations)
        q1_policy = self.critic1(observations, actor_output.sampled_action)
        q2_policy = self.critic2(observations, actor_output.sampled_action)
        actor_loss = compute_actor_loss(
            log_prob=actor_output.log_probability,
            q1_policy=q1_policy,
            q2_policy=q2_policy,
            alpha=active_alpha,
        )
        alpha_loss = compute_alpha_loss(
            log_alpha=active_log_alpha,
            log_prob=actor_output.log_probability,
            target_entropy=self.target_entropy,
        )

        actor_parameters = tuple(self.actor.parameters())
        critic_parameter_groups = tuple(
            tuple(critic.parameters()) for critic in self.ensemble_critics
        )
        critic_parameters = tuple(chain(*critic_parameter_groups))
        actor_gradients = torch.autograd.grad(actor_loss, actor_parameters)
        critic_gradients = torch.autograd.grad(critic_loss, critic_parameters)
        alpha_gradients = torch.autograd.grad(alpha_loss, (self.log_alpha,))

        self._collecting_update_metrics = bool(collect_metrics)
        self._current_actor_loss = actor_loss.detach()
        actor_gradients = self.adjust_actor_gradients(
            gradients=actor_gradients,
            parameters=actor_parameters,
            task_index=task_index,
        )
        critic_gradients = self.adjust_critic_gradients(
            gradients=critic_gradients,
            parameters=critic_parameters,
            task_index=task_index,
        )
        alpha_gradients = self.adjust_alpha_gradients(
            gradients=alpha_gradients,
            parameters=(self.log_alpha,),
            task_index=task_index,
        )
        self._apply_gradients(
            parameters=actor_parameters,
            gradients=actor_gradients,
            group_name="actor",
        )
        self._apply_critic_ensemble_gradients(
            parameter_groups=critic_parameter_groups,
            gradients=critic_gradients,
        )
        self._apply_gradients(
            parameters=(self.log_alpha,),
            gradients=alpha_gradients,
            group_name="alpha",
        )
        self.soft_update_targets()

        td_errors = torch.maximum(
            (q_predictions[0] - q_targets).abs(),
            (q_predictions[1] - q_targets).abs(),
        )
        priorities = td_errors.detach().cpu().numpy().astype(np.float64, copy=False)
        if not collect_metrics:
            return None, priorities

        ensemble_values = torch.stack(q_predictions, dim=0).detach()
        metrics = {
            "actor_loss": float(actor_loss.detach().item()),
            "q1_loss": float(critic_losses[0].detach().item()),
            "q2_loss": float(critic_losses[1].detach().item()),
            "ensemble_critic_loss": float(critic_loss.detach().item()),
            "alpha_loss": float(alpha_loss.detach().item()),
            "alpha": float(active_alpha.detach().item()),
            "q1_mean": float(q_predictions[0].detach().mean().item()),
            "q2_mean": float(q_predictions[1].detach().mean().item()),
            "q_target_mean": float(q_targets.mean().item()),
            "q_ensemble_std_mean": float(
                ensemble_values.std(dim=0, unbiased=False).mean().item()
            ),
            "td_error_abs_mean": float(td_errors.mean().item()),
            "priority_beta": float(priority_beta),
            "importance_weight_mean": float(importance_weights.mean().item()),
            "log_prob_mean": float(
                actor_output.log_probability.detach().mean().item()
            ),
        }
        return metrics, priorities

    def _apply_critic_ensemble_gradients(
        self,
        *,
        parameter_groups: tuple[tuple[nn.Parameter, ...], ...],
        gradients: tuple[torch.Tensor, ...],
    ) -> None:
        """Step all critics without changing the base twin-Q clip scale.

        The original SAC clips critic 1 and critic 2 as one group.  Including
        auxiliary critics in that same norm would shrink the two base critics
        merely because the ensemble is larger.  We therefore retain the exact
        twin-Q group and clip auxiliary members in independent pairs.
        """
        parameters = tuple(chain(*parameter_groups))
        if len(parameters) != len(gradients):
            raise ValueError("Critic parameters and gradients must align.")
        for gradient in gradients:
            if gradient is None or not bool(torch.isfinite(gradient).all()):
                raise FloatingPointError(
                    "A critic-ensemble gradient is missing or non-finite."
                )

        adjusted = list(gradients)
        if self.gradient_clip_norm is not None:
            offset = 0
            indexed_groups: list[tuple[int, ...]] = []
            for group in parameter_groups:
                indexed_groups.append(tuple(range(offset, offset + len(group))))
                offset += len(group)
            for critic_pair_start in range(0, len(indexed_groups), 2):
                indices = tuple(
                    index
                    for group in indexed_groups[
                        critic_pair_start : critic_pair_start + 2
                    ]
                    for index in group
                )
                total_norm = torch.linalg.vector_norm(
                    torch.stack(
                        [torch.linalg.vector_norm(adjusted[index]) for index in indices]
                    )
                )
                scale = min(
                    1.0,
                    self.gradient_clip_norm / (float(total_norm.detach()) + 1e-6),
                )
                for index in indices:
                    adjusted[index] = adjusted[index] * scale

        self.optimizer.zero_grad(set_to_none=True)
        for parameter, gradient in zip(parameters, adjusted, strict=True):
            parameter.grad = gradient.detach()
        self.optimizer.step()

    @torch.no_grad()
    def hard_update_targets(self) -> None:
        super().hard_update_targets()
        if hasattr(self, "extra_critics"):
            for critic, target in zip(
                self.extra_critics,
                self.extra_target_critics,
                strict=True,
            ):
                target.load_state_dict(critic.state_dict(), strict=True)

    @torch.no_grad()
    def soft_update_targets(self) -> None:
        super().soft_update_targets()
        if hasattr(self, "extra_critics"):
            for critic, target in zip(
                self.extra_critics,
                self.extra_target_critics,
                strict=True,
            ):
                self._soft_update(online=critic, target=target)

    def reset_critics(self) -> None:
        super().reset_critics()
        if not hasattr(self, "extra_critics"):
            return
        for index, (critic, target) in enumerate(
            zip(self.extra_critics, self.extra_target_critics, strict=True)
        ):
            source = self.critic1 if index % 2 == 0 else self.critic2
            critic.load_state_dict(source.state_dict(), strict=True)
            target.load_state_dict(source.state_dict(), strict=True)
            target.requires_grad_(False)
            target.eval()

    def initialize_from_base_agent(
        self,
        base_agent: FullBehaviorCloningSACAgent,
    ) -> None:
        """Upgrade a baseline checkpoint for a controlled causal probe."""
        if (
            base_agent.observation_dim != self.observation_dim
            or base_agent.action_dim != self.action_dim
            or base_agent.num_tasks != self.num_tasks
            or base_agent.task_id_dim != self.task_id_dim
        ):
            raise ValueError("Base and ensemble agents have incompatible dimensions.")
        self.actor.load_state_dict(base_agent.actor.state_dict(), strict=True)
        self.critic1.load_state_dict(base_agent.critic1.state_dict(), strict=True)
        self.critic2.load_state_dict(base_agent.critic2.state_dict(), strict=True)
        self.target_critic1.load_state_dict(
            base_agent.target_critic1.state_dict(), strict=True
        )
        self.target_critic2.load_state_dict(
            base_agent.target_critic2.state_dict(), strict=True
        )
        with torch.no_grad():
            self.log_alpha.copy_(base_agent.log_alpha)
        for index, (critic, target) in enumerate(
            zip(self.extra_critics, self.extra_target_critics, strict=True)
        ):
            online_source = base_agent.critic1 if index % 2 == 0 else base_agent.critic2
            target_source = (
                base_agent.target_critic1
                if index % 2 == 0
                else base_agent.target_critic2
            )
            critic.load_state_dict(online_source.state_dict(), strict=True)
            target.load_state_dict(target_source.state_dict(), strict=True)
        self.rebuild_optimizer()

    def task_diagnostics(self) -> dict[str, float | int | None]:
        diagnostics = dict(super().task_diagnostics())
        diagnostics.update(self._last_replay_diagnostics)
        diagnostics.update(
            {
                "critic_ensemble_size": self.critic_ensemble_size,
                "optimistic_ucb_beta": self.optimistic_ucb_beta,
                "optimistic_action_candidates": self.optimistic_action_candidates,
                "ucb_action_calls": self._ucb_action_calls,
                "mean_selected_q_uncertainty": (
                    None
                    if self._ucb_action_calls == 0
                    else self._ucb_uncertainty_sum / self._ucb_action_calls
                ),
            }
        )
        return diagnostics
