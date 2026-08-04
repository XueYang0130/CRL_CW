"""Soft Actor-Critic agent with shared backbones and task-specific heads."""

from __future__ import annotations

import copy
import math
from itertools import chain
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from .networks import GaussianActor, QCritic
from .sac_losses import (
    compute_actor_loss,
    compute_alpha_loss,
    compute_critic_losses,
    compute_q_target,
)


class SACAgent(nn.Module):
    """Multi-head SAC with task-specific entropy coefficients."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        action_low: np.ndarray | Sequence[float],
        action_high: np.ndarray | Sequence[float],
        *,
        num_tasks: int = 1,
        task_id_dim: int = 0,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        polyak: float = 0.995,
        target_entropy: float | None = None,
        initial_log_alpha: float = 1.0,
        device: str | torch.device | None = None,
        hide_task_id: bool = False,
        gradient_clip_norm: float | None = None,
    ) -> None:
        super().__init__()

        self._validate_hyperparameters(
            observation_dim=observation_dim,
            action_dim=action_dim,
            num_tasks=num_tasks,
            task_id_dim=task_id_dim,
            learning_rate=learning_rate,
            gamma=gamma,
            polyak=polyak,
            initial_log_alpha=initial_log_alpha,
            target_entropy=target_entropy,
        )

        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self.num_tasks = int(num_tasks)
        self.task_id_dim = int(task_id_dim)
        self.learning_rate = float(learning_rate)
        self.gamma = float(gamma)
        self.polyak = float(polyak)
        self.hide_task_id = bool(hide_task_id)
        if gradient_clip_norm is not None and (
            not math.isfinite(gradient_clip_norm)
            or gradient_clip_norm <= 0.0
        ):
            raise ValueError(
                "gradient_clip_norm must be finite and positive when provided."
            )
        self.gradient_clip_norm = gradient_clip_norm

        resolved_device = torch.device(device or "cpu")

        action_low_array = np.asarray(action_low, dtype=np.float32)
        action_high_array = np.asarray(action_high, dtype=np.float32)
        self._validate_action_bounds(
            action_low=action_low_array,
            action_high=action_high_array,
        )

        self.actor = GaussianActor(
            observation_dim=self.observation_dim,
            action_low=action_low_array,
            action_high=action_high_array,
            task_id_dim=self.task_id_dim,
            num_heads=self.num_tasks,
            hide_task_id=self.hide_task_id,
        ).to(resolved_device)

        self.critic1 = QCritic(
            observation_dim=self.observation_dim,
            action_dim=self.action_dim,
            task_id_dim=self.task_id_dim,
            num_heads=self.num_tasks,
            hide_task_id=self.hide_task_id,
        ).to(resolved_device)

        self.critic2 = QCritic(
            observation_dim=self.observation_dim,
            action_dim=self.action_dim,
            task_id_dim=self.task_id_dim,
            num_heads=self.num_tasks,
            hide_task_id=self.hide_task_id,
        ).to(resolved_device)

        if self.actor.num_heads != self.num_tasks:
            raise RuntimeError(
                "Actor head count does not match num_tasks."
            )

        if self.critic1.num_heads != self.num_tasks:
            raise RuntimeError(
                "Critic 1 head count does not match num_tasks."
            )

        if self.critic2.num_heads != self.num_tasks:
            raise RuntimeError(
                "Critic 2 head count does not match num_tasks."
            )

        self.target_critic1 = copy.deepcopy(self.critic1).to(
            resolved_device
        )
        self.target_critic2 = copy.deepcopy(self.critic2).to(
            resolved_device
        )
        self._freeze_target_critics()

        self.log_alpha = nn.Parameter(
            torch.full(
                (self.num_tasks,),
                float(initial_log_alpha),
                dtype=torch.float32,
                device=resolved_device,
            )
        )

        self.target_entropy = float(
            -self.action_dim
            if target_entropy is None
            else target_entropy
        )

        self.optimizer = self._build_optimizer()

    def reset_optimizer_state(self) -> None:
        """Reset Adam moments while retaining current parameters."""
        self.optimizer.state.clear()

    def rebuild_optimizer(self) -> None:
        """Recreate the Adam optimizer for the current modules."""
        self.optimizer = self._build_optimizer()

    def copy_alpha(self, *, source_task_index: int, target_task_index: int) -> None:
        """Copy one task entropy parameter to another task."""
        self._validate_task_index(source_task_index)
        self._validate_task_index(target_task_index)
        with torch.no_grad():
            self.log_alpha[target_task_index].copy_(
                self.log_alpha[source_task_index]
            )

    def reset_actor(self) -> None:
        """Reset actor parameters to a fresh initialization."""
        fresh_actor = GaussianActor(
            observation_dim=self.observation_dim,
            action_low=self.actor.action_bias.detach().cpu().numpy()
            - self.actor.action_scale.detach().cpu().numpy(),
            action_high=self.actor.action_bias.detach().cpu().numpy()
            + self.actor.action_scale.detach().cpu().numpy(),
            task_id_dim=self.task_id_dim,
            num_heads=self.num_tasks,
            hide_task_id=self.hide_task_id,
        ).to(self.device)
        self.actor.load_state_dict(
            fresh_actor.state_dict(),
            strict=True,
        )

    def reset_critics(self) -> None:
        """Reset online and target critics to fresh initializations."""
        fresh_critic1 = QCritic(
            observation_dim=self.observation_dim,
            action_dim=self.action_dim,
            task_id_dim=self.task_id_dim,
            num_heads=self.num_tasks,
            hide_task_id=self.hide_task_id,
        ).to(self.device)
        fresh_critic2 = QCritic(
            observation_dim=self.observation_dim,
            action_dim=self.action_dim,
            task_id_dim=self.task_id_dim,
            num_heads=self.num_tasks,
            hide_task_id=self.hide_task_id,
        ).to(self.device)
        self.critic1.load_state_dict(
            fresh_critic1.state_dict(),
            strict=True,
        )
        self.critic2.load_state_dict(
            fresh_critic2.state_dict(),
            strict=True,
        )
        self.target_critic1.load_state_dict(
            self.critic1.state_dict(),
            strict=True,
        )
        self.target_critic2.load_state_dict(
            self.critic2.state_dict(),
            strict=True,
        )
        self._freeze_target_critics()

    @property
    def device(self) -> torch.device:
        """Return the device used by the SAC networks."""
        return next(self.actor.parameters()).device

    def _build_optimizer(self) -> Adam:
        """Create the single Adam optimizer used by this implementation."""
        return Adam(
            chain(
                self.actor.parameters(),
                self.critic1.parameters(),
                self.critic2.parameters(),
                [self.log_alpha],
            ),
            lr=self.learning_rate,
            eps=1e-7,
        )

    @property
    def alpha(self) -> torch.Tensor:
        """Return alpha for a one-task agent."""
        if self.num_tasks != 1:
            raise RuntimeError(
                "This agent has multiple alphas. "
                "Use alpha_for_task(task_index)."
            )
        return self.alpha_for_task(0)

    @property
    def alpha_value(self) -> float:
        """Return alpha as a Python float for a one-task agent."""
        if self.num_tasks != 1:
            raise RuntimeError(
                "This agent has multiple alphas. "
                "Use alpha_value_for_task(task_index)."
            )
        return self.alpha_value_for_task(0)

    def alpha_for_task(self, task_index: int) -> torch.Tensor:
        """Return differentiable alpha for one task."""
        self._validate_task_index(task_index)
        return self.log_alpha[task_index].exp()

    def alpha_value_for_task(self, task_index: int) -> float:
        """Return one task's alpha as a Python float."""
        return float(
            self.alpha_for_task(task_index)
            .detach()
            .cpu()
            .item()
        )

    def diagnostic_alpha_value(self, task_index: int) -> float:
        """Return the alpha value associated with one training task."""
        if self.num_tasks == 1:
            return self.alpha_value
        return self.alpha_value_for_task(task_index)

    def all_alpha_values(self) -> list[float]:
        """Return all task-specific alpha values for diagnostics."""
        return [
            float(value)
            for value in self.log_alpha.detach().exp().cpu().tolist()
        ]

    @torch.inference_mode()
    def select_action(
        self,
        observation: np.ndarray | Sequence[float],
        *,
        deterministic: bool = False,
    ) -> np.ndarray:
        """Select one environment action using its encoded task head."""
        return self.actor.act(
            np.asarray(observation, dtype=np.float32),
            deterministic=deterministic,
            device=self.device,
        )

    @torch.inference_mode()
    def select_action_with_head(
        self,
        observation: np.ndarray | Sequence[float],
        *,
        head_index: int,
        deterministic: bool = False,
    ) -> np.ndarray:
        """Select an action using one explicitly specified actor head.

        The physical part of the observation is kept unchanged. Only
        the appended task one-hot vector is replaced so that the actor
        uses ``head_index``.

        This method supports explicit evaluation or rollout with one
        selected actor head on the current task.
        """
        overridden_observation = self._observation_for_actor_head(
            observation=observation,
            head_index=head_index,
        )

        return self.actor.act(
            overridden_observation,
            deterministic=deterministic,
            device=self.device,
        )

    def select_guide_action(
        self,
        observation: np.ndarray | Sequence[float],
        *,
        guide_task_index: int,
        deterministic: bool = False,
    ) -> np.ndarray:
        """Select an action from a prior task policy used as a guide."""
        return self.select_action_with_head(
            observation,
            head_index=guide_task_index,
            deterministic=deterministic,
        )

    def _observation_for_actor_head(
        self,
        *,
        observation: np.ndarray | Sequence[float],
        head_index: int,
    ) -> np.ndarray:
        """Return an observation copy selecting one requested head."""
        self._validate_task_index(head_index)

        observation_array = np.asarray(
            observation,
            dtype=np.float32,
        )

        expected_shape = (self.observation_dim,)
        if observation_array.shape != expected_shape:
            raise ValueError(
                f"Expected observation shape {expected_shape}, "
                f"received {observation_array.shape}."
            )

        overridden_observation = observation_array.copy()

        if self.task_id_dim == 0:
            if head_index != 0:
                raise ValueError(
                    "A single-task agent only has actor head 0."
                )
            return overridden_observation

        if self.task_id_dim != self.num_tasks:
            raise RuntimeError(
                "task_id_dim must equal num_tasks when selecting "
                "an explicit actor head."
            )

        task_one_hot = overridden_observation[-self.task_id_dim:]
        task_one_hot.fill(0.0)
        task_one_hot[head_index] = 1.0

        return overridden_observation

    def update_batch(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
        *,
        collect_metrics: bool = True,
    ) -> dict[str, float] | None:
        """Perform one SAC update and optionally collect diagnostics."""
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

        task_index = self._task_index_from_observations(observations)
        next_task_index = self._task_index_from_observations(
            next_observations
        )
        if next_task_index != task_index:
            raise RuntimeError(
                "Current and next observations contain different task IDs."
            )

        active_log_alpha = self.log_alpha[task_index]
        active_alpha = active_log_alpha.exp()

        with torch.no_grad():
            next_actor_output = self.actor.sample(next_observations)

            next_q1 = self.target_critic1(
                next_observations,
                next_actor_output.sampled_action,
            )
            next_q2 = self.target_critic2(
                next_observations,
                next_actor_output.sampled_action,
            )

            q_targets = compute_q_target(
                rewards=rewards,
                dones=dones,
                next_q1=next_q1,
                next_q2=next_q2,
                next_log_prob=next_actor_output.log_probability,
                alpha=active_alpha,
                gamma=self.gamma,
            )

        q1_predictions = self.critic1(observations, actions)
        q2_predictions = self.critic2(observations, actions)

        q1_loss, q2_loss = compute_critic_losses(
            q1_predictions=q1_predictions,
            q2_predictions=q2_predictions,
            q_targets=q_targets,
        )
        critic_loss = q1_loss + q2_loss

        actor_output = self.actor.sample(observations)
        q1_policy = self.critic1(
            observations,
            actor_output.sampled_action,
        )
        q2_policy = self.critic2(
            observations,
            actor_output.sampled_action,
        )

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
        critic_parameters = tuple(
            chain(
                self.critic1.parameters(),
                self.critic2.parameters(),
            )
        )

        actor_gradients = torch.autograd.grad(
            actor_loss,
            actor_parameters,
        )
        critic_gradients = torch.autograd.grad(
            critic_loss,
            critic_parameters,
        )
        alpha_gradients = torch.autograd.grad(
            alpha_loss,
            (self.log_alpha,),
        )

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
        self._apply_gradients(
            parameters=critic_parameters,
            gradients=critic_gradients,
            group_name="critic",
        )
        self._apply_gradients(
            parameters=(self.log_alpha,),
            gradients=alpha_gradients,
            group_name="alpha",
        )

        self.soft_update_targets()

        if not collect_metrics:
            return None

        return {
            "actor_loss": float(actor_loss.detach().item()),
            "q1_loss": float(q1_loss.detach().item()),
            "q2_loss": float(q2_loss.detach().item()),
            "alpha_loss": float(alpha_loss.detach().item()),
            "alpha": float(active_alpha.detach().item()),
            "q1_mean": float(q1_predictions.detach().mean().item()),
            "q2_mean": float(q2_predictions.detach().mean().item()),
            "q_target_mean": float(q_targets.detach().mean().item()),
            "log_prob_mean": float(
                actor_output.log_probability.detach().mean().item()
            ),
        }

    def adjust_actor_gradients(
        self,
        *,
        gradients: tuple[torch.Tensor, ...],
        parameters: tuple[nn.Parameter, ...],
        task_index: int,
    ) -> tuple[torch.Tensor, ...]:
        del parameters, task_index
        return gradients

    def adjust_critic_gradients(
        self,
        *,
        gradients: tuple[torch.Tensor, ...],
        parameters: tuple[nn.Parameter, ...],
        task_index: int,
    ) -> tuple[torch.Tensor, ...]:
        del parameters, task_index
        return gradients

    def adjust_alpha_gradients(
        self,
        *,
        gradients: tuple[torch.Tensor, ...],
        parameters: tuple[nn.Parameter, ...],
        task_index: int,
    ) -> tuple[torch.Tensor, ...]:
        del parameters, task_index
        return gradients

    def on_task_end(
        self,
        *,
        task_index: int,
        replay_buffer: object,
        batch_size: int,
    ) -> None:
        del task_index, replay_buffer, batch_size

    def on_task_start(
        self,
        *,
        task_index: int,
        replay_buffer: object,
    ) -> None:
        del task_index, replay_buffer

    def on_evaluation_start(self, task_index: int) -> None:
        del task_index

    def on_evaluation_end(self, task_index: int) -> None:
        del task_index

    def requires_guide_selection(self, task_index: int) -> bool:
        del task_index
        return False

    def initialize_task_from_guide(
        self,
        *,
        task_index: int,
        guide_task_index: int,
    ) -> None:
        del task_index, guide_task_index

    @torch.no_grad()
    def hard_update_targets(self) -> None:
        """Copy online critic parameters exactly to target critics."""
        self.target_critic1.load_state_dict(
            self.critic1.state_dict()
        )
        self.target_critic2.load_state_dict(
            self.critic2.state_dict()
        )

    @torch.no_grad()
    def soft_update_targets(self) -> None:
        """Apply one Polyak update to both target critics."""
        self._soft_update(
            online=self.critic1,
            target=self.target_critic1,
        )
        self._soft_update(
            online=self.critic2,
            target=self.target_critic2,
        )

    def _task_index_from_observations(
        self,
        observations: torch.Tensor,
    ) -> int:
        """Read and validate one homogeneous task ID for a replay batch."""
        if self.num_tasks == 1 and self.task_id_dim == 0:
            return 0

        task_vectors = observations[:, -self.task_id_dim:]
        task_indices = torch.argmax(task_vectors, dim=1)
        expected = torch.zeros_like(task_vectors)
        expected.scatter_(1, task_indices.unsqueeze(1), 1.0)
        if not torch.allclose(task_vectors, expected, atol=1e-5, rtol=0.0):
            raise ValueError(
                "Observations do not contain valid one-hot task IDs."
            )
        # A task-conditioned single-head agent uses the one-hot vector as an
        # ordinary network input and shares one entropy coefficient. The task
        # label is therefore not a head/alpha index.
        if self.num_tasks == 1:
            return 0
        if not bool(torch.all(task_indices == task_indices[0])):
            raise ValueError(
                "One SAC update batch must contain exactly one active task."
            )
        task_index = int(task_indices[0].item())
        self._validate_task_index(task_index)
        return task_index

    def _apply_gradients(
        self,
        *,
        parameters: tuple[nn.Parameter, ...],
        gradients: tuple[torch.Tensor, ...],
        group_name: str,
    ) -> None:
        """Apply one parameter-group update using shared Adam."""
        if len(parameters) != len(gradients):
            raise ValueError(
                "parameters and gradients must have equal length."
            )

        for gradient in gradients:
            if gradient is None:
                raise RuntimeError(
                    f"A required {group_name} gradient is None."
                )
            if not bool(torch.isfinite(gradient).all()):
                raise FloatingPointError(
                    f"Non-finite {group_name} gradient detected before optimizer step."
                )

        if self.gradient_clip_norm is not None:
            total_norm = torch.linalg.vector_norm(
                torch.stack(
                    [torch.linalg.vector_norm(gradient) for gradient in gradients]
                )
            )
            scale = min(
                1.0,
                self.gradient_clip_norm / (float(total_norm.detach()) + 1e-6),
            )
            gradients = tuple(gradient * scale for gradient in gradients)

        self.optimizer.zero_grad(set_to_none=True)

        for parameter, gradient in zip(
            parameters,
            gradients,
            strict=True,
        ):
            parameter.grad = gradient.detach()

        self.optimizer.step()

    def _prepare_update_batch(
        self,
        *,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Move one replay minibatch to the configured device."""
        prepared_observations = observations.to(
            self.device,
            dtype=torch.float32,
        )
        prepared_actions = actions.to(
            self.device,
            dtype=torch.float32,
        )
        prepared_rewards = rewards.to(
            self.device,
            dtype=torch.float32,
        )
        prepared_next_observations = next_observations.to(
            self.device,
            dtype=torch.float32,
        )
        prepared_dones = dones.to(
            self.device,
            dtype=torch.float32,
        )

        for name, tensor in (
            ("observations", prepared_observations),
            ("actions", prepared_actions),
            ("rewards", prepared_rewards),
            ("next_observations", prepared_next_observations),
            ("dones", prepared_dones),
        ):
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"{name} must contain only finite values.")

        if torch.any(prepared_dones < 0.0) or torch.any(
            prepared_dones > 1.0
        ):
            raise ValueError(
                "dones must contain only values in [0, 1]."
            )

        return (
            prepared_observations,
            prepared_actions,
            prepared_rewards,
            prepared_next_observations,
            prepared_dones,
        )

    def _soft_update(
        self,
        *,
        online: nn.Module,
        target: nn.Module,
    ) -> None:
        for online_parameter, target_parameter in zip(
            online.parameters(),
            target.parameters(),
            strict=True,
        ):
            target_parameter.mul_(self.polyak)
            target_parameter.add_(
                online_parameter,
                alpha=1.0 - self.polyak,
            )

    def _freeze_target_critics(self) -> None:
        self.target_critic1.requires_grad_(False)
        self.target_critic2.requires_grad_(False)
        self.target_critic1.eval()
        self.target_critic2.eval()

    def _validate_task_index(self, task_index: int) -> None:
        if not 0 <= task_index < self.num_tasks:
            raise ValueError(
                f"task_index must be in [0, {self.num_tasks}), "
                f"got {task_index}."
            )

    @staticmethod
    def _validate_hyperparameters(
        *,
        observation_dim: int,
        action_dim: int,
        num_tasks: int,
        task_id_dim: int,
        learning_rate: float,
        gamma: float,
        polyak: float,
        initial_log_alpha: float,
        target_entropy: float | None,
    ) -> None:
        if observation_dim <= 0:
            raise ValueError("observation_dim must be positive.")
        if action_dim <= 0:
            raise ValueError("action_dim must be positive.")
        if num_tasks <= 0:
            raise ValueError("num_tasks must be positive.")
        if task_id_dim < 0:
            raise ValueError("task_id_dim must be non-negative.")
        if task_id_dim >= observation_dim:
            raise ValueError(
                "task_id_dim must be smaller than observation_dim."
            )
        if num_tasks > 1 and task_id_dim != num_tasks:
            raise ValueError(
                "For multi-task runs, task_id_dim must equal num_tasks."
            )
        if not math.isfinite(learning_rate) or learning_rate <= 0.0:
            raise ValueError(
                "learning_rate must be finite and positive."
            )
        if not math.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be finite and in [0, 1].")
        if not math.isfinite(polyak) or not 0.0 <= polyak <= 1.0:
            raise ValueError("polyak must be finite and in [0, 1].")
        if not math.isfinite(initial_log_alpha):
            raise ValueError("initial_log_alpha must be finite.")
        if target_entropy is not None and not math.isfinite(
            target_entropy
        ):
            raise ValueError(
                "target_entropy must be finite when provided."
            )

    def _validate_action_bounds(
        self,
        *,
        action_low: np.ndarray,
        action_high: np.ndarray,
    ) -> None:
        expected_shape = (self.action_dim,)

        if action_low.shape != expected_shape:
            raise ValueError(
                f"action_low must have shape {expected_shape}, "
                f"got {action_low.shape}."
            )
        if action_high.shape != expected_shape:
            raise ValueError(
                f"action_high must have shape {expected_shape}, "
                f"got {action_high.shape}."
            )
        if not np.all(np.isfinite(action_low)) or not np.all(
            np.isfinite(action_high)
        ):
            raise ValueError("Action bounds must be finite.")
        if not np.all(action_low < action_high):
            raise ValueError(
                "Every action_low element must be smaller "
                "than action_high."
            )
