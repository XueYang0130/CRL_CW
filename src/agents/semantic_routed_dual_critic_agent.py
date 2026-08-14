from __future__ import annotations

import copy
from itertools import chain

import torch
from torch import nn
from torch.optim import Adam

from agents.full_bc_agent import FullBehaviorCloningSACAgent
from agents.sac_losses import compute_critic_losses, compute_q_target


class SemanticRoutedDualCriticAgent(FullBehaviorCloningSACAgent):
    """Adaptive-PCGrad actor with a conditional temporary reset critic."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._configured_critic_route = "transfer"
        self._active_critic_route = "transfer"
        self._transfer_critic1: nn.Module | None = None
        self._transfer_critic2: nn.Module | None = None
        self._transfer_target_critic1: nn.Module | None = None
        self._transfer_target_critic2: nn.Module | None = None
        self._transfer_critic_optimizer: Adam | None = None
        self.background_critic_updates = 0
        self.last_background_q1_loss = float("nan")
        self.last_background_q2_loss = float("nan")

    @property
    def active_critic_route(self) -> str:
        return self._active_critic_route

    def configure_critic_route(self, route: str) -> None:
        if route not in {"transfer", "reset"}:
            raise ValueError(f"Unsupported critic route: {route}.")
        if self._transfer_critic1 is not None:
            raise RuntimeError("Cannot change critic route during an active task.")
        self._configured_critic_route = route

    def on_task_start(self, *, task_index: int, replay_buffer: object) -> None:
        del replay_buffer
        if task_index == 0 and self._configured_critic_route != "transfer":
            raise RuntimeError("The first task cannot use a reset critic.")
        self._active_critic_route = self._configured_critic_route
        self.background_critic_updates = 0
        self.last_background_q1_loss = float("nan")
        self.last_background_q2_loss = float("nan")
        if self._active_critic_route == "transfer":
            return

        # Preserve the lifelong value learner before the active SAC critics are reset.
        self._transfer_critic1 = copy.deepcopy(self.critic1).to(self.device)
        self._transfer_critic2 = copy.deepcopy(self.critic2).to(self.device)
        self._transfer_target_critic1 = copy.deepcopy(self.target_critic1).to(self.device)
        self._transfer_target_critic2 = copy.deepcopy(self.target_critic2).to(self.device)
        self._freeze_background_targets()
        self._transfer_critic_optimizer = Adam(
            chain(
                self._transfer_critic1.parameters(),
                self._transfer_critic2.parameters(),
            ),
            lr=self.learning_rate,
            eps=1e-7,
        )
        self.reset_critics()
        self.rebuild_optimizer()

    def update_batch(self, *args, **kwargs):
        metrics = super().update_batch(*args, **kwargs)
        if self._active_critic_route == "reset":
            self._update_background_transfer_critic(
                observations=args[0] if args else kwargs["observations"],
                actions=args[1] if len(args) > 1 else kwargs["actions"],
                rewards=args[2] if len(args) > 2 else kwargs["rewards"],
                next_observations=args[3] if len(args) > 3 else kwargs["next_observations"],
                dones=args[4] if len(args) > 4 else kwargs["dones"],
            )
            if metrics is not None:
                metrics["background_q1_loss"] = self.last_background_q1_loss
                metrics["background_q2_loss"] = self.last_background_q2_loss
        return metrics

    def on_task_end(
        self,
        *,
        task_index: int,
        replay_buffer: object,
        batch_size: int,
    ) -> None:
        del task_index, replay_buffer, batch_size
        if self._active_critic_route != "reset":
            return
        if any(
            module is None
            for module in (
                self._transfer_critic1,
                self._transfer_critic2,
                self._transfer_target_critic1,
                self._transfer_target_critic2,
            )
        ):
            raise RuntimeError("Reset route ended without a complete transfer critic.")

        # The temporary reset critic is discarded; only the continuously trained
        # transfer branch is carried into the next task.
        self.critic1.load_state_dict(self._transfer_critic1.state_dict(), strict=True)
        self.critic2.load_state_dict(self._transfer_critic2.state_dict(), strict=True)
        self.target_critic1.load_state_dict(
            self._transfer_target_critic1.state_dict(), strict=True
        )
        self.target_critic2.load_state_dict(
            self._transfer_target_critic2.state_dict(), strict=True
        )
        self._freeze_target_critics()
        self._transfer_critic1 = None
        self._transfer_critic2 = None
        self._transfer_target_critic1 = None
        self._transfer_target_critic2 = None
        self._transfer_critic_optimizer = None
        self._active_critic_route = "transfer"
        self.rebuild_optimizer()

    def _update_background_transfer_critic(
        self,
        *,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
    ) -> None:
        if (
            self._transfer_critic1 is None
            or self._transfer_critic2 is None
            or self._transfer_target_critic1 is None
            or self._transfer_target_critic2 is None
            or self._transfer_critic_optimizer is None
        ):
            raise RuntimeError("Background transfer critic is not initialized.")
        observations, actions, rewards, next_observations, dones = self._prepare_update_batch(
            observations=observations,
            actions=actions,
            rewards=rewards,
            next_observations=next_observations,
            dones=dones,
        )
        task_index = self._task_index_from_observations(observations)
        with torch.no_grad():
            next_actor_output = self.actor.sample(next_observations)
            next_q1 = self._transfer_target_critic1(
                next_observations, next_actor_output.sampled_action
            )
            next_q2 = self._transfer_target_critic2(
                next_observations, next_actor_output.sampled_action
            )
            targets = compute_q_target(
                rewards=rewards,
                dones=dones,
                next_q1=next_q1,
                next_q2=next_q2,
                next_log_prob=next_actor_output.log_probability,
                alpha=self.alpha_for_task(task_index).detach(),
                gamma=self.gamma,
            )
        q1 = self._transfer_critic1(observations, actions)
        q2 = self._transfer_critic2(observations, actions)
        q1_loss, q2_loss = compute_critic_losses(
            q1_predictions=q1,
            q2_predictions=q2,
            q_targets=targets,
        )
        self._transfer_critic_optimizer.zero_grad(set_to_none=True)
        (q1_loss + q2_loss).backward()
        parameters = tuple(
            chain(self._transfer_critic1.parameters(), self._transfer_critic2.parameters())
        )
        if self.gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(parameters, self.gradient_clip_norm)
        for parameter in parameters:
            if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
                raise FloatingPointError("Non-finite background critic gradient detected.")
        self._transfer_critic_optimizer.step()
        with torch.no_grad():
            self._soft_update(
                online=self._transfer_critic1,
                target=self._transfer_target_critic1,
            )
            self._soft_update(
                online=self._transfer_critic2,
                target=self._transfer_target_critic2,
            )
        self.background_critic_updates += 1
        self.last_background_q1_loss = float(q1_loss.detach().item())
        self.last_background_q2_loss = float(q2_loss.detach().item())

    def _freeze_background_targets(self) -> None:
        assert self._transfer_target_critic1 is not None
        assert self._transfer_target_critic2 is not None
        self._transfer_target_critic1.requires_grad_(False)
        self._transfer_target_critic2.requires_grad_(False)
        self._transfer_target_critic1.eval()
        self._transfer_target_critic2.eval()


class SemanticRoutedFrozenTransferCriticAgent(SemanticRoutedDualCriticAgent):
    """Use a temporary reset critic while preserving the transfer critic unchanged."""

    def _update_background_transfer_critic(
        self,
        *,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
    ) -> None:
        # The reset critic handles this task. Keeping this snapshot untouched
        # prevents a routed task from changing the critic carried downstream.
        del observations, actions, rewards, next_observations, dones
