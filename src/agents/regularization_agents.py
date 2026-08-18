from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch

from agents.replay_buffer import ReplayBuffer
from agents.sac_agent import SACAgent


@dataclass(frozen=True)
class BackboneParameterRef:
    name: str
    parameter: torch.nn.Parameter


class RegularizationSACAgent(SACAgent):
    """Actor-backbone regularization baselines such as L2 and EWC."""

    def __init__(
        self,
        *args,
        cl_reg_coef: float = 1_000.0,
        fisher_batches: int = 10,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if not math.isfinite(cl_reg_coef) or cl_reg_coef < 0.0:
            raise ValueError("cl_reg_coef must be finite and non-negative.")
        if fisher_batches <= 0:
            raise ValueError("fisher_batches must be positive.")

        self.cl_reg_coef = float(cl_reg_coef)
        self.fisher_batches = int(fisher_batches)
        self._has_reference = False
        self._reference_parameters: dict[str, torch.Tensor] = {}
        self._importance_weights: dict[str, torch.Tensor] = {}
        self._actor_backbone_parameters = tuple(
            BackboneParameterRef(name=name, parameter=parameter)
            for name, parameter in self.actor.backbone.named_parameters()
        )

    def get_extra_state(self) -> dict[str, Any]:
        return {
            "has_reference": self._has_reference,
            "reference_parameters": {
                name: value.detach().cpu()
                for name, value in self._reference_parameters.items()
            },
            "importance_weights": {
                name: value.detach().cpu()
                for name, value in self._importance_weights.items()
            },
        }

    def set_extra_state(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            raise TypeError("Regularization extra state must be a dictionary.")

        has_reference = bool(state.get("has_reference", False))
        references = state.get("reference_parameters", {})
        importance = state.get("importance_weights", {})
        if not isinstance(references, dict) or not isinstance(importance, dict):
            raise TypeError(
                "Regularization reference parameters and importance weights "
                "must be dictionaries."
            )

        expected = {
            reference.name: reference.parameter
            for reference in self._actor_backbone_parameters
        }
        if has_reference and (
            set(references) != set(expected) or set(importance) != set(expected)
        ):
            raise ValueError(
                "Regularization checkpoint does not contain every actor "
                "backbone parameter."
            )

        restored_references: dict[str, torch.Tensor] = {}
        restored_importance: dict[str, torch.Tensor] = {}
        for name, parameter in expected.items():
            if not has_reference:
                continue
            reference = references[name]
            weight = importance[name]
            if not isinstance(reference, torch.Tensor) or not isinstance(
                weight, torch.Tensor
            ):
                raise TypeError(
                    f"Regularization state for {name} must contain tensors."
                )
            if reference.shape != parameter.shape or weight.shape != parameter.shape:
                raise ValueError(
                    f"Regularization state shape mismatch for {name}."
                )
            restored_references[name] = reference.to(
                device=parameter.device,
                dtype=parameter.dtype,
            )
            restored_importance[name] = weight.to(
                device=parameter.device,
                dtype=parameter.dtype,
            )

        self._has_reference = has_reference
        self._reference_parameters = restored_references
        self._importance_weights = restored_importance

    def on_task_end(
        self,
        *,
        task_index: int,
        replay_buffer: ReplayBuffer,
        batch_size: int,
    ) -> None:
        if len(replay_buffer) == 0:
            return

        new_weights = self._compute_importance_weights(
            replay_buffer=replay_buffer,
            task_index=task_index,
            batch_size=batch_size,
        )
        if not self._importance_weights:
            self._importance_weights = {
                name: value.detach().clone()
                for name, value in new_weights.items()
            }
        else:
            for name, value in new_weights.items():
                self._importance_weights[name] = (
                    self._importance_weights[name] + value.detach()
                )

        self._reference_parameters = {
            reference.name: reference.parameter.detach().clone()
            for reference in self._actor_backbone_parameters
        }
        self._has_reference = True

    def adjust_actor_gradients(
        self,
        *,
        gradients: tuple[torch.Tensor, ...],
        parameters: tuple[torch.nn.Parameter, ...],
        task_index: int,
    ) -> tuple[torch.Tensor, ...]:
        del task_index
        if not self._has_reference or self.cl_reg_coef == 0.0:
            return gradients

        adjusted: list[torch.Tensor] = []
        for gradient, parameter in zip(gradients, parameters, strict=True):
            parameter_name = self._find_actor_backbone_parameter_name(parameter)
            if parameter_name is None:
                adjusted.append(gradient)
                continue
            reference = self._reference_parameters.get(parameter_name)
            importance = self._importance_weights.get(parameter_name)
            if reference is None or importance is None:
                adjusted.append(gradient)
                continue
            reg_gradient = (
                2.0
                * self.cl_reg_coef
                * importance.to(device=gradient.device, dtype=gradient.dtype)
                * (
                    parameter.detach()
                    - reference.to(device=gradient.device, dtype=gradient.dtype)
                )
            )
            adjusted.append(gradient + reg_gradient)
        return tuple(adjusted)

    def _find_actor_backbone_parameter_name(
        self,
        parameter: torch.nn.Parameter,
    ) -> str | None:
        for reference in self._actor_backbone_parameters:
            if reference.parameter is parameter:
                return reference.name
        return None

    def _compute_importance_weights(
        self,
        *,
        replay_buffer: ReplayBuffer,
        task_index: int,
        batch_size: int,
    ) -> dict[str, torch.Tensor]:
        raise NotImplementedError


class L2SACAgent(RegularizationSACAgent):
    """Uniform actor-backbone regularization toward the previous task."""

    def _compute_importance_weights(
        self,
        *,
        replay_buffer: ReplayBuffer,
        task_index: int,
        batch_size: int,
    ) -> dict[str, torch.Tensor]:
        del replay_buffer, task_index, batch_size
        return {
            reference.name: torch.ones_like(reference.parameter.detach())
            for reference in self._actor_backbone_parameters
        }


class EWCSACAgent(RegularizationSACAgent):
    """Diagonal-Fisher EWC on the actor backbone.

    The diagonal is estimated with a Hutchinson projection of the exact
    Gaussian-policy Fisher. Unlike squaring a batch-averaged Jacobian, this is
    an unbiased estimate of the mean per-state squared Jacobian and does not
    cancel gradients from different observations.
    """

    def _compute_importance_weights(
        self,
        *,
        replay_buffer: ReplayBuffer,
        task_index: int,
        batch_size: int,
    ) -> dict[str, torch.Tensor]:
        fisher = {
            reference.name: torch.zeros_like(reference.parameter.detach())
            for reference in self._actor_backbone_parameters
        }
        if len(replay_buffer) == 0:
            return fisher

        num_batches = min(
            self.fisher_batches,
            max(1, math.ceil(len(replay_buffer) / max(1, batch_size))),
        )

        # A local generator keeps Fisher estimation reproducible without
        # perturbing the RNG stream used by subsequent SAC training.
        sign_generator = torch.Generator(device="cpu")
        sign_generator.manual_seed(1729 + int(task_index))

        was_training = self.actor.training
        self.actor.eval()
        try:
            for _ in range(num_batches):
                sample_size = min(batch_size, len(replay_buffer))
                batch = replay_buffer.sample(sample_size, self.device)
                observations = batch.observations
                mean, log_std = self.actor.distribution_parameters(
                    observations
                )
                std = log_std.exp()
                signs = torch.randint(
                    low=0,
                    high=2,
                    size=mean.shape,
                    generator=sign_generator,
                    device="cpu",
                    dtype=torch.int64,
                ).to(device=self.device, dtype=mean.dtype)
                signs = signs.mul(2.0).sub(1.0)
                normalization = math.sqrt(float(mean.shape[0]))
                detached_std = std.detach().clamp_min(1e-6)

                mean_projection = (
                    signs * mean / detached_std
                ).sum() / normalization
                mean_gradients = torch.autograd.grad(
                    mean_projection,
                    tuple(
                        reference.parameter
                        for reference in self._actor_backbone_parameters
                    ),
                    retain_graph=True,
                )

                std_projection = (
                    math.sqrt(2.0) * signs * std / detached_std
                ).sum() / normalization
                std_gradients = torch.autograd.grad(
                    std_projection,
                    tuple(
                        reference.parameter
                        for reference in self._actor_backbone_parameters
                    ),
                )

                for reference, mean_gradient, std_gradient in zip(
                    self._actor_backbone_parameters,
                    mean_gradients,
                    std_gradients,
                    strict=True,
                ):
                    fisher[reference.name] = fisher[reference.name] + (
                        mean_gradient.detach().pow(2)
                        + std_gradient.detach().pow(2)
                    ) / float(num_batches)
        finally:
            self.actor.zero_grad(set_to_none=True)
            if was_training:
                self.actor.train()

        return {
            name: weight.clamp(min=1e-5)
            for name, weight in fisher.items()
        }
