from __future__ import annotations

import math

import numpy as np
import torch

from agents.gradient_diagnostics import GradientConflictDiagnostics
from agents.sac_agent import SACAgent


class FullBehaviorCloningSACAgent(SACAgent):
    """CW10 full-trajectory behavioral cloning with ClonEx-style policy distillation."""

    def __init__(
        self,
        *args,
        episodic_batch_size: int = 128,
        actor_cloning_coefficient: float = 100.0,
        bc_gradient_strategy: str = "standard",
        bc_max_norm_ratio: float = 1.0,
        bc_combination_strategy: str = "average",
        bc_adaptive_target_ratio: float = 0.2,
        bc_adaptive_conflict_ratio: float = 0.05,
        bc_cagrad_alpha: float = 0.5,
        gradient_diagnostics: bool = False,
        gradient_diagnostics_interval: int = 500,
        gradient_diagnostics_source_batch_size: int = 128,
        gradient_diagnostics_seed: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if self.num_tasks <= 1 or self.task_id_dim != self.num_tasks:
            raise ValueError("full_bc requires task-specific actor and critic heads.")
        if episodic_batch_size <= 0:
            raise ValueError("episodic_batch_size must be positive.")
        if (
            not math.isfinite(actor_cloning_coefficient)
            or actor_cloning_coefficient < 0.0
        ):
            raise ValueError(
                "actor_cloning_coefficient must be finite and non-negative."
            )
        valid_gradient_strategies = {
            "standard",
            "norm_balanced",
            "pcgrad_sac_priority",
        }
        if bc_gradient_strategy not in valid_gradient_strategies:
            raise ValueError(
                "Unsupported BC gradient strategy: "
                f"{bc_gradient_strategy}."
            )
        if not math.isfinite(bc_max_norm_ratio) or bc_max_norm_ratio <= 0.0:
            raise ValueError("bc_max_norm_ratio must be finite and positive.")
        valid_combination_strategies = {
            "average",
            "additive",
            "adaptive_additive",
            "cagrad",
        }
        if bc_combination_strategy not in valid_combination_strategies:
            raise ValueError(
                "Unsupported BC combination strategy: "
                f"{bc_combination_strategy}."
            )
        if (
            not math.isfinite(bc_adaptive_target_ratio)
            or bc_adaptive_target_ratio < 0.0
        ):
            raise ValueError("bc_adaptive_target_ratio must be finite and non-negative.")
        if (
            not math.isfinite(bc_adaptive_conflict_ratio)
            or bc_adaptive_conflict_ratio < 0.0
        ):
            raise ValueError("bc_adaptive_conflict_ratio must be finite and non-negative.")
        if bc_adaptive_conflict_ratio > bc_adaptive_target_ratio:
            raise ValueError(
                "bc_adaptive_conflict_ratio cannot exceed "
                "bc_adaptive_target_ratio."
            )
        if not math.isfinite(bc_cagrad_alpha) or not 0.0 <= bc_cagrad_alpha < 1.0:
            raise ValueError("bc_cagrad_alpha must be finite and in [0, 1).")

        self.episodic_batch_size = int(episodic_batch_size)
        self.actor_cloning_coefficient = float(actor_cloning_coefficient)
        self.bc_gradient_strategy = bc_gradient_strategy
        self.bc_max_norm_ratio = float(bc_max_norm_ratio)
        self.bc_combination_strategy = bc_combination_strategy
        self.bc_adaptive_target_ratio = float(bc_adaptive_target_ratio)
        self.bc_adaptive_conflict_ratio = float(bc_adaptive_conflict_ratio)
        self.bc_cagrad_alpha = float(bc_cagrad_alpha)
        self.bc_progress_multiplier = 1.0
        self.gradient_aware_bc = False
        self.gradient_aware_max_norm_ratio = 1.0
        self.last_bc_gradient_cosine = float("nan")
        self.last_bc_gradient_norm_ratio = float("nan")
        self.last_bc_applied_norm_ratio = float("nan")
        self.last_bc_conflict = False
        self._episodic_observations = torch.empty((0, self.observation_dim))
        self._episodic_target_means = torch.empty((0, self.action_dim))
        self._episodic_target_log_stds = torch.empty((0, self.action_dim))
        self._episodic_source_task_indices = torch.empty((0,), dtype=torch.long)
        self._max_reference_source_task_index = -1
        self.gradient_diagnostics = GradientConflictDiagnostics(
            actor=self.actor,
            enabled=gradient_diagnostics,
            interval=gradient_diagnostics_interval,
            source_batch_size=gradient_diagnostics_source_batch_size,
            gradient_clip_norm=self.gradient_clip_norm,
            seed=gradient_diagnostics_seed,
        )

    def set_bc_progress_multiplier(self, multiplier: float) -> None:
        if not math.isfinite(multiplier) or not 0.0 <= multiplier <= 1.0:
            raise ValueError("BC progress multiplier must be finite and in [0, 1].")
        self.bc_progress_multiplier = float(multiplier)

    def configure_gradient_aware_bc(
        self,
        *,
        enabled: bool,
        max_norm_ratio: float = 1.0,
    ) -> None:
        if not math.isfinite(max_norm_ratio) or max_norm_ratio <= 0.0:
            raise ValueError("max_norm_ratio must be finite and positive.")
        self.gradient_aware_bc = bool(enabled)
        self.gradient_aware_max_norm_ratio = float(max_norm_ratio)
        self.bc_gradient_strategy = (
            "pcgrad_sac_priority" if enabled else "standard"
        )
        self.bc_max_norm_ratio = float(max_norm_ratio)

    @property
    def reference_state_count(self) -> int:
        return int(self._episodic_observations.shape[0])

    def clear_reference_memory(self) -> None:
        self._episodic_observations = torch.empty((0, self.observation_dim))
        self._episodic_target_means = torch.empty((0, self.action_dim))
        self._episodic_target_log_stds = torch.empty((0, self.action_dim))
        self._episodic_source_task_indices = torch.empty((0,), dtype=torch.long)
        self._max_reference_source_task_index = -1

    @torch.no_grad()
    def add_reference_memory(
        self,
        *,
        observations: np.ndarray | torch.Tensor,
    ) -> None:
        observations_tensor = torch.as_tensor(observations, dtype=torch.float32)
        if observations_tensor.ndim != 2 or observations_tensor.shape[1] != self.observation_dim:
            raise ValueError("reference observations have the wrong shape.")
        if observations_tensor.shape[0] == 0:
            return
        if not bool(torch.isfinite(observations_tensor).all()):
            raise ValueError("reference observations must be finite.")
        source_task_indices = self._infer_source_task_indices(observations_tensor)
        means, log_stds = self.compute_reference_targets(observations_tensor)
        if not bool(torch.isfinite(means).all()) or not bool(torch.isfinite(log_stds).all()):
            raise RuntimeError("reference policy targets must be finite.")
        self._episodic_observations = torch.cat(
            (self._episodic_observations, observations_tensor.detach().cpu())
        )
        self._episodic_target_means = torch.cat(
            (self._episodic_target_means, means.detach().cpu())
        )
        self._episodic_target_log_stds = torch.cat(
            (self._episodic_target_log_stds, log_stds.detach().cpu())
        )
        self._episodic_source_task_indices = torch.cat(
            (
                self._episodic_source_task_indices,
                source_task_indices,
            )
        )
        self._max_reference_source_task_index = max(
            self._max_reference_source_task_index,
            int(source_task_indices.max().item()),
        )

    def add_reference_memory_with_targets(
        self,
        *,
        observations: np.ndarray | torch.Tensor,
        target_means: np.ndarray | torch.Tensor,
        target_log_stds: np.ndarray | torch.Tensor,
    ) -> None:
        observations_tensor = torch.as_tensor(observations, dtype=torch.float32)
        target_means_tensor = torch.as_tensor(target_means, dtype=torch.float32)
        target_log_stds_tensor = torch.as_tensor(target_log_stds, dtype=torch.float32)
        if observations_tensor.ndim != 2 or observations_tensor.shape[1] != self.observation_dim:
            raise ValueError("reference observations have the wrong shape.")
        if target_means_tensor.ndim != 2 or target_means_tensor.shape[1] != self.action_dim:
            raise ValueError("reference target means have the wrong shape.")
        if target_log_stds_tensor.ndim != 2 or target_log_stds_tensor.shape[1] != self.action_dim:
            raise ValueError("reference target log stds have the wrong shape.")
        if observations_tensor.shape[0] != target_means_tensor.shape[0]:
            raise ValueError("reference observations and target means must have the same length.")
        if observations_tensor.shape[0] != target_log_stds_tensor.shape[0]:
            raise ValueError("reference observations and target log stds must have the same length.")
        if observations_tensor.shape[0] == 0:
            return
        if not bool(torch.isfinite(observations_tensor).all()):
            raise ValueError("reference observations must be finite.")
        if not bool(torch.isfinite(target_means_tensor).all()):
            raise ValueError("reference target means must be finite.")
        if not bool(torch.isfinite(target_log_stds_tensor).all()):
            raise ValueError("reference target log stds must be finite.")
        source_task_indices = self._infer_source_task_indices(observations_tensor)
        self._episodic_observations = torch.cat(
            (self._episodic_observations, observations_tensor.detach().cpu())
        )
        self._episodic_target_means = torch.cat(
            (self._episodic_target_means, target_means_tensor.detach().cpu())
        )
        self._episodic_target_log_stds = torch.cat(
            (self._episodic_target_log_stds, target_log_stds_tensor.detach().cpu())
        )
        self._episodic_source_task_indices = torch.cat(
            (
                self._episodic_source_task_indices,
                source_task_indices,
            )
        )
        self._max_reference_source_task_index = max(
            self._max_reference_source_task_index,
            int(source_task_indices.max().item()),
        )

    def set_reference_memory_with_targets(
        self,
        *,
        observations: np.ndarray | torch.Tensor,
        target_means: np.ndarray | torch.Tensor,
        target_log_stds: np.ndarray | torch.Tensor,
    ) -> None:
        self.clear_reference_memory()
        self.add_reference_memory_with_targets(
            observations=observations,
            target_means=target_means,
            target_log_stds=target_log_stds,
        )

    @torch.no_grad()
    def compute_reference_targets(
        self,
        observations: np.ndarray | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        observations_tensor = torch.as_tensor(observations, dtype=torch.float32)
        if observations_tensor.ndim != 2 or observations_tensor.shape[1] != self.observation_dim:
            raise ValueError("reference observations have the wrong shape.")
        return self._cloning_distribution_parameters(
            observations_tensor.to(self.device)
        )

    def adjust_actor_gradients(
        self,
        *,
        gradients: tuple[torch.Tensor, ...],
        parameters: tuple[torch.nn.Parameter, ...],
        task_index: int,
    ) -> tuple[torch.Tensor, ...]:
        if task_index == 0 or self.reference_state_count == 0:
            return gradients
        if self._max_reference_source_task_index >= task_index:
            invalid_sources = torch.unique(
                self._episodic_source_task_indices[
                    self._episodic_source_task_indices >= task_index
                ]
            ).tolist()
            raise RuntimeError(
                "Behavior-cloning memory contains current or future task data: "
                f"current_task_index={task_index}, invalid_sources={invalid_sources}."
            )
        observations, target_means, target_log_stds = self._sample_episodic_batch()
        current_means, current_log_stds = self._cloning_distribution_parameters(observations)
        raw_cloning_loss = self._gaussian_kl(
            target_means,
            target_log_stds,
            current_means,
            current_log_stds,
        ).mean()
        weighted_cloning_loss = (
            self.actor_cloning_coefficient * raw_cloning_loss
        )
        cloning_gradients = torch.autograd.grad(
            weighted_cloning_loss,
            parameters,
        )
        applied_cloning_gradients, gradient_adjustment = (
            self._apply_bc_gradient_strategy(
                sac_gradients=gradients,
                cloning_gradients=cloning_gradients,
            )
        )
        final_actor_gradients, combination_scale = self._combine_actor_gradients(
            sac_gradients=gradients,
            applied_cloning_gradients=applied_cloning_gradients,
            conflict=bool(gradient_adjustment["conflict"]),
        )
        collect_global = self.gradient_diagnostics.begin_bc_update()
        if collect_global:
            self.gradient_diagnostics.record(
                sac_gradients=gradients,
                bc_gradients=cloning_gradients,
                current_task_index=task_index,
                raw_bc_loss=float(raw_cloning_loss.detach().item()),
                bc_coefficient=self.actor_cloning_coefficient,
                sac_actor_loss=float(self._current_actor_loss.item()),
                reference_memory_states=self.reference_state_count,
                applied_bc_gradients=applied_cloning_gradients,
                final_actor_gradients=final_actor_gradients,
                gradient_strategy=self.bc_gradient_strategy,
                projection_applied=gradient_adjustment["projection_applied"],
                bc_norm_scale=gradient_adjustment["bc_norm_scale"],
                bc_combination_strategy=self.bc_combination_strategy,
                bc_combination_scale=combination_scale,
            )
        if collect_global:
            self._record_source_task_diagnostics(
                sac_gradients=gradients,
                parameters=parameters,
                current_task_index=task_index,
            )
        return final_actor_gradients

    def _combine_actor_gradients(
        self,
        *,
        sac_gradients: tuple[torch.Tensor, ...],
        applied_cloning_gradients: tuple[torch.Tensor, ...],
        conflict: bool,
    ) -> tuple[tuple[torch.Tensor, ...], float]:
        if self.bc_combination_strategy == "average":
            multiplier = self.bc_progress_multiplier
            return tuple(
                (sac_gradient + multiplier * cloning_gradient) / (1.0 + multiplier)
                for sac_gradient, cloning_gradient in zip(
                    sac_gradients,
                    applied_cloning_gradients,
                    strict=True,
                )
            ), multiplier
        if self.bc_combination_strategy == "additive":
            multiplier = self.bc_progress_multiplier
            return tuple(
                sac_gradient + multiplier * cloning_gradient
                for sac_gradient, cloning_gradient in zip(
                    sac_gradients,
                    applied_cloning_gradients,
                    strict=True,
                )
            ), multiplier
        if self.bc_combination_strategy == "cagrad":
            return self._combine_actor_gradients_cagrad(
                sac_gradients=sac_gradients,
                bc_gradients=applied_cloning_gradients,
            )
        if self.bc_combination_strategy != "adaptive_additive":
            raise RuntimeError(
                "Unsupported BC combination strategy reached runtime: "
                f"{self.bc_combination_strategy}."
            )

        shared_indices = self.gradient_diagnostics.shared_indices
        epsilon = 1e-12
        sac_norm = sum(
            sac_gradients[index].square().sum() for index in shared_indices
        ).sqrt()
        bc_norm = sum(
            applied_cloning_gradients[index].square().sum()
            for index in shared_indices
        ).sqrt()
        target_ratio = (
            self.bc_adaptive_conflict_ratio
            if conflict
            else self.bc_adaptive_target_ratio
        ) * self.bc_progress_multiplier
        scale = torch.clamp(
            target_ratio * sac_norm / (bc_norm + epsilon),
            min=0.0,
            max=1.0,
        )
        scale_value = float(scale.detach().item())
        return tuple(
            sac_gradient + scale * cloning_gradient
            for sac_gradient, cloning_gradient in zip(
                sac_gradients,
                applied_cloning_gradients,
                strict=True,
            )
        ), scale_value

    def _combine_actor_gradients_cagrad(
        self,
        *,
        sac_gradients: tuple[torch.Tensor, ...],
        bc_gradients: tuple[torch.Tensor, ...],
    ) -> tuple[tuple[torch.Tensor, ...], float]:
        """Combine SAC and BC with the two-objective CAGrad direction."""
        progress = self.bc_progress_multiplier
        if progress == 0.0:
            return sac_gradients, 0.0

        shared_indices = self.gradient_diagnostics.shared_indices
        scaled_bc = tuple(progress * gradient for gradient in bc_gradients)
        mean_gradient = tuple(
            0.5 * (sac + bc)
            for sac, bc in zip(sac_gradients, scaled_bc, strict=True)
        )
        mean_norm = sum(
            mean_gradient[index].square().sum() for index in shared_indices
        ).sqrt()
        epsilon = 1e-12
        if float(mean_norm.detach().item()) <= epsilon or self.bc_cagrad_alpha == 0.0:
            return mean_gradient, 0.5

        sac_norm_sq = sum(
            sac_gradients[index].square().sum() for index in shared_indices
        )
        bc_norm_sq = sum(
            scaled_bc[index].square().sum() for index in shared_indices
        )
        sac_bc_dot = sum(
            (sac_gradients[index] * scaled_bc[index]).sum()
            for index in shared_indices
        )
        sac_norm_sq_value = float(sac_norm_sq.detach().item())
        bc_norm_sq_value = float(bc_norm_sq.detach().item())
        sac_bc_dot_value = float(sac_bc_dot.detach().item())

        # With two objectives, the simplex optimization is one-dimensional.
        # Evaluate it from the 2x2 Gram matrix rather than revisiting the model.
        def objective(weight: float) -> float:
            bc_weight = 1.0 - weight
            candidate_norm_sq = max(
                0.0,
                weight * weight * sac_norm_sq_value
                + bc_weight * bc_weight * bc_norm_sq_value
                + 2.0 * weight * bc_weight * sac_bc_dot_value,
            )
            alignment = 0.5 * (
                weight * (sac_norm_sq_value + sac_bc_dot_value)
                + bc_weight * (sac_bc_dot_value + bc_norm_sq_value)
            )
            return (
                alignment
                + self.bc_cagrad_alpha
                * float(mean_norm.detach().item())
                * math.sqrt(candidate_norm_sq)
            )

        left, right = 0.0, 1.0
        inverse_phi = (math.sqrt(5.0) - 1.0) / 2.0
        x1 = right - inverse_phi * (right - left)
        x2 = left + inverse_phi * (right - left)
        f1, f2 = objective(x1), objective(x2)
        for _ in range(48):
            if f1 <= f2:
                right, x2, f2 = x2, x1, f1
                x1 = right - inverse_phi * (right - left)
                f1 = objective(x1)
            else:
                left, x1, f1 = x1, x2, f2
                x2 = left + inverse_phi * (right - left)
                f2 = objective(x2)
        sac_weight = 0.5 * (left + right)
        weighted_gradient = tuple(
            sac_weight * sac + (1.0 - sac_weight) * bc
            for sac, bc in zip(sac_gradients, scaled_bc, strict=True)
        )
        weighted_shared_norm = sum(
            weighted_gradient[index].square().sum() for index in shared_indices
        ).sqrt()
        cagrad_scale = (
            self.bc_cagrad_alpha * mean_norm / (weighted_shared_norm + epsilon)
        )
        normalizer = 1.0 + self.bc_cagrad_alpha**2
        combined = tuple(
            (mean + cagrad_scale * weighted) / normalizer
            for mean, weighted in zip(
                mean_gradient,
                weighted_gradient,
                strict=True,
            )
        )
        return combined, float(sac_weight)

    def _apply_bc_gradient_strategy(
        self,
        *,
        sac_gradients: tuple[torch.Tensor, ...],
        cloning_gradients: tuple[torch.Tensor, ...],
    ) -> tuple[tuple[torch.Tensor, ...], dict[str, float | bool]]:
        shared_indices = self.gradient_diagnostics.shared_indices
        epsilon = 1e-12
        sac_norm_sq = sum(
            sac_gradients[index].square().sum() for index in shared_indices
        )
        bc_norm_sq = sum(
            cloning_gradients[index].square().sum() for index in shared_indices
        )
        dot_product = sum(
            (sac_gradients[index] * cloning_gradients[index]).sum()
            for index in shared_indices
        )
        conflict = bool(dot_product.detach().item() < 0.0)
        if self.bc_gradient_strategy == "standard":
            return cloning_gradients, {
                "projection_applied": False,
                "bc_norm_scale": 1.0,
                "conflict": conflict,
            }

        adjusted = list(cloning_gradients)
        projection_applied = bool(
            self.bc_gradient_strategy == "pcgrad_sac_priority"
            and sac_norm_sq.detach().item() > epsilon
            and conflict
        )
        if projection_applied:
            projection_scale = dot_product / (sac_norm_sq + epsilon)
            for index in shared_indices:
                adjusted[index] = (
                    adjusted[index]
                    - projection_scale * sac_gradients[index]
                )

        sac_norm = sac_norm_sq.sqrt()
        bc_norm = bc_norm_sq.sqrt()
        adjusted_shared_norm = sum(
            adjusted[index].square().sum() for index in shared_indices
        ).sqrt()
        norm_scale = torch.clamp(
            self.bc_max_norm_ratio
            * sac_norm
            / (adjusted_shared_norm + epsilon),
            max=1.0,
        )
        adjusted = [gradient * norm_scale for gradient in adjusted]
        scale_value = float(norm_scale.detach().item())

        self.last_bc_gradient_cosine = float(
            (
                dot_product
                / (
                    sac_norm
                    * bc_norm
                    + epsilon
                )
            ).detach().item()
        )
        self.last_bc_gradient_norm_ratio = float(
            (
                bc_norm / (sac_norm + epsilon)
            ).detach().item()
        )
        self.last_bc_applied_norm_ratio = float(
            (adjusted_shared_norm * norm_scale / (sac_norm + epsilon))
            .detach()
            .item()
        )
        self.last_bc_conflict = conflict
        return tuple(adjusted), {
            "projection_applied": projection_applied,
            "bc_norm_scale": scale_value,
            "conflict": self.last_bc_conflict,
        }

    def _record_source_task_diagnostics(
        self,
        *,
        sac_gradients: tuple[torch.Tensor, ...],
        parameters: tuple[torch.nn.Parameter, ...],
        current_task_index: int,
    ) -> None:
        for source_task_index in sorted(
            int(value)
            for value in torch.unique(self._episodic_source_task_indices).tolist()
        ):
            if source_task_index >= current_task_index:
                raise RuntimeError("Gradient diagnostics detected future-task memory leakage.")
            candidates = torch.nonzero(
                self._episodic_source_task_indices == source_task_index,
                as_tuple=False,
            ).flatten()
            indices = self.gradient_diagnostics.sample_indices(candidates)
            observations = self._episodic_observations[indices].to(self.device)
            target_means = self._episodic_target_means[indices].to(self.device)
            target_log_stds = self._episodic_target_log_stds[indices].to(self.device)
            current_means, current_log_stds = self._cloning_distribution_parameters(
                observations
            )
            raw_loss = self._gaussian_kl(
                target_means,
                target_log_stds,
                current_means,
                current_log_stds,
            ).mean()
            source_gradients = torch.autograd.grad(raw_loss, parameters)
            scaled_gradients = tuple(
                self.actor_cloning_coefficient * gradient
                for gradient in source_gradients
            )
            self.gradient_diagnostics.record_task_pair(
                sac_gradients=sac_gradients,
                bc_gradients=scaled_gradients,
                current_task_index=current_task_index,
                source_task_index=source_task_index,
                raw_bc_loss=float(raw_loss.detach().item()),
                bc_coefficient=self.actor_cloning_coefficient,
                sac_actor_loss=float(self._current_actor_loss.item()),
                reference_memory_states=self.reference_state_count,
                source_memory_states=int(candidates.numel()),
            )

    def drain_gradient_diagnostics(self) -> dict[str, list[dict[str, object]]]:
        return self.gradient_diagnostics.drain()

    def gradient_diagnostics_summary(self) -> dict[str, float | int | bool]:
        return self.gradient_diagnostics.summary()

    def _infer_source_task_indices(self, observations: torch.Tensor) -> torch.Tensor:
        task_vectors = observations[:, -self.task_id_dim :].detach().cpu()
        indices = torch.argmax(task_vectors, dim=1)
        expected = torch.zeros_like(task_vectors)
        expected.scatter_(1, indices.unsqueeze(1), 1.0)
        if not torch.allclose(task_vectors, expected, atol=1e-5, rtol=0.0):
            raise ValueError("Reference observations do not contain valid one-hot task IDs.")
        return indices.to(dtype=torch.long)

    def _sample_episodic_batch(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        memory_size = self._episodic_observations.shape[0]
        sample_size = min(self.episodic_batch_size, memory_size)
        indices = np.random.choice(
            memory_size,
            size=sample_size,
            replace=False,
        )
        return (
            self._episodic_observations[indices].to(self.device),
            self._episodic_target_means[indices].to(self.device),
            self._episodic_target_log_stds[indices].to(self.device),
        )

    @staticmethod
    def _gaussian_kl(
        first_mean: torch.Tensor,
        first_log_std: torch.Tensor,
        second_mean: torch.Tensor,
        second_log_std: torch.Tensor,
    ) -> torch.Tensor:
        epsilon = 1e-6
        first_variance = first_log_std.exp().square() + epsilon
        second_variance = second_log_std.exp().square() + epsilon
        return (
            second_log_std
            - first_log_std
            + (first_variance + (first_mean - second_mean).square())
            / (2.0 * second_variance)
            - 0.5
        ).sum(dim=-1)

    def _cloning_distribution_parameters(
        self,
        observations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Return pre-tanh raw means so KL is computed in the same space as log_stds.
        return self.actor.distribution_parameters(observations)
