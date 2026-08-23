from __future__ import annotations

from dataclasses import dataclass

import torch

from agents.full_bc_agent import FullBehaviorCloningSACAgent


@dataclass(frozen=True)
class _GlobalHeadScaling:
    """Baseline global scaling retained for task-specific actor heads."""

    sac_norm: torch.Tensor
    applied_bc_norm: torch.Tensor
    norm_scale: torch.Tensor
    conflict: bool


class LayerwiseAdaptivePCGradAgent(FullBehaviorCloningSACAgent):
    """Adaptive PCGrad with independent conflict handling per backbone layer.

    Only the shared actor backbone differs from the original adaptive-PCGrad
    agent. Task-specific head gradients retain the exact global scaling that
    the original method would apply to the same SAC and BC gradients.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.bc_gradient_strategy != "pcgrad_sac_priority":
            raise ValueError(
                "LayerwiseAdaptivePCGradAgent requires pcgrad_sac_priority."
            )
        if self.bc_combination_strategy != "adaptive_additive":
            raise ValueError(
                "LayerwiseAdaptivePCGradAgent requires adaptive_additive."
            )
        covered = {
            index
            for indices in self.gradient_diagnostics.layer_indices.values()
            for index in indices
        }
        expected = set(self.gradient_diagnostics.shared_indices)
        if covered != expected:
            raise RuntimeError(
                "Actor backbone layer groups do not cover every shared parameter."
            )
        self._layer_conflicts: dict[str, bool] | None = None
        self._global_head_scaling: _GlobalHeadScaling | None = None

    @staticmethod
    def _norm_sq(
        gradients: tuple[torch.Tensor, ...] | list[torch.Tensor],
        indices: tuple[int, ...],
    ) -> torch.Tensor:
        return sum(gradients[index].square().sum() for index in indices)

    @staticmethod
    def _dot(
        first: tuple[torch.Tensor, ...] | list[torch.Tensor],
        second: tuple[torch.Tensor, ...] | list[torch.Tensor],
        indices: tuple[int, ...],
    ) -> torch.Tensor:
        return sum((first[index] * second[index]).sum() for index in indices)

    def _baseline_global_head_scaling(
        self,
        *,
        sac_gradients: tuple[torch.Tensor, ...],
        cloning_gradients: tuple[torch.Tensor, ...],
    ) -> _GlobalHeadScaling:
        """Reproduce the original global-PCGrad backbone scale for old heads."""
        shared = self.gradient_diagnostics.shared_indices
        epsilon = 1e-12
        sac_norm_sq = self._norm_sq(sac_gradients, shared)
        dot_product = self._dot(sac_gradients, cloning_gradients, shared)
        conflict = bool(dot_product.detach().item() < 0.0)
        globally_adjusted = list(cloning_gradients)
        if conflict and float(sac_norm_sq.detach().item()) > epsilon:
            projection_scale = dot_product / (sac_norm_sq + epsilon)
            for index in shared:
                globally_adjusted[index] = (
                    globally_adjusted[index]
                    - projection_scale * sac_gradients[index]
                )
        sac_norm = sac_norm_sq.sqrt()
        adjusted_norm = self._norm_sq(globally_adjusted, shared).sqrt()
        norm_scale = torch.clamp(
            self.bc_max_norm_ratio * sac_norm / (adjusted_norm + epsilon),
            max=1.0,
        )
        return _GlobalHeadScaling(
            sac_norm=sac_norm,
            applied_bc_norm=adjusted_norm * norm_scale,
            norm_scale=norm_scale,
            conflict=conflict,
        )

    def _apply_bc_gradient_strategy(
        self,
        *,
        sac_gradients: tuple[torch.Tensor, ...],
        cloning_gradients: tuple[torch.Tensor, ...],
    ) -> tuple[tuple[torch.Tensor, ...], dict[str, float | bool]]:
        epsilon = 1e-12
        shared = self.gradient_diagnostics.shared_indices
        shared_set = set(shared)
        head_scaling = self._baseline_global_head_scaling(
            sac_gradients=sac_gradients,
            cloning_gradients=cloning_gradients,
        )
        adjusted = list(cloning_gradients)
        layer_conflicts: dict[str, bool] = {}
        projection_applied = False

        for layer, indices in self.gradient_diagnostics.layer_indices.items():
            sac_norm_sq = self._norm_sq(sac_gradients, indices)
            dot_product = self._dot(sac_gradients, cloning_gradients, indices)
            conflict = bool(dot_product.detach().item() < 0.0)
            layer_conflicts[layer] = conflict
            if conflict and float(sac_norm_sq.detach().item()) > epsilon:
                projection_scale = dot_product / (sac_norm_sq + epsilon)
                for index in indices:
                    adjusted[index] = (
                        adjusted[index]
                        - projection_scale * sac_gradients[index]
                    )
                projection_applied = True

            sac_norm = sac_norm_sq.sqrt()
            adjusted_norm = self._norm_sq(adjusted, indices).sqrt()
            layer_norm_scale = torch.clamp(
                self.bc_max_norm_ratio * sac_norm / (adjusted_norm + epsilon),
                max=1.0,
            )
            for index in indices:
                adjusted[index] = adjusted[index] * layer_norm_scale

        # Preserve the original method's scaling for old task-specific heads.
        for index in range(len(adjusted)):
            if index not in shared_set:
                adjusted[index] = adjusted[index] * head_scaling.norm_scale

        raw_sac_norm = self._norm_sq(sac_gradients, shared).sqrt()
        raw_bc_norm = self._norm_sq(cloning_gradients, shared).sqrt()
        applied_bc_norm = self._norm_sq(adjusted, shared).sqrt()
        raw_dot = self._dot(sac_gradients, cloning_gradients, shared)
        aggregate_norm_scale = applied_bc_norm / (raw_bc_norm + epsilon)

        self._layer_conflicts = layer_conflicts
        self._global_head_scaling = head_scaling
        self.last_bc_gradient_cosine = float(
            (raw_dot / (raw_sac_norm * raw_bc_norm + epsilon)).detach().item()
        )
        self.last_bc_gradient_norm_ratio = float(
            (raw_bc_norm / (raw_sac_norm + epsilon)).detach().item()
        )
        self.last_bc_applied_norm_ratio = float(
            (applied_bc_norm / (raw_sac_norm + epsilon)).detach().item()
        )
        self.last_bc_conflict = head_scaling.conflict
        return tuple(adjusted), {
            "projection_applied": projection_applied,
            "bc_norm_scale": float(aggregate_norm_scale.detach().item()),
            "conflict": head_scaling.conflict,
        }

    def _combine_actor_gradients(
        self,
        *,
        sac_gradients: tuple[torch.Tensor, ...],
        applied_cloning_gradients: tuple[torch.Tensor, ...],
        conflict: bool,
    ) -> tuple[tuple[torch.Tensor, ...], float]:
        del conflict
        if self._layer_conflicts is None or self._global_head_scaling is None:
            raise RuntimeError(
                "Layer-wise gradient combination was called without adjustment state."
            )

        epsilon = 1e-12
        shared_set = set(self.gradient_diagnostics.shared_indices)
        combined = list(sac_gradients)
        effective_bc = [torch.zeros_like(gradient) for gradient in sac_gradients]

        for layer, indices in self.gradient_diagnostics.layer_indices.items():
            sac_norm = self._norm_sq(sac_gradients, indices).sqrt()
            bc_norm = self._norm_sq(applied_cloning_gradients, indices).sqrt()
            target_ratio = (
                self.bc_adaptive_conflict_ratio
                if self._layer_conflicts[layer]
                else self.bc_adaptive_target_ratio
            ) * self.bc_progress_multiplier
            layer_scale = torch.clamp(
                target_ratio * sac_norm / (bc_norm + epsilon),
                min=0.0,
                max=1.0,
            )
            for index in indices:
                effective_bc[index] = layer_scale * applied_cloning_gradients[index]
                combined[index] = sac_gradients[index] + effective_bc[index]

        global_target_ratio = (
            self.bc_adaptive_conflict_ratio
            if self._global_head_scaling.conflict
            else self.bc_adaptive_target_ratio
        ) * self.bc_progress_multiplier
        head_combination_scale = torch.clamp(
            global_target_ratio
            * self._global_head_scaling.sac_norm
            / (self._global_head_scaling.applied_bc_norm + epsilon),
            min=0.0,
            max=1.0,
        )
        for index in range(len(combined)):
            if index not in shared_set:
                effective_bc[index] = (
                    head_combination_scale * applied_cloning_gradients[index]
                )
                combined[index] = sac_gradients[index] + effective_bc[index]

        applied_shared_norm = self._norm_sq(
            applied_cloning_gradients,
            self.gradient_diagnostics.shared_indices,
        ).sqrt()
        effective_shared_norm = self._norm_sq(
            effective_bc,
            self.gradient_diagnostics.shared_indices,
        ).sqrt()
        aggregate_scale = effective_shared_norm / (applied_shared_norm + epsilon)
        self._layer_conflicts = None
        self._global_head_scaling = None
        return tuple(combined), float(aggregate_scale.detach().item())
