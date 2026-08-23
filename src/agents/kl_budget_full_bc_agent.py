from __future__ import annotations

import math

import torch

from agents.full_bc_agent import FullBehaviorCloningSACAgent


class KLBudgetFullBehaviorCloningSACAgent(FullBehaviorCloningSACAgent):
    """Adaptive PCGrad with BC strength controlled by measured policy drift."""

    def __init__(
        self,
        *args,
        kl_budget_low: float = 0.05,
        kl_budget_high: float = 0.5,
        kl_budget_ema_beta: float = 0.99,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if not math.isfinite(kl_budget_low) or kl_budget_low < 0.0:
            raise ValueError("kl_budget_low must be finite and non-negative.")
        if not math.isfinite(kl_budget_high) or kl_budget_high <= kl_budget_low:
            raise ValueError("kl_budget_high must be finite and exceed kl_budget_low.")
        if (
            not math.isfinite(kl_budget_ema_beta)
            or not 0.0 <= kl_budget_ema_beta < 1.0
        ):
            raise ValueError("kl_budget_ema_beta must be finite and in [0, 1).")

        self.kl_budget_low = float(kl_budget_low)
        self.kl_budget_high = float(kl_budget_high)
        self.kl_budget_ema_beta = float(kl_budget_ema_beta)
        self.kl_budget_ema: float | None = None
        self.kl_budget_baseline: float | None = None
        self.kl_budget_excess = 0.0
        self.kl_budget_multiplier = 0.0
        self._kl_budget_task_index: int | None = None
        self._kl_budget_updates = 0
        self._kl_budget_multiplier_sum = 0.0
        self._kl_budget_active_updates = 0
        self._kl_budget_full_updates = 0

    @torch.no_grad()
    def _measure_reference_kl(self, *, batch_size: int = 4096) -> float:
        if self.reference_state_count == 0:
            return 0.0
        total = 0.0
        for start in range(0, self.reference_state_count, batch_size):
            stop = min(start + batch_size, self.reference_state_count)
            observations = self._episodic_observations[start:stop].to(self.device)
            target_means = self._episodic_target_means[start:stop].to(self.device)
            target_log_stds = self._episodic_target_log_stds[start:stop].to(self.device)
            current_means, current_log_stds = self._cloning_distribution_parameters(
                observations
            )
            values = self._gaussian_kl(
                target_means,
                target_log_stds,
                current_means,
                current_log_stds,
            )
            total += float(values.sum().item())
        return total / self.reference_state_count

    def _start_kl_budget_task(self, *, task_index: int, baseline_kl: float) -> None:
        if not math.isfinite(baseline_kl) or baseline_kl < 0.0:
            raise RuntimeError("Initial reference KL must be finite and non-negative.")
        self._kl_budget_task_index = int(task_index)
        self.kl_budget_baseline = float(baseline_kl)
        self.kl_budget_ema = float(baseline_kl)
        self.kl_budget_excess = 0.0
        self.kl_budget_multiplier = 0.0
        self.bc_progress_multiplier = 0.0

    def _update_kl_budget(self, *, raw_kl: float, task_index: int) -> float:
        if not math.isfinite(raw_kl) or raw_kl < 0.0:
            raise RuntimeError("Measured behavior-cloning KL must be finite and non-negative.")
        if self._kl_budget_task_index != task_index:
            self._start_kl_budget_task(task_index=task_index, baseline_kl=raw_kl)
        assert self.kl_budget_ema is not None
        assert self.kl_budget_baseline is not None
        beta = self.kl_budget_ema_beta
        self.kl_budget_ema = beta * self.kl_budget_ema + (1.0 - beta) * raw_kl

        width = self.kl_budget_high - self.kl_budget_low
        self.kl_budget_excess = max(
            0.0,
            self.kl_budget_ema - self.kl_budget_baseline,
        )
        multiplier = min(
            1.0,
            max(0.0, (self.kl_budget_excess - self.kl_budget_low) / width),
        )
        self.kl_budget_multiplier = float(multiplier)
        self.bc_progress_multiplier = self.kl_budget_multiplier
        self._kl_budget_updates += 1
        self._kl_budget_multiplier_sum += self.kl_budget_multiplier
        self._kl_budget_active_updates += int(self.kl_budget_multiplier > 0.0)
        self._kl_budget_full_updates += int(self.kl_budget_multiplier >= 1.0)
        return self.kl_budget_multiplier

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
        if self._kl_budget_task_index != task_index:
            self._start_kl_budget_task(
                task_index=task_index,
                baseline_kl=self._measure_reference_kl(),
            )

        observations, target_means, target_log_stds = self._sample_episodic_batch()
        current_means, current_log_stds = self._cloning_distribution_parameters(
            observations
        )
        raw_cloning_loss = self._gaussian_kl(
            target_means,
            target_log_stds,
            current_means,
            current_log_stds,
        ).mean()
        self._update_kl_budget(
            raw_kl=float(raw_cloning_loss.detach().item()),
            task_index=task_index,
        )

        weighted_cloning_loss = self.actor_cloning_coefficient * raw_cloning_loss
        cloning_gradients = torch.autograd.grad(weighted_cloning_loss, parameters)
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
                bc_combination_strategy="kl_budget_adaptive_additive",
                bc_combination_scale=combination_scale,
            )
            self._record_source_task_diagnostics(
                sac_gradients=gradients,
                parameters=parameters,
                current_task_index=task_index,
            )
        return final_actor_gradients

    def gradient_diagnostics_summary(self) -> dict[str, float | int | bool]:
        summary = super().gradient_diagnostics_summary()
        updates = self._kl_budget_updates
        summary.update(
            {
                "kl_budget_enabled": True,
                "kl_budget_low": self.kl_budget_low,
                "kl_budget_high": self.kl_budget_high,
                "kl_budget_ema_beta": self.kl_budget_ema_beta,
                "kl_budget_latest_ema": (
                    float("nan") if self.kl_budget_ema is None else self.kl_budget_ema
                ),
                "kl_budget_baseline": (
                    float("nan")
                    if self.kl_budget_baseline is None
                    else self.kl_budget_baseline
                ),
                "kl_budget_latest_excess": self.kl_budget_excess,
                "kl_budget_latest_multiplier": self.kl_budget_multiplier,
                "kl_budget_mean_multiplier": (
                    0.0 if updates == 0 else self._kl_budget_multiplier_sum / updates
                ),
                "kl_budget_active_fraction": (
                    0.0 if updates == 0 else self._kl_budget_active_updates / updates
                ),
                "kl_budget_full_fraction": (
                    0.0 if updates == 0 else self._kl_budget_full_updates / updates
                ),
                "kl_budget_updates": updates,
            }
        )
        return summary
