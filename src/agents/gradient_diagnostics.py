from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import torch
from torch import nn


GRADIENT_WINDOW_FIELDS = (
    "update_index",
    "evaluation_index",
    "global_step",
    "active_task_step",
    "current_task_index",
    "current_task_name",
    "scope",
    "reference_memory_states",
    "raw_bc_loss",
    "weighted_bc_loss",
    "sac_actor_loss",
    "weighted_bc_to_abs_sac_loss_ratio",
    "bc_coefficient",
    "gradient_strategy",
    "projection_applied",
    "bc_norm_scale",
    "bc_combination_strategy",
    "bc_combination_scale",
    "sac_gradient_norm",
    "bc_gradient_norm",
    "combined_gradient_norm",
    "bc_to_sac_norm_ratio",
    "combined_to_sac_norm_ratio",
    "dot_product",
    "cosine_similarity",
    "conflict",
    "conflict_mass",
    "bc_parallel_norm",
    "bc_orthogonal_norm",
    "applied_bc_gradient_norm",
    "applied_bc_to_sac_norm_ratio",
    "applied_cosine_similarity",
    "actor_gradient_clip_norm",
    "gradient_clip_scale",
)

GRADIENT_LAYER_FIELDS = GRADIENT_WINDOW_FIELDS + ("layer",)
GRADIENT_TASK_PAIR_FIELDS = GRADIENT_WINDOW_FIELDS + (
    "source_task_index",
    "source_task_name",
    "source_memory_states",
)


def _gradient_metrics(
    sac_gradients: tuple[torch.Tensor, ...],
    bc_gradients: tuple[torch.Tensor, ...],
    indices: Iterable[int],
) -> dict[str, float | bool]:
    selected = tuple(indices)
    if not selected:
        raise ValueError("At least one gradient index is required.")
    epsilon = 1e-12
    sac_norm_sq = sum(sac_gradients[index].square().sum() for index in selected)
    bc_norm_sq = sum(bc_gradients[index].square().sum() for index in selected)
    dot = sum(
        (sac_gradients[index] * bc_gradients[index]).sum()
        for index in selected
    )
    combined_norm_sq = sum(
        ((sac_gradients[index] + bc_gradients[index]) / 2.0).square().sum()
        for index in selected
    )
    sac_norm = sac_norm_sq.sqrt()
    bc_norm = bc_norm_sq.sqrt()
    combined_norm = combined_norm_sq.sqrt()
    cosine = dot / (sac_norm * bc_norm + epsilon)
    signed_parallel = dot / (sac_norm + epsilon)
    parallel_norm = signed_parallel.abs()
    orthogonal_sq = torch.clamp(
        bc_norm_sq - signed_parallel.square(),
        min=0.0,
    )
    conflict_mass = torch.relu(-dot) / (sac_norm * bc_norm + epsilon)
    result = {
        "sac_gradient_norm": float(sac_norm.detach().item()),
        "bc_gradient_norm": float(bc_norm.detach().item()),
        "combined_gradient_norm": float(combined_norm.detach().item()),
        "bc_to_sac_norm_ratio": float((bc_norm / (sac_norm + epsilon)).detach().item()),
        "combined_to_sac_norm_ratio": float(
            (combined_norm / (sac_norm + epsilon)).detach().item()
        ),
        "dot_product": float(dot.detach().item()),
        "cosine_similarity": float(cosine.detach().item()),
        "conflict": bool(dot.detach().item() < 0.0),
        "conflict_mass": float(conflict_mass.detach().item()),
        "bc_parallel_norm": float(parallel_norm.detach().item()),
        "bc_orthogonal_norm": float(orthogonal_sq.sqrt().detach().item()),
    }
    return result


def _gradient_norm(
    gradients: tuple[torch.Tensor, ...],
    indices: Iterable[int],
) -> torch.Tensor:
    selected = tuple(indices)
    if not selected:
        raise ValueError("At least one gradient index is required.")
    return sum(gradients[index].square().sum() for index in selected).sqrt()


class GradientConflictDiagnostics:
    """Sparse, read-only diagnostics for SAC and behavior-cloning gradients."""

    def __init__(
        self,
        *,
        actor: nn.Module,
        enabled: bool,
        interval: int,
        source_batch_size: int,
        gradient_clip_norm: float | None,
        seed: int,
    ) -> None:
        if interval <= 0:
            raise ValueError("gradient diagnostic interval must be positive.")
        if source_batch_size <= 0:
            raise ValueError("gradient diagnostic source batch size must be positive.")
        self.enabled = bool(enabled)
        self.interval = int(interval)
        self.source_batch_size = int(source_batch_size)
        self.gradient_clip_norm = gradient_clip_norm
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(int(seed))
        self.update_index = 0
        self._window_rows: list[dict[str, object]] = []
        self._layer_rows: list[dict[str, object]] = []
        self._task_pair_rows: list[dict[str, object]] = []
        self._summary: dict[str, list[float]] = defaultdict(list)

        parameters = tuple(actor.parameters())
        backbone_ids = {id(parameter) for parameter in actor.backbone.parameters()}
        self.shared_indices = tuple(
            index for index, parameter in enumerate(parameters) if id(parameter) in backbone_ids
        )
        if not self.shared_indices:
            raise RuntimeError("Actor backbone parameters were not found.")
        self.all_indices = tuple(range(len(parameters)))

        names = tuple(name for name, _ in actor.named_parameters())
        grouped: dict[str, list[int]] = defaultdict(list)
        for index in self.shared_indices:
            name = names[index]
            if "hidden_layers." in name:
                layer_index = name.split("hidden_layers.", 1)[1].split(".", 1)[0]
                group = f"backbone_layer_{layer_index}"
            elif "first_layer_norm" in name:
                group = "backbone_layer_0"
            else:
                group = "backbone_other"
            grouped[group].append(index)
        self.layer_indices = {
            layer: tuple(indices) for layer, indices in sorted(grouped.items())
        }

    def begin_bc_update(self) -> bool:
        self.update_index += 1
        return self.enabled and self.update_index % self.interval == 0

    def record(
        self,
        *,
        sac_gradients: tuple[torch.Tensor, ...],
        bc_gradients: tuple[torch.Tensor, ...],
        current_task_index: int,
        raw_bc_loss: float,
        bc_coefficient: float,
        sac_actor_loss: float,
        reference_memory_states: int,
        applied_bc_gradients: tuple[torch.Tensor, ...],
        final_actor_gradients: tuple[torch.Tensor, ...],
        gradient_strategy: str,
        projection_applied: bool,
        bc_norm_scale: float,
        bc_combination_strategy: str,
        bc_combination_scale: float,
    ) -> None:
        base = {
            "update_index": self.update_index,
            "evaluation_index": None,
            "global_step": None,
            "active_task_step": None,
            "current_task_index": current_task_index,
            "reference_memory_states": reference_memory_states,
            "raw_bc_loss": raw_bc_loss,
            "weighted_bc_loss": raw_bc_loss * bc_coefficient,
            "sac_actor_loss": sac_actor_loss,
            "weighted_bc_to_abs_sac_loss_ratio": (
                raw_bc_loss * bc_coefficient / (abs(sac_actor_loss) + 1e-12)
            ),
            "bc_coefficient": bc_coefficient,
            "gradient_strategy": gradient_strategy,
            "projection_applied": projection_applied,
            "bc_norm_scale": bc_norm_scale,
            "bc_combination_strategy": bc_combination_strategy,
            "bc_combination_scale": bc_combination_scale,
            "actor_gradient_clip_norm": self.gradient_clip_norm,
        }
        metrics = _gradient_metrics(sac_gradients, bc_gradients, self.shared_indices)
        applied_metrics = _gradient_metrics(
            sac_gradients,
            applied_bc_gradients,
            self.shared_indices,
        )
        shared_sac_norm = _gradient_norm(sac_gradients, self.shared_indices)
        shared_final_norm = _gradient_norm(final_actor_gradients, self.shared_indices)
        shared_row = {
            **base,
            "scope": "shared_actor_backbone",
            **metrics,
            "combined_gradient_norm": float(shared_final_norm.detach().item()),
            "combined_to_sac_norm_ratio": float(
                (shared_final_norm / (shared_sac_norm + 1e-12)).detach().item()
            ),
            "applied_bc_gradient_norm": applied_metrics["bc_gradient_norm"],
            "applied_bc_to_sac_norm_ratio": applied_metrics[
                "bc_to_sac_norm_ratio"
            ],
            "applied_cosine_similarity": applied_metrics["cosine_similarity"],
            "gradient_clip_scale": None,
        }
        full_metrics = _gradient_metrics(sac_gradients, bc_gradients, self.all_indices)
        applied_full_metrics = _gradient_metrics(
            sac_gradients,
            applied_bc_gradients,
            self.all_indices,
        )
        full_sac_norm = _gradient_norm(sac_gradients, self.all_indices)
        full_final_norm = _gradient_norm(final_actor_gradients, self.all_indices)
        full_norm = float(full_final_norm.detach().item())
        clip_scale = (
            1.0
            if self.gradient_clip_norm is None
            else min(1.0, self.gradient_clip_norm / (full_norm + 1e-6))
        )
        self._window_rows.extend(
            (
                shared_row,
                {
                    **base,
                    "scope": "full_actor",
                    **full_metrics,
                    "combined_gradient_norm": full_norm,
                    "combined_to_sac_norm_ratio": float(
                        (full_final_norm / (full_sac_norm + 1e-12)).detach().item()
                    ),
                    "applied_bc_gradient_norm": applied_full_metrics[
                        "bc_gradient_norm"
                    ],
                    "applied_bc_to_sac_norm_ratio": applied_full_metrics[
                        "bc_to_sac_norm_ratio"
                    ],
                    "applied_cosine_similarity": applied_full_metrics[
                        "cosine_similarity"
                    ],
                    "gradient_clip_scale": clip_scale,
                },
            )
        )
        for key in (
            "raw_bc_loss",
            "sac_gradient_norm",
            "bc_gradient_norm",
            "combined_gradient_norm",
            "bc_to_sac_norm_ratio",
            "cosine_similarity",
            "conflict_mass",
        ):
            self._summary[key].append(float(shared_row[key]))
        self._summary["conflict"].append(float(bool(shared_row["conflict"])))

        for layer, indices in self.layer_indices.items():
            raw_layer_metrics = _gradient_metrics(
                sac_gradients,
                bc_gradients,
                indices,
            )
            applied_layer_metrics = _gradient_metrics(
                sac_gradients,
                applied_bc_gradients,
                indices,
            )
            layer_sac_norm = _gradient_norm(sac_gradients, indices)
            layer_final_norm = _gradient_norm(final_actor_gradients, indices)
            self._layer_rows.append(
                {
                    **base,
                    "scope": "shared_actor_backbone_layer",
                    **raw_layer_metrics,
                    "combined_gradient_norm": float(layer_final_norm.detach().item()),
                    "combined_to_sac_norm_ratio": float(
                        (layer_final_norm / (layer_sac_norm + 1e-12)).detach().item()
                    ),
                    "applied_bc_gradient_norm": applied_layer_metrics[
                        "bc_gradient_norm"
                    ],
                    "applied_bc_to_sac_norm_ratio": applied_layer_metrics[
                        "bc_to_sac_norm_ratio"
                    ],
                    "applied_cosine_similarity": applied_layer_metrics[
                        "cosine_similarity"
                    ],
                    "gradient_clip_scale": None,
                    "layer": layer,
                }
            )

    def record_task_pair(
        self,
        *,
        sac_gradients: tuple[torch.Tensor, ...],
        bc_gradients: tuple[torch.Tensor, ...],
        current_task_index: int,
        source_task_index: int,
        raw_bc_loss: float,
        bc_coefficient: float,
        sac_actor_loss: float,
        reference_memory_states: int,
        source_memory_states: int,
    ) -> None:
        metrics = _gradient_metrics(
            sac_gradients,
            bc_gradients,
            self.shared_indices,
        )
        self._task_pair_rows.append(
            {
                "update_index": self.update_index,
                "evaluation_index": None,
                "global_step": None,
                "active_task_step": None,
                "current_task_index": current_task_index,
                "scope": "shared_actor_backbone",
                "reference_memory_states": reference_memory_states,
                "raw_bc_loss": raw_bc_loss,
                "weighted_bc_loss": raw_bc_loss * bc_coefficient,
                "sac_actor_loss": sac_actor_loss,
                "weighted_bc_to_abs_sac_loss_ratio": (
                    raw_bc_loss * bc_coefficient / (abs(sac_actor_loss) + 1e-12)
                ),
                "bc_coefficient": bc_coefficient,
                "gradient_strategy": "source_attribution_raw",
                "projection_applied": False,
                "bc_norm_scale": 1.0,
                "bc_combination_strategy": "source_attribution_raw",
                "bc_combination_scale": 1.0,
                **metrics,
                "applied_bc_gradient_norm": metrics["bc_gradient_norm"],
                "applied_bc_to_sac_norm_ratio": metrics["bc_to_sac_norm_ratio"],
                "applied_cosine_similarity": metrics["cosine_similarity"],
                "actor_gradient_clip_norm": self.gradient_clip_norm,
                "gradient_clip_scale": None,
                "source_task_index": source_task_index,
                "source_memory_states": source_memory_states,
            }
        )

    def sample_indices(self, candidates: torch.Tensor) -> torch.Tensor:
        if candidates.ndim != 1 or candidates.numel() == 0:
            raise ValueError("Diagnostic candidates must be a non-empty vector.")
        positions = torch.randint(
            candidates.numel(),
            (self.source_batch_size,),
            generator=self.generator,
        )
        return candidates[positions]

    def drain(self) -> dict[str, list[dict[str, object]]]:
        rows = {
            "windows": self._window_rows,
            "layers": self._layer_rows,
            "task_pairs": self._task_pair_rows,
        }
        self._window_rows = []
        self._layer_rows = []
        self._task_pair_rows = []
        return rows

    def summary(self) -> dict[str, float | int | bool]:
        result: dict[str, float | int | bool] = {
            "enabled": self.enabled,
            "bc_updates_seen": self.update_index,
            "diagnostic_samples": len(self._summary.get("cosine_similarity", [])),
            "interval": self.interval,
            "source_batch_size": self.source_batch_size,
        }
        for key, values in self._summary.items():
            if values:
                result[f"mean_{key}"] = sum(values) / len(values)
                result[f"min_{key}"] = min(values)
                result[f"max_{key}"] = max(values)
        return result
