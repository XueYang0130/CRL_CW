"""Optional Weights & Biases logging for CRL-CW experiments.

The rest of the project can run without W&B by using mode="disabled".
When W&B is enabled, this module logs scalar metrics, per-task metrics,
summary values, tables, result artifacts, and an optional model artifact.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


_ALLOWED_MODES = {"disabled", "online", "offline"}


class WandbLogger:
    """Small optional wrapper around the W&B Python SDK."""

    def __init__(
        self,
        *,
        mode: str,
        project: str,
        entity: str | None,
        group: str | None,
        run_name: str,
        config: Mapping[str, Any],
        run_directory: Path,
        tags: Sequence[str] | None = None,
        notes: str | None = None,
        log_code: bool = False,
        code_root: str | Path = ".",
    ) -> None:
        """Create an optional W&B run.

        Args:
            mode:
                One of ``disabled``, ``online``, or ``offline``.

            project:
                W&B project name.

            entity:
                Optional W&B username, team, or organization.

            group:
                Optional group used to compare related runs, such as all
                seeds of one method.

            run_name:
                Human-readable run name.

            config:
                Experiment configuration stored by W&B.

            run_directory:
                Local experiment output directory. W&B's own local files
                are stored below this directory.

            tags:
                Optional run tags.

            notes:
                Optional run notes.

            log_code:
                Whether to save Python source files as a code artifact.

            code_root:
                Root directory used by W&B code logging.
        """
        normalized_mode = str(mode).strip().lower()
        if normalized_mode not in _ALLOWED_MODES:
            raise ValueError(
                "mode must be one of "
                f"{sorted(_ALLOWED_MODES)}, got {mode!r}."
            )

        self._mode = normalized_mode
        self._wandb: Any | None = None
        self._run: Any | None = None

        if normalized_mode == "disabled":
            return

        try:
            import wandb  # type: ignore[import-not-found]
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "W&B logging was enabled, but the 'wandb' package is not "
                "installed. Run: python -m pip install wandb"
            ) from error

        self._wandb = wandb
        self._run = wandb.init(
            project=project,
            entity=entity,
            group=group,
            name=run_name,
            config=dict(config),
            mode=normalized_mode,
            dir=str(run_directory),
            tags=list(tags) if tags else None,
            notes=notes,
            job_type="training",
        )

        if self._run is None:
            raise RuntimeError("wandb.init() did not return a run object.")

        self._define_metrics()

        if log_code:
            self._run.log_code(root=str(code_root))

    @property
    def enabled(self) -> bool:
        """Return whether W&B logging is active."""
        return self._run is not None

    @property
    def mode(self) -> str:
        """Return the configured W&B mode."""
        return self._mode

    @property
    def run_url(self) -> str | None:
        """Return the W&B run URL when available."""
        if self._run is None:
            return None
        url = getattr(self._run, "url", None)
        return None if url is None else str(url)

    def _define_metrics(self) -> None:
        """Use global environment steps as the x-axis."""
        if self._run is None:
            return

        self._run.define_metric("global_step")
        for metric_glob in (
            "train/*",
            "eval/*",
            "stochastic_success/*",
            "deterministic_success/*",
            "stochastic_return/*",
            "deterministic_return/*",
            "task/*",
        ):
            self._run.define_metric(
                metric_glob,
                step_metric="global_step",
            )

    def log_evaluation(
        self,
        *,
        global_step: int,
        gradient_updates: int,
        evaluation_index: int,
        active_task_index: int,
        active_task_name: str,
        active_task_step: int,
        training_metrics: Mapping[str, float] | None,
        stochastic_success: Mapping[str, float],
        deterministic_success: Mapping[str, float],
        stochastic_return: Mapping[str, float],
        deterministic_return: Mapping[str, float],
        elapsed_seconds: float,
    ) -> None:
        """Log one evaluation with optional stochastic diagnostics.

        Deterministic success and return may be used as the formal
        evaluation protocol. Stochastic success and return are optional
        and may both be empty when stochastic evaluation is disabled.
        """
        if self._run is None:
            return

        stochastic_values = (
            self._validated_values(
                stochastic_success,
                name="stochastic_success",
            )
            if stochastic_success
            else {}
        )
        stochastic_returns = (
            self._validated_values(
                stochastic_return,
                name="stochastic_return",
            )
            if stochastic_return
            else {}
        )
        deterministic_values = (
            self._validated_values(
                deterministic_success,
                name="deterministic_success",
            )
            if deterministic_success
            else {}
        )
        deterministic_returns = (
            self._validated_values(
                deterministic_return,
                name="deterministic_return",
            )
            if deterministic_return
            else {}
        )

        if bool(stochastic_values) != bool(stochastic_returns):
            raise ValueError(
                "stochastic_success and stochastic_return must "
                "either both be empty or both contain values."
            )

        if (
            stochastic_values
            and stochastic_values.keys()
            != stochastic_returns.keys()
        ):
            raise ValueError(
                "stochastic_success and stochastic_return must "
                "have identical task keys."
            )

        if bool(deterministic_values) != bool(deterministic_returns):
            raise ValueError(
                "deterministic_success and deterministic_return must "
                "either both be empty or both contain values."
            )

        if (
            deterministic_values
            and deterministic_values.keys()
            != deterministic_returns.keys()
        ):
            raise ValueError(
                "deterministic_success and deterministic_return must "
                "have identical task keys."
            )

        if not stochastic_values and not deterministic_values:
            raise ValueError(
                "At least one evaluation mode must contain values."
            )

        if (
            stochastic_values
            and active_task_name not in stochastic_values
        ):
            raise KeyError(
                f"Active task {active_task_name!r} is absent from "
                "stochastic success values."
            )

        if (
            stochastic_returns
            and active_task_name not in stochastic_returns
        ):
            raise KeyError(
                f"Active task {active_task_name!r} is absent from "
                "stochastic return values."
            )

        if (
            deterministic_values
            and active_task_name not in deterministic_values
        ):
            raise KeyError(
                f"Active task {active_task_name!r} is absent from "
                "deterministic success values."
            )

        if (
            deterministic_returns
            and active_task_name not in deterministic_returns
        ):
            raise KeyError(
                f"Active task {active_task_name!r} is absent from "
                "deterministic return values."
            )

        payload: dict[str, Any] = {
            "global_step": int(global_step),
            "eval/evaluation_index": int(evaluation_index),
            "eval/gradient_updates": int(gradient_updates),
            "eval/active_task_index": int(active_task_index),
            "eval/active_task_step": int(active_task_step),
            "eval/active_task_name": active_task_name,
            "eval/elapsed_seconds": self._finite_float(
                elapsed_seconds,
                name="elapsed_seconds",
            ),
        }

        if stochastic_values:
            payload["eval/average_stochastic_success"] = self._mean(
                stochastic_values.values()
            )
            payload["eval/active_stochastic_success"] = (
                stochastic_values[active_task_name]
            )

        if deterministic_values:
            payload["eval/average_deterministic_success"] = self._mean(
                deterministic_values.values()
            )
            payload["eval/active_deterministic_success"] = (
                deterministic_values[active_task_name]
            )

        for task_name, value in stochastic_values.items():
            payload[f"stochastic_success/{task_name}"] = value
        for task_name, value in deterministic_values.items():
            payload[f"deterministic_success/{task_name}"] = value
        for task_name, value in stochastic_returns.items():
            payload[f"stochastic_return/{task_name}"] = value
        for task_name, value in deterministic_returns.items():
            payload[f"deterministic_return/{task_name}"] = value

        if training_metrics is not None:
            for metric_name, metric_value in training_metrics.items():
                payload[f"train/{metric_name}"] = self._finite_float(
                    metric_value,
                    name=f"training_metrics[{metric_name!r}]",
                )

        self._run.log(payload)

    def log_task_summary(
        self,
        *,
        global_step: int,
        task_index: int,
        task_name: str,
        task_gradient_updates: int,
        cumulative_gradient_updates: int,
        completed_training_episodes: int,
        mean_training_return: float,
        final_alpha: float,
        replay_buffer_size: int,
        elapsed_seconds: float,
    ) -> None:
        """Log one task-boundary summary."""
        if self._run is None:
            return

        self._run.log(
            {
                "global_step": int(global_step),
                "task/index": int(task_index),
                "task/name": task_name,
                "task/gradient_updates": int(task_gradient_updates),
                "task/cumulative_gradient_updates": int(
                    cumulative_gradient_updates
                ),
                "task/completed_training_episodes": int(
                    completed_training_episodes
                ),
                "task/mean_training_return": self._finite_float(
                    mean_training_return,
                    name="mean_training_return",
                ),
                "task/final_alpha": self._finite_float(
                    final_alpha,
                    name="final_alpha",
                ),
                "task/replay_buffer_size": int(replay_buffer_size),
                "task/elapsed_seconds": self._finite_float(
                    elapsed_seconds,
                    name="elapsed_seconds",
                ),
            }
        )

    def log_final_metrics(
        self,
        *,
        global_step: int,
        average_performance: float,
        average_forgetting: float,
        final_per_task: Mapping[str, float],
        end_of_task_per_task: Mapping[str, float],
        forgetting_per_task: Mapping[str, float],
    ) -> None:
        """Log final continual-learning metrics and a per-task table."""
        if self._run is None or self._wandb is None:
            return

        average_performance_value = self._finite_float(
            average_performance,
            name="average_performance",
        )
        average_forgetting_value = self._finite_float(
            average_forgetting,
            name="average_forgetting",
        )

        final_values = self._validated_values(
            final_per_task,
            name="final_per_task",
        )
        end_values = self._validated_values(
            end_of_task_per_task,
            name="end_of_task_per_task",
        )
        forgetting_values = self._validated_values(
            forgetting_per_task,
            name="forgetting_per_task",
        )

        if not (
            final_values.keys()
            == end_values.keys()
            == forgetting_values.keys()
        ):
            raise ValueError(
                "Final, end-of-task, and forgetting mappings must have "
                "identical task keys."
            )

        self._run.log(
            {
                "global_step": int(global_step),
                "final/average_performance": average_performance_value,
                "final/average_forgetting": average_forgetting_value,
            }
        )

        self._run.summary["average_performance"] = average_performance_value
        self._run.summary["average_forgetting"] = average_forgetting_value
        self._run.summary["forward_transfer"] = "unavailable"

        table = self._wandb.Table(
            columns=[
                "task_name",
                "end_of_task_success",
                "final_success",
                "forgetting",
            ]
        )
        for task_name in final_values:
            table.add_data(
                task_name,
                end_values[task_name],
                final_values[task_name],
                forgetting_values[task_name],
            )

        self._run.log({"final/per_task_metrics": table})

    def log_results_artifact(
        self,
        *,
        name: str,
        files: Sequence[str | Path],
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Upload local result files as one versioned W&B artifact."""
        if self._run is None or self._wandb is None:
            return

        artifact = self._wandb.Artifact(
            name=self._sanitize_artifact_name(name),
            type="results",
            metadata=dict(metadata) if metadata is not None else None,
        )

        added_files = 0
        for file_path in files:
            path = Path(file_path)
            if not path.is_file():
                continue
            artifact.add_file(local_path=str(path))
            added_files += 1

        if added_files > 0:
            self._run.log_artifact(
                artifact,
                aliases=["latest"],
            )

    def log_model_artifact(
        self,
        *,
        name: str,
        checkpoint_path: str | Path,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Upload one final checkpoint as a model artifact."""
        if self._run is None or self._wandb is None:
            return

        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Checkpoint does not exist: {path}"
            )

        artifact = self._wandb.Artifact(
            name=self._sanitize_artifact_name(name),
            type="model",
            metadata=dict(metadata) if metadata is not None else None,
        )
        artifact.add_file(local_path=str(path))
        self._run.log_artifact(
            artifact,
            aliases=["latest", "final"],
        )

    def finish(self, *, exit_code: int = 0) -> None:
        """Finish and flush the W&B run."""
        if self._run is None:
            return

        run = self._run
        self._run = None
        run.finish(exit_code=int(exit_code))

    @staticmethod
    def _validated_values(
        values: Mapping[str, float],
        *,
        name: str,
    ) -> dict[str, float]:
        if not values:
            raise ValueError(f"{name} must not be empty.")

        return {
            str(key): WandbLogger._finite_float(
                value,
                name=f"{name}[{key!r}]",
            )
            for key, value in values.items()
        }

    @staticmethod
    def _finite_float(value: float, *, name: str) -> float:
        converted = float(value)
        if not math.isfinite(converted):
            raise ValueError(f"{name} must be finite, got {converted}.")
        return converted

    @staticmethod
    def _mean(values: Any) -> float:
        converted = [float(value) for value in values]
        if not converted:
            raise ValueError("Cannot calculate a mean from no values.")
        return float(sum(converted) / len(converted))

    @staticmethod
    def _sanitize_artifact_name(name: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip("-.")
        if not cleaned:
            raise ValueError("Artifact name becomes empty after sanitization.")
        return cleaned
