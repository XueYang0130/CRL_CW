"""Tests for aggregating independent single-task baseline curves."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from crl_cw.evaluation.single_task_baselines import (
    aggregate_single_task_baselines,
)


class SingleTaskBaselineAggregationTests(unittest.TestCase):
    def test_aggregate_writes_aligned_curves(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            batch_directory = Path(temporary_directory) / "batch"
            runs_directory = batch_directory / "runs"
            runs_directory.mkdir(parents=True)

            tasks = ["hammer-v1", "push-wall-v1"]
            batch_config = {
                "tasks": tasks,
                "seed": 0,
                "steps_per_task": 1_000,
                "eval_every": 500,
            }
            self._write_json(
                batch_directory / "batch_config.json",
                batch_config,
            )

            manifest_rows: list[dict[str, object]] = []
            for task_index, task_name in enumerate(tasks):
                run_name = f"task_{task_index + 1:02d}_{task_name}_seed0"
                run_directory = runs_directory / run_name
                run_directory.mkdir()

                self._write_json(
                    run_directory / "config.json",
                    {
                        "task": task_name,
                        "seed": 0,
                        "total_steps": 1_000,
                        "eval_every": 500,
                    },
                )
                self._write_json(
                    run_directory / "summary.json",
                    {"task_name": task_name},
                )
                self._write_evaluations(
                    run_directory / "evaluations.csv",
                    first_success=0.1 * task_index,
                    second_success=0.2 + 0.1 * task_index,
                )

                manifest_rows.append(
                    {
                        "task_index": task_index,
                        "task_name": task_name,
                        "status": "completed",
                        "run_name": run_name,
                        "run_directory": str(
                            run_directory.relative_to(batch_directory)
                        ),
                        "log_path": f"logs/{run_name}.log",
                        "return_code": 0,
                        "started_at_utc": "",
                        "finished_at_utc": "",
                        "elapsed_seconds": 1.0,
                        "error": "",
                    }
                )

            self._write_manifest(
                batch_directory / "batch_manifest.csv",
                manifest_rows,
            )

            result = aggregate_single_task_baselines(
                batch_directory=batch_directory,
                tail_size=1,
            )

            with result.curves_json_path.open("r", encoding="utf-8") as file:
                curves = json.load(file)

            self.assertEqual(curves["tasks"], tasks)
            self.assertEqual(curves["evaluation_steps"], [500, 1_000])
            actual_curves = curves["stochastic_success_curves"]
            expected_curves = [[0.0, 0.2], [0.1, 0.3]]

            self.assertEqual(len(actual_curves), len(expected_curves))

            for actual_curve, expected_curve in zip(
                actual_curves,
                expected_curves,
            ):
                self.assertEqual(len(actual_curve), len(expected_curve))

                for actual_value, expected_value in zip(
                    actual_curve,
                    expected_curve,
                ):
                    self.assertAlmostEqual(
                        actual_value,
                        expected_value,
                        places=7,
                    )
            self.assertTrue(result.stochastic_success_csv_path.is_file())
            self.assertTrue(result.task_summaries_csv_path.is_file())

    def test_aggregate_rejects_misaligned_steps(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            batch_directory = Path(temporary_directory) / "batch"
            runs_directory = batch_directory / "runs"
            runs_directory.mkdir(parents=True)
            tasks = ["hammer-v1", "push-wall-v1"]

            self._write_json(
                batch_directory / "batch_config.json",
                {
                    "tasks": tasks,
                    "seed": 0,
                    "steps_per_task": 1_000,
                    "eval_every": 500,
                },
            )

            manifest_rows: list[dict[str, object]] = []
            for task_index, task_name in enumerate(tasks):
                run_name = f"run_{task_index}"
                run_directory = runs_directory / run_name
                run_directory.mkdir()
                self._write_json(
                    run_directory / "config.json",
                    {
                        "task": task_name,
                        "seed": 0,
                        "total_steps": 1_000,
                        "eval_every": 500,
                    },
                )
                self._write_json(
                    run_directory / "summary.json",
                    {"task_name": task_name},
                )
                steps = [500, 1_000] if task_index == 0 else [400, 1_000]
                self._write_evaluations_with_steps(
                    run_directory / "evaluations.csv",
                    steps,
                )
                manifest_rows.append(
                    {
                        "task_index": task_index,
                        "task_name": task_name,
                        "status": "completed",
                        "run_name": run_name,
                        "run_directory": str(
                            run_directory.relative_to(batch_directory)
                        ),
                        "log_path": "",
                        "return_code": 0,
                        "started_at_utc": "",
                        "finished_at_utc": "",
                        "elapsed_seconds": 1.0,
                        "error": "",
                    }
                )

            self._write_manifest(
                batch_directory / "batch_manifest.csv",
                manifest_rows,
            )

            with self.assertRaisesRegex(ValueError, "same evaluation step grid"):
                aggregate_single_task_baselines(
                    batch_directory=batch_directory,
                )

    @staticmethod
    def _write_json(path: Path, value: dict[str, object]) -> None:
        with path.open("w", encoding="utf-8") as file:
            json.dump(value, file)

    @staticmethod
    def _write_manifest(
        path: Path,
        rows: list[dict[str, object]],
    ) -> None:
        fieldnames = list(rows[0])
        with path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _write_evaluations(
        self,
        path: Path,
        *,
        first_success: float,
        second_success: float,
    ) -> None:
        self._write_evaluations_with_steps(
            path,
            [500, 1_000],
            successes=[first_success, second_success],
        )

    @staticmethod
    def _write_evaluations_with_steps(
        path: Path,
        steps: list[int],
        *,
        successes: list[float] | None = None,
    ) -> None:
        success_values = successes or [0.0 for _ in steps]
        fieldnames = [
            "environment_step",
            "gradient_updates",
            "deterministic_average_return",
            "deterministic_success_rate",
            "deterministic_average_episode_length",
            "stochastic_average_return",
            "stochastic_success_rate",
            "stochastic_average_episode_length",
            "actor_loss",
            "q1_loss",
            "q2_loss",
            "alpha_loss",
            "alpha",
            "q1_mean",
            "q2_mean",
            "q_target_mean",
            "log_prob_mean",
            "elapsed_seconds",
        ]
        rows = []
        for index, (step, success) in enumerate(
            zip(steps, success_values, strict=True),
            start=1,
        ):
            rows.append(
                {
                    "environment_step": step,
                    "gradient_updates": index,
                    "deterministic_average_return": 1.0 + index,
                    "deterministic_success_rate": success,
                    "deterministic_average_episode_length": 200,
                    "stochastic_average_return": 2.0 + index,
                    "stochastic_success_rate": success,
                    "stochastic_average_episode_length": 200,
                    "actor_loss": "",
                    "q1_loss": "",
                    "q2_loss": "",
                    "alpha_loss": "",
                    "alpha": "",
                    "q1_mean": "",
                    "q2_mean": "",
                    "q_target_mean": "",
                    "log_prob_mean": "",
                    "elapsed_seconds": float(index),
                }
            )
        with path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
