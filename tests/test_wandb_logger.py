import tempfile
import unittest
from pathlib import Path

from crl_cw.utils.wandb_logger import WandbLogger


class TestWandbLogger(unittest.TestCase):
    def test_disabled_mode_is_a_safe_no_op(self) -> None:
        """Disabled mode must not import W&B or create a run."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            logger = WandbLogger(
                mode="disabled",
                project="crl-cw",
                entity=None,
                group="test",
                run_name="disabled-test",
                config={"seed": 0},
                run_directory=Path(temporary_directory),
            )

            self.assertFalse(logger.enabled)
            self.assertEqual(logger.mode, "disabled")
            self.assertIsNone(logger.run_url)

            logger.log_evaluation(
                global_step=1,
                gradient_updates=0,
                evaluation_index=1,
                active_task_index=0,
                active_task_name="task-a",
                active_task_step=1,
                training_metrics=None,
                stochastic_success={"task-a": 0.0},
                deterministic_success={"task-a": 0.0},
                stochastic_return={"task-a": 0.0},
                deterministic_return={"task-a": 0.0},
                elapsed_seconds=0.1,
            )
            logger.log_task_summary(
                global_step=1,
                task_index=0,
                task_name="task-a",
                task_gradient_updates=0,
                cumulative_gradient_updates=0,
                completed_training_episodes=0,
                mean_training_return=0.0,
                final_alpha=1.0,
                replay_buffer_size=1,
                elapsed_seconds=0.1,
            )
            logger.log_final_metrics(
                global_step=1,
                average_performance=0.0,
                average_forgetting=0.0,
                final_per_task={"task-a": 0.0},
                end_of_task_per_task={"task-a": 0.0},
                forgetting_per_task={"task-a": 0.0},
            )
            logger.log_results_artifact(
                name="test-results",
                files=[],
            )
            logger.finish(exit_code=0)

    def test_invalid_mode_is_rejected(self) -> None:
        """Only disabled, online, and offline modes are valid."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaises(ValueError):
                WandbLogger(
                    mode="unknown",
                    project="crl-cw",
                    entity=None,
                    group=None,
                    run_name="invalid-test",
                    config={},
                    run_directory=Path(temporary_directory),
                )

    def test_artifact_name_is_sanitized(self) -> None:
        """Artifact names must not retain spaces or path separators."""
        result = WandbLogger._sanitize_artifact_name(
            "CW10 / seed 0 results"
        )
        self.assertEqual(result, "CW10-seed-0-results")


if __name__ == "__main__":
    unittest.main()
