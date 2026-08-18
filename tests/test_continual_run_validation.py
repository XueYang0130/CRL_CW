import unittest

from training.continual_experiment_copy import validate_baseline_curves_for_run


class TestContinualRunValidation(unittest.TestCase):
    def test_matching_baseline_schedule_is_accepted(self) -> None:
        validate_baseline_curves_for_run(
            [[0.0] * 25 for _ in range(10)],
            task_count=10,
            steps_per_task=500_000,
            eval_every=20_000,
        )

    def test_wrong_task_count_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            validate_baseline_curves_for_run(
                [[0.0] * 25 for _ in range(9)],
                task_count=10,
                steps_per_task=500_000,
                eval_every=20_000,
            )

    def test_wrong_checkpoint_count_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            validate_baseline_curves_for_run(
                [[0.0] * 24 for _ in range(10)],
                task_count=10,
                steps_per_task=500_000,
                eval_every=20_000,
            )

    def test_perfect_baseline_is_rejected_before_training(self) -> None:
        with self.assertRaises(ValueError):
            validate_baseline_curves_for_run(
                [[1.0] * 25],
                task_count=1,
                steps_per_task=500_000,
                eval_every=20_000,
            )


if __name__ == "__main__":
    unittest.main()
