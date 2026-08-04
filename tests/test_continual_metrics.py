import unittest

import numpy as np

from evaluation import (
    compute_area_forward_transfer,
    compute_continual_metrics,
    compute_end_of_task_performance,
    compute_final_task_performance,
    compute_forward_transfer,
    mean_last_success,
    summarize_continual_run,
)


class TestContinualMetrics(unittest.TestCase):
    def test_mean_last_success_uses_last_five_points(
        self,
    ) -> None:
        """Only the final five evaluation points should be averaged."""
        result = mean_last_success(
            [
                0.0,
                0.1,
                0.2,
                0.4,
                0.6,
                0.8,
                1.0,
            ],
            count=5,
        )

        # Last five:
        #
        # 0.2, 0.4, 0.6, 0.8, 1.0
        #
        # Mean = 0.6
        self.assertAlmostEqual(
            result,
            0.6,
            places=7,
        )

    def test_mean_last_success_uses_all_available_points(
        self,
    ) -> None:
        """Short runs may contain fewer than five evaluations."""
        result = mean_last_success(
            [
                0.2,
                0.6,
            ],
            count=5,
        )

        self.assertAlmostEqual(
            result,
            0.4,
            places=7,
        )

    def test_final_task_performance_averages_tail_rows(
        self,
    ) -> None:
        """Final performance must be calculated separately per task."""
        final_evaluations = np.asarray(
            [
                [
                    0.6,
                    0.2,
                ],
                [
                    0.8,
                    0.4,
                ],
                [
                    1.0,
                    0.6,
                ],
            ],
            dtype=np.float64,
        )

        result = compute_final_task_performance(
            final_evaluations,
            tail_size=2,
        )

        # Task 0:
        # mean(0.8, 1.0) = 0.9
        #
        # Task 1:
        # mean(0.4, 0.6) = 0.5
        self.assertEqual(
            result,
            (
                0.9,
                0.5,
            ),
        )

    def test_end_of_task_performance_uses_each_curve_tail(
        self,
    ) -> None:
        """Every task must use the tail of its own active-task curve."""
        result = (
            compute_end_of_task_performance(
                [
                    [
                        0.1,
                        0.5,
                        0.9,
                        1.0,
                        0.9,
                    ],
                    [
                        0.0,
                        0.2,
                        0.6,
                        0.8,
                        0.7,
                    ],
                ],
                tail_size=2,
            )
        )

        self.assertEqual(
            result,
            (
                0.95,
                0.75,
            ),
        )

    def test_forward_transfer_matches_reference_formula(
        self,
    ) -> None:
        """Raw and normalized forward transfer must match CW formula."""
        raw, normalized = (
            compute_forward_transfer(
                active_task_success_curves=[
                    [
                        0.0,
                        0.5,
                        1.0,
                    ],
                    [
                        0.2,
                        0.4,
                        0.6,
                    ],
                ],
                baseline_success_curves=[
                    [
                        0.0,
                        0.25,
                        0.5,
                    ],
                    [
                        0.1,
                        0.2,
                        0.3,
                    ],
                ],
            )
        )

        # Task 0:
        #
        # continual mean = 0.5
        # baseline mean  = 0.25
        # raw FT         = 0.25
        # normalized FT  = 0.25 / 0.75 = 1/3
        self.assertAlmostEqual(
            raw[0],
            0.25,
            places=7,
        )

        self.assertAlmostEqual(
            normalized[0],
            1.0 / 3.0,
            places=7,
        )

        # Task 1:
        #
        # continual mean = 0.4
        # baseline mean  = 0.2
        # raw FT         = 0.2
        # normalized FT  = 0.2 / 0.8 = 0.25
        self.assertAlmostEqual(
            raw[1],
            0.2,
            places=7,
        )

        self.assertAlmostEqual(
            normalized[1],
            0.25,
            places=7,
        )

    def test_complete_metric_summary(
        self,
    ) -> None:
        """Complete result must combine performance, forgetting and FT."""
        result = compute_continual_metrics(
            final_evaluation_successes=[
                [
                    0.6,
                    0.2,
                ],
                [
                    0.8,
                    0.4,
                ],
                [
                    1.0,
                    0.6,
                ],
            ],
            active_task_success_curves=[
                [
                    0.1,
                    0.5,
                    0.9,
                    1.0,
                    0.9,
                ],
                [
                    0.0,
                    0.2,
                    0.6,
                    0.8,
                    0.7,
                ],
            ],
            baseline_success_curves=[
                [
                    0.0,
                    0.2,
                    0.4,
                    0.6,
                    0.8,
                ],
                [
                    0.0,
                    0.1,
                    0.2,
                    0.3,
                    0.4,
                ],
            ],
            tail_size=2,
        )

        self.assertEqual(
            result.num_tasks,
            2,
        )

        # Final task performance:
        #
        # task 0 = mean(0.8, 1.0) = 0.9
        # task 1 = mean(0.4, 0.6) = 0.5
        self.assertEqual(
            result.final_per_task,
            (
                0.9,
                0.5,
            ),
        )

        self.assertAlmostEqual(
            result.average_performance,
            0.7,
            places=7,
        )

        # End-of-task:
        #
        # task 0 = mean(1.0, 0.9) = 0.95
        # task 1 = mean(0.8, 0.7) = 0.75
        #
        # Forgetting:
        #
        # task 0 = 0.95 - 0.9 = 0.05
        # task 1 = 0.75 - 0.5 = 0.25
        self.assertAlmostEqual(
            result.forgetting_per_task[0],
            0.05,
            places=7,
        )

        self.assertAlmostEqual(
            result.forgetting_per_task[1],
            0.25,
            places=7,
        )

        self.assertAlmostEqual(
            result.average_forgetting,
            0.15,
            places=7,
        )

        self.assertTrue(
            np.isfinite(
                result.average_forward_transfer
            )
        )

        self.assertEqual(
            set(result.as_dict()),
            {
                "average_performance",
                "average_forgetting",
                "forward_transfer",
                "raw_forward_transfer",
                "area_forward_transfer",
            },
        )

    def test_area_forward_transfer_matches_trapezoid_formula(
        self,
    ) -> None:
        result = compute_area_forward_transfer(
            active_task_success_curves=[
                [
                    0.0,
                    0.6,
                    1.0,
                ],
            ],
            baseline_success_curves=[
                [
                    0.0,
                    0.2,
                    0.4,
                ],
            ],
        )

        # Numerator:
        # trapz([0.0, 0.4, 0.6]) = 0.7
        #
        # Denominator:
        # trapz([1.0, 0.8, 0.6]) = 1.6
        #
        # area FT = 0.7 / 1.6 = 0.4375
        self.assertAlmostEqual(
            result[0],
            0.4375,
            places=7,
        )

    def test_complete_metric_summary_includes_area_forward_transfer(
        self,
    ) -> None:
        result = compute_continual_metrics(
            final_evaluation_successes=[
                [
                    0.6,
                    0.2,
                ],
                [
                    0.8,
                    0.4,
                ],
                [
                    1.0,
                    0.6,
                ],
            ],
            active_task_success_curves=[
                [
                    0.1,
                    0.5,
                    0.9,
                    1.0,
                    0.9,
                ],
                [
                    0.0,
                    0.2,
                    0.6,
                    0.8,
                    0.7,
                ],
            ],
            baseline_success_curves=[
                [
                    0.0,
                    0.2,
                    0.4,
                    0.6,
                    0.8,
                ],
                [
                    0.0,
                    0.1,
                    0.2,
                    0.3,
                    0.4,
                ],
            ],
            tail_size=2,
        )

        self.assertEqual(
            len(result.area_forward_transfer_per_task),
            2,
        )
        self.assertTrue(
            np.isfinite(
                result.average_area_forward_transfer
            )
        )

    def test_negative_forgetting_is_not_clipped(
        self,
    ) -> None:
        """Later improvement should produce negative forgetting."""
        result = compute_continual_metrics(
            final_evaluation_successes=[
                [
                    0.6,
                ],
            ],
            active_task_success_curves=[
                [
                    0.2,
                    0.4,
                ],
            ],
            baseline_success_curves=[
                [
                    0.1,
                    0.2,
                ],
            ],
            tail_size=1,
        )

        # End-of-task = 0.4
        # Final = 0.6
        # Forgetting = -0.2
        self.assertAlmostEqual(
            result.average_forgetting,
            -0.2,
            places=7,
        )

    def test_mismatched_curve_lengths_are_rejected(
        self,
    ) -> None:
        """CL and baseline curves need matching evaluation schedules."""
        with self.assertRaises(
            ValueError
        ):
            compute_forward_transfer(
                active_task_success_curves=[
                    [
                        0.0,
                        0.5,
                    ],
                ],
                baseline_success_curves=[
                    [
                        0.0,
                        0.25,
                        0.5,
                    ],
                ],
            )

    def test_invalid_success_rate_is_rejected(
        self,
    ) -> None:
        """Success rates must remain between zero and one."""
        with self.assertRaises(
            ValueError
        ):
            mean_last_success(
                [
                    0.5,
                    1.2,
                ]
            )

    def test_perfect_baseline_mean_produces_undefined_normalized_ft(
        self,
    ) -> None:
        """Normalized FT is undefined with zero remaining headroom."""
        raw, normalized = compute_forward_transfer(
            active_task_success_curves=[[1.0, 1.0]],
            baseline_success_curves=[[1.0, 1.0]],
        )
        self.assertEqual(raw, (0.0,))
        self.assertTrue(np.isnan(normalized[0]))

    def test_zero_headroom_area_produces_undefined_area_ft(
        self,
    ) -> None:
        result = compute_area_forward_transfer(
            active_task_success_curves=[[1.0, 1.0]],
            baseline_success_curves=[[1.0, 1.0]],
        )
        self.assertTrue(np.isnan(result[0]))

    def test_aggregate_metrics_allow_undefined_area_ft(self) -> None:
        result = compute_continual_metrics(
            final_evaluation_successes=[[0.0, 0.0]],
            active_task_success_curves=[[0.0], [0.0]],
            baseline_success_curves=[[0.0], [0.0]],
            tail_size=1,
        )
        self.assertTrue(np.isnan(result.average_area_forward_transfer))
        self.assertTrue(all(np.isnan(result.area_forward_transfer_per_task)))

    def test_summary_without_baselines_keeps_non_transfer_metrics(self) -> None:
        summary = summarize_continual_run(
            final_success_rows=[[0.2, 0.4]],
            active_task_success_curves=[[0.2], [0.4]],
            baseline_success_curves=None,
            tail_size=1,
        )
        self.assertFalse(summary["forward_transfer_available"])
        self.assertIsNone(summary["forward_transfer"])
        self.assertIsNone(summary["area_forward_transfer"])
        self.assertAlmostEqual(summary["average_performance"], 0.3)

    def test_aggregate_forward_transfer_excludes_first_task(self) -> None:
        result = compute_continual_metrics(
            final_evaluation_successes=[[0.5, 0.5]],
            active_task_success_curves=[[1.0, 1.0], [0.4, 0.6]],
            baseline_success_curves=[[0.0, 0.0], [0.2, 0.2]],
            tail_size=1,
        )
        self.assertAlmostEqual(result.average_raw_forward_transfer, 0.3)
        self.assertAlmostEqual(result.average_forward_transfer, 0.375)

    def test_invalid_tail_size_is_rejected(
        self,
    ) -> None:
        """Tail size must be positive."""
        with self.assertRaises(
            ValueError
        ):
            compute_final_task_performance(
                [
                    [
                        0.5,
                    ],
                ],
                tail_size=0,
            )


if __name__ == "__main__":
    unittest.main()
