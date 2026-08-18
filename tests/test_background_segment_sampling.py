import unittest

import numpy as np

from training.continual_experiment_copy import build_hybrid_memory_from_store
from training.semantic_segments import SEGMENT_ORDER


def sample_background_count(selected_count: int, background_count: int, ratio: float) -> int:
    kept = 0
    if selected_count and background_count and ratio > 0.0:
        kept = min(int(round(selected_count * ratio)), background_count)
    return kept


class TestBackgroundSegmentSampling(unittest.TestCase):
    def test_hybrid_memory_rejects_nonfinite_ratios(self) -> None:
        store = {segment: [] for segment in SEGMENT_ORDER}
        for field, value in (
            ("background_ratio", float("nan")),
            ("background_ratio", float("inf")),
            ("task_specific_ratio", float("nan")),
            ("task_specific_ratio", float("inf")),
        ):
            kwargs = {
                "payload_store": store,
                "selected_segments": ("approach",),
                "segment_weights": None,
                "background_ratio": 0.0,
                "task_specific_ratio": 0.0,
            }
            kwargs[field] = value
            with self.subTest(field=field, value=value), self.assertRaisesRegex(
                ValueError, "finite and non-negative"
            ):
                build_hybrid_memory_from_store(**kwargs)

    def test_zero_ratio_keeps_no_background(self) -> None:
        self.assertEqual(sample_background_count(100, 50, 0.0), 0)

    def test_positive_ratio_scales_with_selected_count(self) -> None:
        self.assertEqual(sample_background_count(100, 50, 0.1), 10)
        self.assertEqual(sample_background_count(100, 50, 0.2), 20)

    def test_background_count_is_capped(self) -> None:
        self.assertEqual(sample_background_count(100, 5, 0.2), 5)

    def test_no_selected_states_means_no_background(self) -> None:
        self.assertEqual(sample_background_count(0, 50, 0.2), 0)

    def test_hybrid_memory_applies_explicit_weights_and_both_side_channels(self) -> None:
        def payload(segment_index: int) -> dict[str, np.ndarray]:
            observations = np.full((10, 2), segment_index, dtype=np.float32)
            targets = np.full((10, 1), segment_index, dtype=np.float32)
            return {
                "observations": observations,
                "target_means": targets,
                "target_log_stds": targets,
            }

        store = {
            "approach": [payload(0)],
            "contact_or_alignment": [payload(1)],
            "manipulation": [payload(2)],
            "finish_or_stabilize": [payload(3)],
        }
        observations, _, _, background_count, task_specific_count = (
            build_hybrid_memory_from_store(
                payload_store=store,
                selected_segments=("contact_or_alignment", "manipulation"),
                segment_weights={"contact_or_alignment": 1.0, "manipulation": 2.0},
                background_ratio=0.2,
                task_specific_segments=("finish_or_stabilize",),
                task_specific_ratio=0.2,
            )
        )
        self.assertEqual(background_count, 4)
        self.assertEqual(task_specific_count, 4)
        self.assertEqual(observations.shape[0], 28)
        self.assertEqual(int(np.sum(observations[:, 0] == 2)), 13)
        self.assertEqual(int(np.sum(observations[:, 0] == 1)), 7)

    def test_task_specific_side_channel_uses_fine_grained_store(self) -> None:
        def payload(value: int, count: int = 10) -> dict[str, np.ndarray]:
            observations = np.full((count, 2), value, dtype=np.float32)
            targets = np.full((count, 1), value, dtype=np.float32)
            return {
                "observations": observations,
                "target_means": targets,
                "target_log_stds": targets,
            }

        general_store = {
            "approach": [],
            "contact_or_alignment": [payload(1)],
            "manipulation": [payload(2)],
            "finish_or_stabilize": [],
        }
        fine_grained_store = {
            "rotate_faucet_closed": [payload(9)],
            "complete_and_stabilize": [payload(8)],
        }
        observations, _, _, background_count, task_specific_count = (
            build_hybrid_memory_from_store(
                payload_store=general_store,
                selected_segments=("contact_or_alignment", "manipulation"),
                segment_weights={"contact_or_alignment": 0.5, "manipulation": 0.5},
                background_ratio=0.0,
                task_specific_ratio=0.2,
                task_specific_payload_store=fine_grained_store,
            )
        )
        self.assertEqual(background_count, 0)
        self.assertEqual(task_specific_count, 4)
        self.assertEqual(observations.shape[0], 24)
        self.assertEqual(int(np.sum(observations[:, 0] >= 8)), 4)

    def test_equal_weights_do_not_duplicate_the_first_segment(self) -> None:
        def payload(value: int) -> dict[str, np.ndarray]:
            observations = np.full((10, 2), value, dtype=np.float32)
            targets = np.full((10, 1), value, dtype=np.float32)
            return {
                "observations": observations,
                "target_means": targets,
                "target_log_stds": targets,
            }

        store = {
            "approach": [],
            "contact_or_alignment": [payload(1)],
            "manipulation": [payload(2)],
            "finish_or_stabilize": [payload(3)],
        }
        observations, _, _, _, _ = build_hybrid_memory_from_store(
            payload_store=store,
            selected_segments=("contact_or_alignment", "manipulation"),
            segment_weights={"contact_or_alignment": 1.0, "manipulation": 1.0},
            background_ratio=0.0,
        )
        self.assertEqual(observations.shape[0], 20)
        self.assertEqual(int(np.sum(observations[:, 0] == 1)), 10)
        self.assertEqual(int(np.sum(observations[:, 0] == 2)), 10)

    def test_small_finish_anchor_keeps_requested_fraction(self) -> None:
        def payload(value: int) -> dict[str, np.ndarray]:
            observations = np.full((10, 2), value, dtype=np.float32)
            targets = np.full((10, 1), value, dtype=np.float32)
            return {
                "observations": observations,
                "target_means": targets,
                "target_log_stds": targets,
            }

        store = {
            "approach": [],
            "contact_or_alignment": [payload(1)],
            "manipulation": [payload(2)],
            "finish_or_stabilize": [payload(3)],
        }
        observations, _, _, _, _ = build_hybrid_memory_from_store(
            payload_store=store,
            selected_segments=(
                "contact_or_alignment",
                "manipulation",
                "finish_or_stabilize",
            ),
            segment_weights={
                "contact_or_alignment": 0.4,
                "manipulation": 0.4,
                "finish_or_stabilize": 0.2,
            },
            background_ratio=0.0,
        )
        self.assertEqual(observations.shape[0], 30)
        self.assertEqual(int(np.sum(observations[:, 0] == 1)), 12)
        self.assertEqual(int(np.sum(observations[:, 0] == 2)), 12)
        self.assertEqual(int(np.sum(observations[:, 0] == 3)), 6)


if __name__ == "__main__":
    unittest.main()
