import unittest

import numpy as np

from training.continual_experiment import build_hybrid_memory_from_store


def sample_background_count(selected_count: int, background_count: int, ratio: float) -> int:
    kept = 0
    if selected_count and background_count and ratio > 0.0:
        kept = min(int(round(selected_count * ratio)), background_count)
    return kept


class TestBackgroundSegmentSampling(unittest.TestCase):
    def test_zero_ratio_keeps_no_background(self) -> None:
        self.assertEqual(sample_background_count(100, 50, 0.0), 0)

    def test_positive_ratio_scales_with_selected_count(self) -> None:
        self.assertEqual(sample_background_count(100, 50, 0.1), 10)
        self.assertEqual(sample_background_count(100, 50, 0.2), 20)

    def test_background_count_is_capped(self) -> None:
        self.assertEqual(sample_background_count(100, 5, 0.2), 5)

    def test_no_selected_states_means_no_background(self) -> None:
        self.assertEqual(sample_background_count(0, 50, 0.2), 0)

    def test_hybrid_memory_applies_priority_and_both_side_channels(self) -> None:
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
                priority=("manipulation", "contact_or_alignment"),
                background_ratio=0.2,
                task_specific_segments=("finish_or_stabilize",),
                task_specific_ratio=0.2,
            )
        )
        self.assertEqual(background_count, 6)
        self.assertEqual(task_specific_count, 6)
        self.assertEqual(observations.shape[0], 42)
        self.assertEqual(int(np.sum(observations[:, 0] == 2)), 20)
        self.assertEqual(int(np.sum(observations[:, 0] == 1)), 10)


if __name__ == "__main__":
    unittest.main()
