import json
import tempfile
import unittest
from pathlib import Path

from training.segment_selection import (
    SegmentSelection,
    load_segment_selection_manifest,
    select_task_specific_segments_for_task,
    select_segments_for_task_pair,
)


class TestSegmentSelection(unittest.TestCase):
    def test_fallback_selection_without_manifest(self) -> None:
        result = select_segments_for_task_pair(
            previous_task_name="hammer-v3",
            new_task_name="push-wall-v3",
            manifest=None,
            fallback_segments=["contact_or_alignment", "manipulation"],
        )
        self.assertEqual(
            result,
            SegmentSelection(
                selected_segments=("contact_or_alignment", "manipulation"),
                priority=("contact_or_alignment", "manipulation"),
                weights=(("contact_or_alignment", 0.5), ("manipulation", 0.5)),
                reason="No segment selection manifest provided.",
                selection_source="default",
            ),
        )

    def test_pair_specific_selection_from_manifest(self) -> None:
        manifest = {
            "default_segments": ["contact_or_alignment", "manipulation"],
            "pairs": {
                "push-back-v3->stick-pull-v3": {
                    "selected_segments": [
                        "approach",
                        "contact_or_alignment",
                        "manipulation",
                    ],
                    "priority": [
                        "approach",
                        "contact_or_alignment",
                        "manipulation",
                    ],
                    "weights": {
                        "approach": 0.5,
                        "contact_or_alignment": 1.0,
                        "manipulation": 1.0,
                    },
                    "reason": "Multi-stage tool use requires richer preservation.",
                }
            },
        }
        result = select_segments_for_task_pair(
            previous_task_name="push-back-v3",
            new_task_name="stick-pull-v3",
            manifest=manifest,
            fallback_segments=["contact_or_alignment", "manipulation"],
        )
        self.assertEqual(
            result.selected_segments,
            ("approach", "contact_or_alignment", "manipulation"),
        )
        self.assertEqual(result.selection_source, "manifest")
        self.assertEqual(dict(result.weights)["approach"], 0.2)
        self.assertAlmostEqual(sum(dict(result.weights).values()), 1.0)

    def test_manifest_loader(self) -> None:
        payload = {"default_segments": ["contact_or_alignment", "manipulation"], "pairs": {}}
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "manifest.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            loaded = load_segment_selection_manifest(path)
        self.assertEqual(loaded, payload)

    def test_task_specific_selection_from_manifest(self) -> None:
        manifest = {
            "tasks": {
                "stick-pull-v3": {
                    "selected_segments": ["finish_or_stabilize"],
                    "priority": ["finish_or_stabilize"],
                    "reason": "Protect late-stage tool use.",
                }
            }
        }
        result = select_task_specific_segments_for_task(
            task_name="stick-pull-v3",
            manifest=manifest,
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.selected_segments, ("finish_or_stabilize",))
        self.assertEqual(result.selection_source, "task_specific_manifest")

    def test_rejects_duplicate_segments_and_unselected_priority(self) -> None:
        duplicate_manifest = {
            "default_segments": ["manipulation", "manipulation"],
        }
        with self.assertRaisesRegex(ValueError, "duplicate"):
            select_segments_for_task_pair(
                previous_task_name=None,
                new_task_name="hammer-v3",
                manifest=duplicate_manifest,
                fallback_segments=["manipulation"],
            )

        invalid_priority_manifest = {
            "default_segments": ["manipulation"],
            "pairs": {
                "hammer-v3->push-wall-v3": {
                    "selected_segments": ["manipulation"],
                    "priority": ["approach"],
                }
            },
        }
        with self.assertRaisesRegex(ValueError, "unselected"):
            select_segments_for_task_pair(
                previous_task_name="hammer-v3",
                new_task_name="push-wall-v3",
                manifest=invalid_priority_manifest,
                fallback_segments=["manipulation"],
            )


if __name__ == "__main__":
    unittest.main()
