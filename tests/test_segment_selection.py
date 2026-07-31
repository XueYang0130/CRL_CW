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


if __name__ == "__main__":
    unittest.main()
