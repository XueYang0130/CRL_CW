import sys
import unittest
from unittest.mock import patch

from scripts.run import parse_args


class TestCliPrecedence(unittest.TestCase):
    def test_semantic_methods_default_to_validated_v3_segmenter(self) -> None:
        for method in ("semantic_local_bc", "semantic_hybrid_bc"):
            argv = ["run.py", "--mode", "continual", "--method", method]
            if method == "semantic_hybrid_bc":
                argv.extend(
                    [
                        "--task-specific-segment-manifest",
                        "configs/segment_selection/cw10_v3_task_specific_example.json",
                    ]
                )
            with self.subTest(method=method), patch.object(sys, "argv", argv):
                args = parse_args()
            self.assertEqual(args.semantic_segment_scheme, "task_aware_v3")

    def test_explicit_segment_scheme_overrides_method_preset(self) -> None:
        argv = [
            "run.py",
            "--mode",
            "continual",
            "--method",
            "semantic_local_bc",
            "--semantic-segment-scheme",
            "task_aware_v2",
        ]
        with patch.object(sys, "argv", argv):
            args = parse_args()
        self.assertEqual(args.semantic_segment_scheme, "task_aware_v2")

    def test_explicit_values_override_method_presets(self) -> None:
        argv = [
            "run.py",
            "--mode",
            "continual",
            "--method",
            "semantic_hybrid_bc",
            "--exploration-strategy",
            "random",
            "--background-segment-ratio",
            "0",
            "--task-specific-segment-manifest",
            "configs/segment_selection/cw10_v3_task_specific_example.json",
        ]
        with patch.object(sys, "argv", argv):
            args = parse_args()
        self.assertEqual(args.exploration_strategy, "random")
        self.assertEqual(args.background_segment_ratio, 0.0)


if __name__ == "__main__":
    unittest.main()
