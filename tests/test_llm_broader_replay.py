from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from training.llm_broader_replay_controller import (
    build_broader_replay_prompt,
    call_broader_replay_controller,
    random_fallback_decision,
)


class LLMBroaderReplayTests(unittest.TestCase):
    def test_prompt_contains_protocol_without_future_tasks(self) -> None:
        prompt = build_broader_replay_prompt(
            task_name="stick-pull-v3",
            available_counts={"early": 4, "middle": 5, "late": 6},
            prompt_dir=Path("prompts/llm_broader_replay"),
        )
        self.assertIn("stick-pull-v3", prompt)
        self.assertIn("v3 observation realignment", prompt)
        self.assertIn('"late": 6', prompt)
        self.assertNotIn("future_task", prompt)

    @patch("training.llm_broader_replay_controller.call_openai_json_controller")
    def test_valid_controller_selection(self, controller) -> None:
        controller.return_value = (
            {
                "selected_segments": ["middle", "late"],
                "priority": ["middle", "late"],
                "weights": {"middle": 0.5, "late": 0.5},
                "reason": "recovery boundary",
            },
            "raw",
        )
        decision = call_broader_replay_controller(
            api_key="test",
            model="test-model",
            prompt="test prompt",
            max_output_tokens=100,
        )
        self.assertEqual(decision.selected_bins, ("middle", "late"))
        self.assertEqual(decision.source, "llm")

    @patch("training.llm_broader_replay_controller.call_openai_json_controller")
    def test_invalid_controller_selection_is_rejected(self, controller) -> None:
        controller.return_value = (
            {
                "selected_segments": ["future"],
                "priority": ["future"],
                "weights": {"future": 1.0},
                "reason": "invalid",
            },
            "raw",
        )
        with self.assertRaisesRegex(ValueError, "invalid broader bins"):
            call_broader_replay_controller(
                api_key="test",
                model="test-model",
                prompt="test prompt",
                max_output_tokens=100,
            )

    def test_fallback_selects_all_bins(self) -> None:
        decision = random_fallback_decision("offline")
        self.assertEqual(decision.selected_bins, ("early", "middle", "late"))
        self.assertEqual(decision.source, "random_fallback")


if __name__ == "__main__":
    unittest.main()
