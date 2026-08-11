import json
import unittest
from unittest.mock import patch

from training.llm_bc_schedule_controller import (
    SCHEDULE_DELAYED,
    SCHEDULE_STANDARD,
    build_task_schedule_prompt,
    call_task_schedule_controller,
    standard_fallback_decision,
)


class _FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


class LLMBCScheduleControllerTests(unittest.TestCase):
    def test_prompt_contains_current_and_past_but_not_future_tasks(self):
        prompt = build_task_schedule_prompt(
            current_task_name="stick-pull-v3",
            previous_task_names=[
                "hammer-v3",
                "push-wall-v3",
                "faucet-close-v3",
                "push-back-v3",
            ],
            prompt_dir="prompts/llm_bc_schedule",
        )
        self.assertIn("Grasp a stick and use it to pull a box", prompt)
        self.assertIn("Pull a puck to a goal", prompt)
        self.assertIn("v1-compatible reward observation alignment", prompt)
        self.assertNotIn("shelf-place-v3", prompt)

    def test_accepts_delayed_schedule(self):
        response = {
            "status": "completed",
            "output_text": json.dumps(
                {
                    "schedule": SCHEDULE_DELAYED,
                    "reason_codes": [
                        "tool_mediated",
                        "multi_stage_alignment",
                        "indirect_object_control",
                        "low_prior_skill_compatibility",
                    ],
                    "confidence": 0.9,
                    "reason": "Novel tool-use skill conflicts with direct manipulation.",
                }
            ),
        }
        with patch(
            "training.llm_bc_schedule_controller.urllib.request.urlopen",
            return_value=_FakeResponse(response),
        ):
            decision = call_task_schedule_controller(
                api_key="test-key",
                model="gpt-5-mini",
                prompt="prompt",
                max_output_tokens=400,
            )
        self.assertEqual(decision.schedule, SCHEDULE_DELAYED)
        self.assertEqual(decision.source, "llm")

    def test_rejects_unconstrained_schedule(self):
        response = {
            "status": "completed",
            "output_text": json.dumps(
                {
                    "schedule": "custom_bc_42",
                    "reason_codes": ["tool_mediated"],
                    "confidence": 0.9,
                    "reason": "invalid",
                }
            ),
        }
        with patch(
            "training.llm_bc_schedule_controller.urllib.request.urlopen",
            return_value=_FakeResponse(response),
        ):
            with self.assertRaisesRegex(ValueError, "Unsupported BC schedule"):
                call_task_schedule_controller(
                    api_key="test-key",
                    model="gpt-5-mini",
                    prompt="prompt",
                    max_output_tokens=400,
                )

    def test_rejects_delayed_schedule_without_indirect_control_evidence(self):
        response = {
            "status": "completed",
            "output_text": json.dumps(
                {
                    "schedule": SCHEDULE_DELAYED,
                    "reason_codes": [
                        "tool_mediated",
                        "multi_stage_alignment",
                        "low_prior_skill_compatibility",
                    ],
                    "confidence": 0.9,
                    "reason": "Evidence is incomplete.",
                }
            ),
        }
        with patch(
            "training.llm_bc_schedule_controller.urllib.request.urlopen",
            return_value=_FakeResponse(response),
        ):
            with self.assertRaisesRegex(ValueError, "indirect_object_control"):
                call_task_schedule_controller(
                    api_key="test-key",
                    model="gpt-5-mini",
                    prompt="prompt",
                    max_output_tokens=400,
                )

    def test_fallback_is_standard_bc(self):
        decision = standard_fallback_decision("API unavailable")
        self.assertEqual(decision.schedule, SCHEDULE_STANDARD)
        self.assertEqual(decision.source, "fallback")


if __name__ == "__main__":
    unittest.main()
