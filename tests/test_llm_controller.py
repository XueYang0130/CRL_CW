import json
import unittest
from unittest.mock import patch

from training.llm_controller import (
    _extract_text_from_response,
    build_llm_gate_prompt,
    call_openai_json_controller,
)


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self):
        return json.dumps(self._payload).encode("utf-8")


class LLMControllerTests(unittest.TestCase):
    def test_prompt_loads_protocol_and_official_task_descriptions(self):
        prompt = build_llm_gate_prompt(
            previous_task_name="push-back-v3",
            current_task_name="stick-pull-v3",
            evaluation_index_within_task=2,
            task_step=40_000,
            task_total_steps=500_000,
            recent_success_curve=[0.0, 0.2],
            recent_return_curve=[10.0, 20.0],
            available_segment_counts={"manipulation": 100},
            available_task_specific_counts={"rotate_faucet_closed": 25},
            current_selected_segments=("manipulation",),
            current_segment_weights={"manipulation": 1.0},
        )
        self.assertIn("Grasp a stick and use it to pull a box", prompt)
        self.assertIn("Pull a puck to a goal", prompt)
        self.assertIn("v1_compatible", prompt)
        self.assertIn("sum to exactly 1.0", prompt)
        self.assertIn('"rotate_faucet_closed": 25', prompt)
        self.assertNotIn("shelf-place-v3", prompt)

    def test_extract_text_reports_incomplete_response_without_dumping_payload(self):
        payload = {
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
            "output": [{"type": "reasoning", "encrypted_content": "secret"}],
        }

        with self.assertRaisesRegex(ValueError, "max_output_tokens") as context:
            _extract_text_from_response(payload)

        self.assertNotIn("secret", str(context.exception))

    def test_controller_retries_reasoning_only_response(self):
        incomplete = {
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
            "output": [{"type": "reasoning", "content": []}],
        }
        complete = {
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": (
                                '{"selected_segments":["manipulation"],'
                                '"priority":["manipulation"],'
                                '"weights":{"manipulation":1.0},'
                                '"reason":"current bottleneck"}'
                            ),
                        }
                    ],
                }
            ],
        }
        requests = []

        def fake_urlopen(request, timeout):
            requests.append(json.loads(request.data.decode("utf-8")))
            return _FakeResponse(incomplete if len(requests) == 1 else complete)

        with patch("training.llm_controller.urllib.request.urlopen", side_effect=fake_urlopen):
            decision, raw_text = call_openai_json_controller(
                api_key="test-key",
                model="gpt-5-mini",
                prompt="test prompt",
                max_output_tokens=160,
            )

        self.assertEqual(decision["selected_segments"], ["manipulation"])
        self.assertIn("current bottleneck", raw_text)
        self.assertEqual(len(requests), 2)
        self.assertEqual(requests[0]["reasoning"], {"effort": "minimal"})
        self.assertEqual(requests[0]["text"], {"verbosity": "low"})
        self.assertEqual(requests[0]["max_output_tokens"], 512)

    def test_controller_normalizes_positive_weight_scores(self):
        complete = {
            "status": "completed",
            "output_text": (
                '{"selected_segments":["manipulation","finish_or_stabilize"],'
                '"priority":["manipulation","finish_or_stabilize"],'
                '"weights":{"manipulation":3,"finish_or_stabilize":1},'
                '"reason":"normalize scores"}'
            ),
        }
        with patch(
            "training.llm_controller.urllib.request.urlopen",
            return_value=_FakeResponse(complete),
        ):
            decision, _ = call_openai_json_controller(
                api_key="test-key",
                model="gpt-5-mini",
                prompt="test prompt",
                max_output_tokens=400,
            )
        self.assertEqual(
            decision["weights"],
            {"manipulation": 0.75, "finish_or_stabilize": 0.25},
        )

    def test_controller_retries_json_with_wrong_field_types(self):
        malformed = {
            "status": "completed",
            "output_text": (
                '{"selected_segments":"manipulation",'
                '"priority":1,"reason":"wrong shape"}'
            ),
        }
        complete = {
            "status": "completed",
            "output_text": (
                '{"selected_segments":["manipulation"],'
                '"priority":["manipulation"],'
                '"weights":{"manipulation":1.0},"reason":"valid shape"}'
            ),
        }
        responses = [_FakeResponse(malformed), _FakeResponse(complete)]

        with patch(
            "training.llm_controller.urllib.request.urlopen",
            side_effect=responses,
        ) as urlopen:
            decision, _ = call_openai_json_controller(
                api_key="test-key",
                model="gpt-5-mini",
                prompt="test prompt",
                max_output_tokens=400,
            )

        self.assertEqual(urlopen.call_count, 2)
        self.assertEqual(decision["selected_segments"], ["manipulation"])

    def test_controller_rejects_missing_weights(self):
        invalid = {
            "status": "completed",
            "output_text": (
                '{"selected_segments":["manipulation"],'
                '"priority":["manipulation"],"reason":"missing weights"}'
            ),
        }
        valid = {
            "status": "completed",
            "output_text": (
                '{"selected_segments":["manipulation"],'
                '"priority":["manipulation"],'
                '"weights":{"manipulation":0.5},"reason":"valid weights"}'
            ),
        }
        with patch(
            "training.llm_controller.urllib.request.urlopen",
            side_effect=[_FakeResponse(invalid), _FakeResponse(valid)],
        ):
            decision, _ = call_openai_json_controller(
                api_key="test-key",
                model="gpt-5-mini",
                prompt="test prompt",
                max_output_tokens=400,
            )
        self.assertEqual(decision["weights"], {"manipulation": 1.0})

    def test_controller_retries_duplicate_segments(self):
        duplicate = {
            "status": "completed",
            "output_text": (
                '{"selected_segments":["manipulation","manipulation"],'
                '"priority":["manipulation"],'
                '"weights":{"manipulation":1.0},"reason":"duplicate"}'
            ),
        }
        valid = {
            "status": "completed",
            "output_text": (
                '{"selected_segments":["manipulation"],'
                '"priority":["manipulation"],'
                '"weights":{"manipulation":1.0},"reason":"valid"}'
            ),
        }
        with patch(
            "training.llm_controller.urllib.request.urlopen",
            side_effect=[_FakeResponse(duplicate), _FakeResponse(valid)],
        ) as urlopen:
            decision, _ = call_openai_json_controller(
                api_key="test-key",
                model="gpt-5-mini",
                prompt="test prompt",
                max_output_tokens=400,
            )
        self.assertEqual(urlopen.call_count, 2)
        self.assertEqual(decision["selected_segments"], ["manipulation"])


if __name__ == "__main__":
    unittest.main()
