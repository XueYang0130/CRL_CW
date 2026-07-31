import json
import unittest
from unittest.mock import patch

from training.llm_controller import (
    _extract_text_from_response,
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
                '"priority":["manipulation"],"reason":"valid shape"}'
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


if __name__ == "__main__":
    unittest.main()
