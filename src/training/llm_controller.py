from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LLMGateDecision:
    selected_segments: tuple[str, ...]
    priority: tuple[str, ...]
    reason: str
    raw_response_text: str
    model: str


def _extract_text_from_response(payload: dict[str, Any]) -> str:
    output = payload.get("output", [])
    texts: list[str] = []
    for item in output:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "message":
            for content in item.get("content", []):
                if not isinstance(content, dict):
                    continue
                if content.get("type") == "output_text":
                    text_value = content.get("text")
                    if isinstance(text_value, str):
                        texts.append(text_value)
                    elif isinstance(text_value, dict) and isinstance(text_value.get("value"), str):
                        texts.append(text_value["value"])
        for content in item.get("content", []):
            if not isinstance(content, dict):
                continue
            if content.get("type") == "output_text" and isinstance(content.get("text"), str):
                texts.append(content["text"])
            elif content.get("type") == "output_text":
                text_value = content.get("text")
                if isinstance(text_value, dict) and isinstance(text_value.get("value"), str):
                    texts.append(text_value["value"])
    if texts:
        return "\n".join(texts).strip()
    if isinstance(payload.get("output_text"), str):
        return payload["output_text"].strip()
    status = payload.get("status", "unknown")
    incomplete_reason = payload.get("incomplete_details")
    raise ValueError(
        "OpenAI response contained no text output "
        f"(status={status!r}, incomplete_details={incomplete_reason!r})."
    )


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            return "\n".join(lines[1:-1]).strip()
    return stripped


def _parse_first_json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    stripped = text.lstrip()
    for index, char in enumerate(stripped):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(stripped[index:])
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict):
            raise ValueError("Controller output must decode to a JSON object.")
        return parsed
    raise ValueError(f"Could not parse a JSON object from controller output: {text[:4000]}")


def _truncate_reason_field(text: str) -> str:
    marker = '"reason"'
    marker_index = text.find(marker)
    if marker_index < 0:
        return text
    colon_index = text.find(":", marker_index + len(marker))
    if colon_index < 0:
        return text
    first_quote_index = text.find('"', colon_index + 1)
    if first_quote_index < 0:
        return text
    cursor = first_quote_index + 1
    while cursor < len(text):
        char = text[cursor]
        if char == '"' and text[cursor - 1] != "\\":
            return text
        if char in "\r\n":
            return text[:cursor] + '"}'
        cursor += 1
    return text


def call_openai_json_controller(
    *,
    api_key: str,
    model: str,
    prompt: str,
    max_output_tokens: int,
) -> tuple[dict[str, Any], str]:
    # Reasoning models can consume part of this budget before emitting JSON.
    # Keep enough headroom even though the requested response is intentionally tiny.
    effective_max_output_tokens = min(max(max_output_tokens, 512), 1024)
    prompts = [
        prompt,
        (
            prompt
            + "\n\nIMPORTANT: Return exactly one minified JSON object on a single line. "
            + 'Use at most 2 selected_segments. Keep "reason" to 2-6 words only. '
            + "No markdown. No repetition. No line breaks inside strings."
        ),
    ]
    last_error: Exception | None = None
    last_text = ""
    for attempt_prompt in prompts:
        body = {
            "model": model,
            "input": attempt_prompt,
            "max_output_tokens": effective_max_output_tokens,
            "reasoning": {"effort": "minimal"},
            "text": {"verbosity": "low"},
        }
        request = urllib.request.Request(
            "https://api.openai.com/v1/responses",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            details = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenAI API HTTP error {error.code}: {details}") from error
        except urllib.error.URLError as error:
            raise RuntimeError(f"OpenAI API connection error: {error}") from error

        try:
            text = _extract_text_from_response(payload)
        except ValueError as error:
            last_error = error
            continue
        last_text = text
        json_text = _truncate_reason_field(_strip_code_fence(text))
        try:
            parsed = json.loads(json_text)
        except json.JSONDecodeError:
            try:
                parsed = _parse_first_json_object(json_text)
            except Exception as error:
                last_error = error
                continue
        if not isinstance(parsed, dict):
            last_error = ValueError("Controller output must be a JSON object.")
            continue
        selected_segments = parsed.get("selected_segments")
        if (
            not isinstance(selected_segments, list)
            or not selected_segments
            or not all(isinstance(segment, str) for segment in selected_segments)
        ):
            last_error = ValueError(
                "Controller selected_segments must be a non-empty array of strings."
            )
            continue
        parsed["selected_segments"] = selected_segments[:2]
        priority = parsed.get("priority")
        if not isinstance(priority, list) or not all(
            isinstance(segment, str) for segment in priority
        ):
            last_error = ValueError("Controller priority must be an array of strings.")
            continue
        allowed = set(parsed["selected_segments"])
        parsed["priority"] = [segment for segment in priority if segment in allowed][:2]
        reason = parsed.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            parsed["reason"] = "controller fallback"
        else:
            parsed["reason"] = " ".join(reason.split())[:64]
        return parsed, text

    raise RuntimeError(
        "Could not parse controller JSON response after retry. "
        f"Last response text: {last_text[:4000]}. "
        f"Last error: {last_error}"
    )


def build_llm_gate_prompt(
    *,
    previous_task_name: str | None,
    current_task_name: str,
    evaluation_index_within_task: int,
    task_step: int,
    task_total_steps: int,
    recent_success_curve: list[float],
    recent_return_curve: list[float],
    available_segment_counts: dict[str, int],
) -> str:
    previous_task_line = previous_task_name or "None (first task)"
    return f"""You are an online semantic replay controller for continual reinforcement learning.

Current transition:
- previous_task: {previous_task_line}
- current_task: {current_task_name}

Current training state:
- evaluation_index_within_task: {evaluation_index_within_task}
- task_step: {task_step}
- task_total_steps: {task_total_steps}
- recent_success_curve: {json.dumps(recent_success_curve)}
- recent_return_curve: {json.dumps(recent_return_curve)}

Available replay segments and counts:
{json.dumps(available_segment_counts, indent=2, sort_keys=True)}

Your job:
- Choose which semantic replay segments should be emphasized for the next training stage.
- Prefer a small, high-value subset when the current task is bottlenecked.
- If the current task still seems to need broader support, include more segments.

Output JSON only with keys:
- "selected_segments": array of segment names
- "priority": ordered array of segment names
- "reason": short string

Constraints:
- Every segment must be one of the available segment names.
- "priority" must contain only segments from "selected_segments".
- Select at most 2 segments.
- Do not output any prose outside JSON.
- Return exactly one minified JSON object on a single line.
- Keep the reason to 2-6 words only.
"""


def read_api_key(env_var_name: str) -> str:
    api_key = os.environ.get(env_var_name, "").strip()
    if not api_key:
        load_dotenv_if_present(Path.cwd() / ".env")
        api_key = os.environ.get(env_var_name, "").strip()
    if not api_key:
        raise RuntimeError(
            f"Missing API key. Please set environment variable {env_var_name}."
        )
    return api_key


def load_dotenv_if_present(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def append_controller_log(path: Path, row: dict[str, Any]) -> None:
    existing: list[dict[str, Any]]
    if path.exists():
        with path.open("r", encoding="utf-8") as file:
            existing = json.load(file)
        if not isinstance(existing, list):
            raise ValueError("controller_decisions.json must contain a JSON array.")
    else:
        existing = []
    existing.append(row)
    path.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
