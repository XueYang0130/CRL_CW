from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Any


SCHEDULE_STANDARD = "standard_bc"
SCHEDULE_DELAYED = "delayed_progress_bc"
ALLOWED_SCHEDULES = frozenset({SCHEDULE_STANDARD, SCHEDULE_DELAYED})
ALLOWED_REASON_CODES = frozenset(
    {
        "direct_manipulation",
        "high_prior_skill_compatibility",
        "tool_mediated",
        "multi_stage_alignment",
        "indirect_object_control",
        "low_prior_skill_compatibility",
    }
)
DELAYED_REQUIRED_REASON_CODES = frozenset(
    {
        "tool_mediated",
        "indirect_object_control",
        "low_prior_skill_compatibility",
    }
)
DELAYED_MIN_CONFIDENCE = 0.7


@dataclass(frozen=True)
class LLMBCScheduleDecision:
    schedule: str
    reason_codes: tuple[str, ...]
    confidence: float
    reason: str
    source: str
    raw_response_text: str


def standard_fallback_decision(reason: str) -> LLMBCScheduleDecision:
    return LLMBCScheduleDecision(
        schedule=SCHEDULE_STANDARD,
        reason_codes=("high_prior_skill_compatibility",),
        confidence=0.0,
        reason=reason,
        source="fallback",
        raw_response_text="",
    )


def build_task_schedule_prompt(
    *,
    current_task_name: str,
    previous_task_names: list[str],
    prompt_dir: str | Path,
) -> str:
    asset_dir = Path(prompt_dir).expanduser().resolve()
    template = Template(
        (asset_dir / "task_bc_schedule.txt").read_text(encoding="utf-8")
    )
    descriptions = json.loads(
        (asset_dir / "cw10_v3_task_descriptions.json").read_text(encoding="utf-8")
    )
    profiles = descriptions.get("tasks", {})
    if current_task_name not in profiles:
        raise ValueError(f"Missing task description for {current_task_name}.")
    unknown_previous = [name for name in previous_task_names if name not in profiles]
    if unknown_previous:
        raise ValueError(f"Missing previous task descriptions: {unknown_previous}.")
    current_profile = {"task_name": current_task_name, **profiles[current_task_name]}
    previous_profiles = [
        {"task_name": name, **profiles[name]} for name in previous_task_names
    ]
    return template.substitute(
        current_task_profile=json.dumps(current_profile, indent=2, sort_keys=True),
        previous_task_profiles=json.dumps(previous_profiles, indent=2, sort_keys=True),
    )


def _response_text(payload: dict[str, Any]) -> str:
    if isinstance(payload.get("output_text"), str):
        return payload["output_text"].strip()
    texts: list[str] = []
    for item in payload.get("output", []):
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []):
            if not isinstance(content, dict) or content.get("type") != "output_text":
                continue
            value = content.get("text")
            if isinstance(value, str):
                texts.append(value)
            elif isinstance(value, dict) and isinstance(value.get("value"), str):
                texts.append(value["value"])
    if not texts:
        raise ValueError(
            "OpenAI response contained no text output "
            f"(status={payload.get('status', 'unknown')!r})."
        )
    return "\n".join(texts).strip()


def _parse_decision(text: str) -> LLMBCScheduleDecision:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            stripped = "\n".join(lines[1:-1]).strip()
    payload = json.loads(stripped)
    if not isinstance(payload, dict):
        raise ValueError("Schedule response must be a JSON object.")
    schedule = payload.get("schedule")
    if schedule not in ALLOWED_SCHEDULES:
        raise ValueError(f"Unsupported BC schedule: {schedule!r}.")
    reason_codes = payload.get("reason_codes")
    if (
        not isinstance(reason_codes, list)
        or not reason_codes
        or not all(isinstance(code, str) for code in reason_codes)
        or len(set(reason_codes)) != len(reason_codes)
    ):
        raise ValueError("reason_codes must be a non-empty list of unique strings.")
    invalid_codes = sorted(set(reason_codes).difference(ALLOWED_REASON_CODES))
    if invalid_codes:
        raise ValueError(f"Unsupported reason codes: {invalid_codes}.")
    confidence = float(payload.get("confidence"))
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be in [0, 1].")
    if schedule == SCHEDULE_DELAYED:
        missing_codes = sorted(
            DELAYED_REQUIRED_REASON_CODES.difference(reason_codes)
        )
        if missing_codes:
            raise ValueError(
                "delayed_progress_bc is missing required semantic evidence: "
                f"{missing_codes}."
            )
        if confidence < DELAYED_MIN_CONFIDENCE:
            raise ValueError(
                "delayed_progress_bc confidence must be at least "
                f"{DELAYED_MIN_CONFIDENCE}."
            )
    reason = payload.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("reason must be a non-empty string.")
    return LLMBCScheduleDecision(
        schedule=schedule,
        reason_codes=tuple(reason_codes),
        confidence=confidence,
        reason=" ".join(reason.split())[:240],
        source="llm",
        raw_response_text=text,
    )


def call_task_schedule_controller(
    *,
    api_key: str,
    model: str,
    prompt: str,
    max_output_tokens: int,
) -> LLMBCScheduleDecision:
    body = {
        "model": model,
        "input": prompt,
        "max_output_tokens": min(max(max_output_tokens, 512), 1024),
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
            response_payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        details = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI API HTTP error {error.code}: {details}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"OpenAI API connection error: {error}") from error
    text = _response_text(response_payload)
    return _parse_decision(text)
