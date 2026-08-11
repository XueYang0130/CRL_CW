from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from string import Template

@dataclass(frozen=True)
class LLMPolicyPriorDecision:
    source_task_name: str | None
    confidence: float
    reason: str
    source: str
    raw_response_text: str


def no_transfer_decision(reason: str, *, source: str = "fallback") -> LLMPolicyPriorDecision:
    return LLMPolicyPriorDecision(
        source_task_name=None,
        confidence=0.0,
        reason=reason,
        source=source,
        raw_response_text="",
    )


def resolve_prior_exploration(
    *,
    task_index: int,
    decision: LLMPolicyPriorDecision,
) -> tuple[str | None, int | None, str]:
    """Choose exploration only after the policy-prior decision is finalized."""
    if task_index == 0 or decision.source_task_name is None:
        return None, None, "random"
    return "current", task_index + 1, "current_initialized_prior"


def build_policy_prior_prompt(
    *,
    current_task_name: str,
    previous_task_names: list[str],
    prompt_dir: str | Path,
) -> str:
    asset_dir = Path(prompt_dir).expanduser().resolve()
    template = Template((asset_dir / "head_selector.txt").read_text(encoding="utf-8"))
    descriptions = json.loads(
        (asset_dir / "cw10_v3_task_descriptions.json").read_text(encoding="utf-8")
    )
    profiles = descriptions.get("tasks", {})
    names = [current_task_name, *previous_task_names]
    missing = [name for name in names if name not in profiles]
    if missing:
        raise ValueError(f"Missing task descriptions: {missing}.")
    return template.substitute(
        current_task_profile=json.dumps(
            {"task_name": current_task_name, **profiles[current_task_name]},
            indent=2,
            sort_keys=True,
        ),
        previous_task_profiles=json.dumps(
            [{"task_name": name, **profiles[name]} for name in previous_task_names],
            indent=2,
            sort_keys=True,
        ),
        allowed_source_names=json.dumps(previous_task_names),
    )


def call_policy_prior_controller(
    *,
    api_key: str,
    model: str,
    prompt: str,
    max_output_tokens: int,
    previous_task_names: list[str],
    minimum_confidence: float,
) -> LLMPolicyPriorDecision:
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
    raw_text = _response_text(response_payload)
    payload = _parse_json_object(raw_text)
    source_name = payload.get("source_task_name")
    if source_name == "none":
        source_name = None
    if source_name is not None and source_name not in previous_task_names:
        raise ValueError(f"LLM selected unavailable or future task: {source_name!r}.")
    confidence = float(payload.get("confidence"))
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be in [0, 1].")
    reason = payload.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("reason must be a non-empty string.")
    if source_name is not None and confidence < minimum_confidence:
        return LLMPolicyPriorDecision(
            source_task_name=None,
            confidence=confidence,
            reason=f"Rejected low-confidence transfer ({confidence:.3f}): {reason}",
            source="confidence_fallback",
            raw_response_text=raw_text,
        )
    return LLMPolicyPriorDecision(
        source_task_name=source_name,
        confidence=confidence,
        reason=" ".join(reason.split())[:240],
        source="llm",
        raw_response_text=raw_text,
    )


def _response_text(payload: dict[str, object]) -> str:
    if isinstance(payload.get("output_text"), str):
        return str(payload["output_text"]).strip()
    texts: list[str] = []
    output = payload.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            content_items = item.get("content")
            if not isinstance(content_items, list):
                continue
            for content in content_items:
                if not isinstance(content, dict) or content.get("type") != "output_text":
                    continue
                text = content.get("text")
                if isinstance(text, str):
                    texts.append(text)
    if not texts:
        raise ValueError("OpenAI response contained no text output.")
    return "\n".join(texts).strip()


def _parse_json_object(text: str) -> dict[str, object]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            stripped = "\n".join(lines[1:-1]).strip()
    payload = json.loads(stripped)
    if not isinstance(payload, dict):
        raise ValueError("Policy-prior response must be a JSON object.")
    return payload
