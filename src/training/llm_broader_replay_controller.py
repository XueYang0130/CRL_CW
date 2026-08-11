from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from string import Template

from training.llm_controller import call_openai_json_controller
from training.success_replay import BROADER_TEMPORAL_BINS


@dataclass(frozen=True)
class LLMBroaderReplayDecision:
    selected_bins: tuple[str, ...]
    reason: str
    source: str
    raw_response_text: str


def random_fallback_decision(reason: str) -> LLMBroaderReplayDecision:
    return LLMBroaderReplayDecision(
        selected_bins=BROADER_TEMPORAL_BINS,
        reason=reason,
        source="random_fallback",
        raw_response_text="",
    )


def random_baseline_decision(reason: str) -> LLMBroaderReplayDecision:
    return LLMBroaderReplayDecision(
        selected_bins=BROADER_TEMPORAL_BINS,
        reason=reason,
        source="random",
        raw_response_text="",
    )


def build_broader_replay_prompt(
    *,
    task_name: str,
    available_counts: dict[str, int],
    prompt_dir: str | Path,
) -> str:
    asset_dir = Path(prompt_dir).expanduser().resolve()
    template = Template((asset_dir / "select_broader_replay.txt").read_text(encoding="utf-8"))
    descriptions_path = asset_dir / "cw10_v3_task_descriptions.json"
    if not descriptions_path.exists():
        descriptions_path = asset_dir.parent / "llm_controller" / "cw10_v3_task_descriptions.json"
    descriptions = json.loads(descriptions_path.read_text(encoding="utf-8"))
    profile = descriptions.get("tasks", {}).get(task_name)
    if not isinstance(profile, dict):
        raise ValueError(f"Missing broader-replay task description for {task_name}.")
    return template.substitute(
        task_profile=json.dumps(
            {"task_name": task_name, **profile}, indent=2, sort_keys=True
        ),
        available_counts=json.dumps(available_counts, indent=2, sort_keys=True),
        allowed_bins=json.dumps(list(BROADER_TEMPORAL_BINS)),
    )


def call_broader_replay_controller(
    *,
    api_key: str,
    model: str,
    prompt: str,
    max_output_tokens: int,
) -> LLMBroaderReplayDecision:
    payload, raw_text = call_openai_json_controller(
        api_key=api_key,
        model=model,
        prompt=prompt,
        max_output_tokens=max_output_tokens,
    )
    selected = tuple(payload["selected_segments"])
    invalid = sorted(set(selected).difference(BROADER_TEMPORAL_BINS))
    if invalid:
        raise ValueError(f"LLM selected invalid broader bins: {invalid}.")
    if not selected:
        raise ValueError("LLM must select at least one broader temporal bin.")
    return LLMBroaderReplayDecision(
        selected_bins=selected,
        reason=str(payload.get("reason", "")),
        source="llm",
        raw_response_text=raw_text,
    )
