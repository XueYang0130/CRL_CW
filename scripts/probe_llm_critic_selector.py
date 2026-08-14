from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from string import Template
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
import sys

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs import get_cw10_tasks
from training.llm_controller import (
    _extract_text_from_response,
    _parse_first_json_object,
    load_dotenv_if_present,
    read_api_key,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--max-output-tokens", type=int, default=1500)
    parser.add_argument("--run-name", default="cw10_v3_semantic_critic_selector")
    return parser.parse_args()


def validate(payload: dict[str, Any], tasks: list[str]) -> dict[str, Any]:
    decisions = payload.get("decisions")
    if not isinstance(decisions, list) or len(decisions) != len(tasks) - 1:
        raise ValueError(f"Expected {len(tasks) - 1} decisions, got {decisions!r}.")
    normalized: list[dict[str, Any]] = []
    for expected_index, item in enumerate(decisions, start=1):
        if not isinstance(item, dict):
            raise ValueError("Each decision must be an object.")
        if item.get("task_index") != expected_index:
            raise ValueError(f"Expected task_index {expected_index}, got {item.get('task_index')!r}.")
        if item.get("task_name") != tasks[expected_index]:
            raise ValueError(f"Unexpected task_name at index {expected_index}: {item.get('task_name')!r}.")
        decision = item.get("decision")
        if decision not in {"transfer", "reset"}:
            raise ValueError(f"Invalid decision: {decision!r}.")
        confidence = float(item.get("confidence"))
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1].")
        shift = item.get("value_structure_shift")
        if shift not in {"low", "medium", "high"}:
            raise ValueError(f"Invalid value_structure_shift: {shift!r}.")
        reason = item.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("reason must be a non-empty string.")
        normalized.append(
            {
                "task_index": expected_index,
                "task_name": tasks[expected_index],
                "decision": decision,
                "confidence": confidence,
                "value_structure_shift": shift,
                "reason": " ".join(reason.split()),
            }
        )
    rationale = payload.get("global_rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("global_rationale must be a non-empty string.")
    return {"decisions": normalized, "global_rationale": " ".join(rationale.split())}


def main() -> None:
    args = parse_args()
    load_dotenv_if_present(PROJECT_ROOT / ".env")
    model = args.model or os.environ.get("OPENAI_MODEL", "gpt-5-mini").strip()
    api_key = read_api_key(args.api_key_env)
    tasks = get_cw10_tasks("v3")
    asset_dir = PROJECT_ROOT / "prompts" / "llm_controller"
    descriptions = json.loads((asset_dir / "cw10_v3_task_descriptions.json").read_text(encoding="utf-8"))
    protocol = json.loads((asset_dir / "protocol.json").read_text(encoding="utf-8"))
    profiles = [
        {"task_index": index, "task_name": task, **descriptions["tasks"][task]}
        for index, task in enumerate(tasks)
    ]
    template = Template(
        (PROJECT_ROOT / "prompts" / "llm_critic_selector" / "task_boundary_selector.txt").read_text(encoding="utf-8")
    )
    prompt = template.substitute(
        protocol=json.dumps(protocol, indent=2, sort_keys=True),
        task_profiles=json.dumps(profiles, indent=2, sort_keys=True),
    )

    # Reuse the repository's Responses API client. Its segment-specific validator is
    # bypassed here because this probe has a different, strictly validated schema.
    import urllib.error
    import urllib.request

    body = {
        "model": model,
        "input": prompt,
        "max_output_tokens": args.max_output_tokens,
        "reasoning": {"effort": "minimal"},
        "text": {"verbosity": "low"},
    }
    request = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        details = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI API HTTP error {error.code}: {details}") from error
    raw_text = _extract_text_from_response(response_payload)
    parsed = _parse_first_json_object(raw_text)
    result = validate(parsed, tasks)
    result["model"] = model

    output_dir = PROJECT_ROOT / "outputs" / "llm_critic_selector" / args.run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    (output_dir / "raw_response.txt").write_text(raw_text + "\n", encoding="utf-8")
    (output_dir / "decision.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(output_dir), **result}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
