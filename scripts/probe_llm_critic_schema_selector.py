from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from string import Template
import sys
from typing import Any
import urllib.error
import urllib.request

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
for path in (SRC_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from envs import get_cw10_tasks
from training.llm_controller import (
    _extract_text_from_response,
    _parse_first_json_object,
    load_dotenv_if_present,
    read_api_key,
)

CONTROLLED_ENTITIES = {"object", "articulated_part", "tool", "tool_mediated_object"}
CONTACT_RELATIONS = {"direct", "grasped", "tool_mediated"}
MOTION_CONSTRAINTS = {"free_3d", "planar", "linear_axis", "rotational_axis"}
PRIMITIVES = {
    "reach", "contact", "grasp", "lift", "align", "push", "pull", "rotate",
    "press", "slide", "place", "release", "stabilize",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--max-output-tokens", type=int, default=3500)
    parser.add_argument("--run-name", default="cw10_v3_schema_critic_selector")
    parser.add_argument("--schema-input", default=None)
    return parser.parse_args()


def _unique_strings(value: Any, field: str) -> list[str]:
    if not isinstance(value, list) or not value or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{field} must be a non-empty string array.")
    if len(value) != len(set(value)):
        raise ValueError(f"{field} must not contain duplicates.")
    return value


def validate(payload: dict[str, Any], tasks: list[str]) -> list[dict[str, Any]]:
    schemas = payload.get("schemas")
    if not isinstance(schemas, list) or len(schemas) != len(tasks):
        raise ValueError(f"Expected {len(tasks)} schemas.")
    normalized: list[dict[str, Any]] = []
    expected_fields = {
        "task_index", "task_name", "controlled_entity", "contact_relation",
        "motion_constraint", "action_primitives", "stage_dependencies", "structural_relations",
    }
    for index, schema in enumerate(schemas):
        if not isinstance(schema, dict) or set(schema) != expected_fields:
            raise ValueError(f"Schema {index} fields do not match the fixed ontology.")
        if schema["task_index"] != index or schema["task_name"] != tasks[index]:
            raise ValueError(f"Schema order mismatch at task {index}.")
        if schema["controlled_entity"] not in CONTROLLED_ENTITIES:
            raise ValueError(f"Invalid controlled_entity at task {index}.")
        if schema["contact_relation"] not in CONTACT_RELATIONS:
            raise ValueError(f"Invalid contact_relation at task {index}.")
        if schema["motion_constraint"] not in MOTION_CONSTRAINTS:
            raise ValueError(f"Invalid motion_constraint at task {index}.")
        primitives = _unique_strings(schema["action_primitives"], "action_primitives")
        if not set(primitives).issubset(PRIMITIVES):
            raise ValueError(f"Unknown action primitive at task {index}.")
        dependencies = schema["stage_dependencies"]
        if not isinstance(dependencies, list):
            raise ValueError("stage_dependencies must be an array.")
        normalized_edges: list[list[str]] = []
        for edge in dependencies:
            if not isinstance(edge, list) or len(edge) != 2 or not all(isinstance(x, str) for x in edge):
                raise ValueError(f"Invalid dependency edge at task {index}.")
            if edge[0] not in primitives or edge[1] not in primitives:
                raise ValueError(f"Dependency edge uses absent primitive at task {index}.")
            normalized_edges.append(edge)
        relations = _unique_strings(schema["structural_relations"], "structural_relations")
        normalized.append({**schema, "action_primitives": primitives,
                           "stage_dependencies": normalized_edges,
                           "structural_relations": relations})
    return normalized


def select(schemas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Parameter-free selector using compositional coverage.

    Direct and grasped manipulation reuse the persistent critic because their reusable
    primitives and relations can compose across earlier tasks. Tool-mediated control is
    treated as a distinct value topology: it transfers only when a prior tool-mediated
    task has the same terminal effect and motion constraint.
    """
    terminal_primitives = {"push", "pull", "rotate", "press", "slide", "place"}
    decisions: list[dict[str, Any]] = []
    for index, current in enumerate(schemas):
        if index == 0:
            decisions.append({"task_index": 0, "task_name": current["task_name"],
                              "decision": "fresh", "covering_task": None,
                              "uncovered_relations": []})
            continue
        current_terminal = sorted(set(current["action_primitives"]) & terminal_primitives)
        covering_task: str | None = None
        if current["contact_relation"] == "tool_mediated":
            for previous in schemas[:index]:
                previous_terminal = sorted(
                    set(previous["action_primitives"]) & terminal_primitives
                )
                if (
                    previous["contact_relation"] == "tool_mediated"
                    and previous["motion_constraint"] == current["motion_constraint"]
                    and previous_terminal == current_terminal
                ):
                    covering_task = previous["task_name"]
                    break
            decision = "transfer" if covering_task is not None else "reset"
            rationale = "matched prior tool-control program" if covering_task else "novel tool-control program"
        else:
            decision = "transfer"
            rationale = "compositional direct/grasped manipulation transfer"
        decisions.append({
            "task_index": index,
            "task_name": current["task_name"],
            "decision": decision,
            "covering_task": covering_task,
            "terminal_primitives": current_terminal,
            "rationale": rationale,
        })
    return decisions


def main() -> None:
    args = parse_args()
    load_dotenv_if_present(PROJECT_ROOT / ".env")
    model = args.model or os.environ.get("OPENAI_MODEL", "gpt-5-mini").strip()
    tasks = get_cw10_tasks("v3")
    source_dir = PROJECT_ROOT / "prompts" / "llm_controller"
    descriptions = json.loads((source_dir / "cw10_v3_task_descriptions.json").read_text(encoding="utf-8"))
    protocol = json.loads((source_dir / "protocol.json").read_text(encoding="utf-8"))
    profiles = [{"task_index": i, "task_name": task, **descriptions["tasks"][task]}
                for i, task in enumerate(tasks)]
    template = Template((PROJECT_ROOT / "prompts" / "llm_critic_schema" / "schema_extractor.txt").read_text(encoding="utf-8"))
    prompt = template.substitute(protocol=json.dumps(protocol, indent=2, sort_keys=True),
                                 task_profiles=json.dumps(profiles, indent=2, sort_keys=True))
    output_dir = PROJECT_ROOT / "outputs" / "llm_critic_selector" / args.run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    if args.schema_input:
        source_payload = json.loads(Path(args.schema_input).expanduser().resolve().read_text(encoding="utf-8"))
        schemas = validate({"schemas": source_payload["schemas"]}, tasks)
        raw_text = ""
    else:
        api_key = read_api_key(args.api_key_env)
        body = {"model": model, "input": prompt, "max_output_tokens": args.max_output_tokens,
                "reasoning": {"effort": "minimal"}, "text": {"verbosity": "low"}}
        request = urllib.request.Request(
            "https://api.openai.com/v1/responses", data=json.dumps(body).encode("utf-8"),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST")
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            details = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenAI API HTTP error {error.code}: {details}") from error
        raw_text = _extract_text_from_response(response_payload)
        (output_dir / "raw_response.txt").write_text(raw_text + "\n", encoding="utf-8")
        schemas = validate(_parse_first_json_object(raw_text), tasks)
    decisions = select(schemas)
    result = {"model": model, "schemas": schemas, "decisions": decisions,
              "rule": "compositional coverage with exact tool-control-program matching; no confidence or fitted threshold"}
    (output_dir / "schema_and_decision.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), **result}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
