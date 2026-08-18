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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--max-output-tokens", type=int, default=2500)
    parser.add_argument("--run-name", default="cw10_v3_critic_clusters")
    return parser.parse_args()


def _nonempty_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be non-empty text.")
    return " ".join(value.split())


def validate(payload: dict[str, Any], tasks: list[str]) -> dict[str, Any]:
    clusters = payload.get("clusters")
    if not isinstance(clusters, list) or not 2 <= len(clusters) <= 6:
        raise ValueError("clusters must contain between 2 and 6 entries.")
    expected_fields = {
        "cluster_name",
        "task_indices",
        "task_names",
        "shared_value_structure",
        "separation_reason",
    }
    seen_indices: list[int] = []
    names: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for cluster in clusters:
        if not isinstance(cluster, dict) or set(cluster) != expected_fields:
            raise ValueError("Each cluster must contain exactly the requested fields.")
        cluster_name = _nonempty_text(cluster["cluster_name"], "cluster_name")
        if cluster_name in names:
            raise ValueError(f"Duplicate cluster_name: {cluster_name}.")
        names.add(cluster_name)
        indices = cluster["task_indices"]
        task_names = cluster["task_names"]
        if (
            not isinstance(indices, list)
            or not indices
            or not all(isinstance(index, int) for index in indices)
        ):
            raise ValueError("task_indices must be a non-empty integer array.")
        if len(indices) != len(set(indices)):
            raise ValueError("task_indices must be unique within each cluster.")
        if not isinstance(task_names, list) or task_names != [tasks[index] for index in indices]:
            raise ValueError(f"task_names do not match task_indices in {cluster_name}.")
        ordered_pairs = sorted(zip(indices, task_names), key=lambda pair: pair[0])
        indices = [pair[0] for pair in ordered_pairs]
        task_names = [pair[1] for pair in ordered_pairs]
        seen_indices.extend(indices)
        normalized.append(
            {
                "cluster_name": cluster_name,
                "task_indices": indices,
                "task_names": task_names,
                "shared_value_structure": _nonempty_text(
                    cluster["shared_value_structure"], "shared_value_structure"
                ),
                "separation_reason": _nonempty_text(
                    cluster["separation_reason"], "separation_reason"
                ),
            }
        )
    if sorted(seen_indices) != list(range(len(tasks))):
        raise ValueError("Every task must appear exactly once across clusters.")
    return {
        "clusters": normalized,
        "global_rationale": _nonempty_text(payload.get("global_rationale"), "global_rationale"),
    }


def main() -> None:
    args = parse_args()
    load_dotenv_if_present(PROJECT_ROOT / ".env")
    model = args.model or os.environ.get("OPENAI_MODEL", "gpt-5-mini").strip()
    tasks = get_cw10_tasks("v3")
    asset_dir = PROJECT_ROOT / "prompts" / "llm_controller"
    descriptions = json.loads(
        (asset_dir / "cw10_v3_task_descriptions.json").read_text(encoding="utf-8")
    )
    protocol = json.loads((asset_dir / "protocol.json").read_text(encoding="utf-8"))
    profiles = [
        {"task_index": index, "task_name": task, **descriptions["tasks"][task]}
        for index, task in enumerate(tasks)
    ]
    template = Template(
        (PROJECT_ROOT / "prompts" / "llm_critic_clustering" / "task_clusterer.txt").read_text(
            encoding="utf-8"
        )
    )
    prompt = template.substitute(
        protocol=json.dumps(protocol, indent=2, sort_keys=True),
        task_profiles=json.dumps(profiles, indent=2, sort_keys=True),
    )
    output_dir = PROJECT_ROOT / "outputs" / "llm_critic_clusters" / args.run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "prompt.txt").write_text(prompt, encoding="utf-8")

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
        headers={
            "Authorization": f"Bearer {read_api_key(args.api_key_env)}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        details = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI API HTTP error {error.code}: {details}") from error
    raw_text = _extract_text_from_response(response_payload)
    (output_dir / "raw_response.txt").write_text(raw_text + "\n", encoding="utf-8")
    result = validate(_parse_first_json_object(raw_text), tasks)
    result["model"] = model
    (output_dir / "clusters.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(output_dir), **result}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
