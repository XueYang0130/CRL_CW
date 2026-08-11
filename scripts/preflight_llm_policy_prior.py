from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs import get_cw10_tasks
from training.llm_controller import load_dotenv_if_present, read_api_key
from training.llm_policy_prior_controller import (
    build_policy_prior_prompt,
    call_policy_prior_controller,
    no_transfer_decision,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit all LLM policy-prior choices without running training."
    )
    parser.add_argument("--env-version", choices=("v2", "v3"), default="v3")
    parser.add_argument("--sequence-task-count", type=int, default=10)
    parser.add_argument("--prompt-dir", default="prompts/llm_policy_prior")
    parser.add_argument("--minimum-confidence", type=float, default=0.7)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-output-tokens", type=int, default=1024)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    if not 1 <= args.sequence_task_count <= 10:
        parser.error("--sequence-task-count must be in [1, 10].")
    if not 0.0 <= args.minimum_confidence <= 1.0:
        parser.error("--minimum-confidence must be in [0, 1].")
    return args


def main() -> None:
    args = parse_args()
    load_dotenv_if_present(PROJECT_ROOT / ".env")
    model = args.model or os.environ.get("OPENAI_MODEL", "gpt-5-mini").strip()
    api_key = read_api_key(args.api_key_env)
    tasks = get_cw10_tasks(args.env_version)[: args.sequence_task_count]
    output_dir = PROJECT_ROOT / "outputs" / "llm_policy_prior_preflight" / args.run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    prompt_output_dir = output_dir / "prompts"
    prompt_output_dir.mkdir()

    decisions: list[dict[str, object]] = []
    for task_index, task_name in enumerate(tasks):
        visible = list(tasks[:task_index])
        prompt_path: Path | None = None
        if task_index == 0:
            decision = no_transfer_decision(
                "No previous policy head exists for the first task.",
                source="first_task",
            )
        else:
            prompt = build_policy_prior_prompt(
                current_task_name=task_name,
                previous_task_names=visible,
                prompt_dir=args.prompt_dir,
            )
            prompt_path = prompt_output_dir / f"task_{task_index:02d}_{task_name}.txt"
            prompt_path.write_text(prompt, encoding="utf-8")
            try:
                decision = call_policy_prior_controller(
                    api_key=api_key,
                    model=model,
                    prompt=prompt,
                    max_output_tokens=args.max_output_tokens,
                    previous_task_names=visible,
                    minimum_confidence=args.minimum_confidence,
                )
            except Exception as error:
                decision = no_transfer_decision(
                    f"Controller failure: {type(error).__name__}: {error}"
                )
        row = {
            "task_index": task_index,
            "task_name": task_name,
            "visible_previous_task_names": visible,
            "source_task_name": decision.source_task_name,
            "confidence": decision.confidence,
            "reason": decision.reason,
            "source": decision.source,
            "raw_response_text": decision.raw_response_text,
            "prompt_path": (
                None if prompt_path is None else str(prompt_path.relative_to(output_dir))
            ),
        }
        decisions.append(row)
        print(
            f"[prior-preflight] task={task_name} "
            f"source={decision.source_task_name or 'none'} "
            f"confidence={decision.confidence:.3f} decision_source={decision.source}"
        )

    payload = {
        "model": model,
        "env_version": args.env_version,
        "minimum_confidence": args.minimum_confidence,
        "tasks": tasks,
        "decisions": decisions,
    }
    (output_dir / "decisions.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(output_dir), "decisions": len(decisions)}, indent=2))


if __name__ == "__main__":
    main()
