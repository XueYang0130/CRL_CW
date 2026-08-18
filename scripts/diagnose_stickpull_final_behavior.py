from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs import DEFAULT_EPISODE_LENGTH, extract_success, make_cw_env
from methods import get_method
from scripts.build_stickpull_matched_memory import build_agent_from_config
from utils import load_sac_checkpoint, write_csv, write_json


EPISODE_FIELDS = (
    "condition",
    "episode",
    "episode_return",
    "success",
    "near_object_any",
    "grasp_success_any",
    "pick_condition_any",
    "inserted_any",
    "handle_target_any",
    "joint_terminal_any",
    "near_object_fraction",
    "grasp_success_fraction",
    "pick_condition_fraction",
    "inserted_fraction",
    "handle_target_fraction",
    "joint_terminal_fraction",
    "min_hand_stick_distance",
    "max_stick_lift",
    "min_stick_container_distance",
    "min_handle_target_distance",
    "min_end_handle_distance",
    "max_container_displacement",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Locate the behavioral failure stage of trained stick-pull policies "
            "without performing learning updates."
        )
    )
    parser.add_argument("--run-dirs", nargs="+", required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=910_000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output-dir",
        default="outputs/diagnostics/stickpull_final_behavior_matrix_seed0",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object in {path}.")
    return payload


def seed_episode(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def unwrap_attribute(env: Any, name: str) -> Any:
    current = env
    while current is not None:
        if name in getattr(current, "__dict__", {}):
            return getattr(current, name)
        current = getattr(current, "env", None)
    raise AttributeError(f"Environment wrapper chain has no attribute {name!r}.")


def rollout_episode(
    *,
    condition: str,
    episode: int,
    episode_seed: int,
    env: Any,
    agent: Any,
    task_index: int,
    max_episode_steps: int,
) -> dict[str, float | int | str]:
    seed_episode(episode_seed)
    observation, _ = env.reset(seed=episode_seed)
    observation = np.asarray(observation, dtype=np.float32)
    physical = observation[:-agent.task_id_dim] if agent.task_id_dim else observation
    initial_stick = physical[4:7].copy()
    initial_container = physical[11:14].copy() + np.asarray(
        [0.05, 0.0, 0.0], dtype=np.float32
    )

    rewards: list[float] = []
    near_object: list[bool] = []
    grasp_success: list[bool] = []
    pick_condition: list[bool] = []
    inserted: list[bool] = []
    handle_target: list[bool] = []
    joint_terminal: list[bool] = []
    hand_stick_distance: list[float] = []
    stick_lift: list[float] = []
    stick_container_distance: list[float] = []
    handle_target_distance: list[float] = []
    end_handle_distance: list[float] = []
    container_displacement: list[float] = []
    succeeded = False

    base_env = env.unwrapped
    for _ in range(max_episode_steps):
        action = agent.select_action_with_head(
            observation,
            head_index=task_index,
            deterministic=False,
        )
        next_observation, reward, terminated, truncated, info = env.step(action)
        observation = np.asarray(next_observation, dtype=np.float32)
        physical = (
            observation[:-agent.task_id_dim] if agent.task_id_dim else observation
        )
        hand = physical[0:3]
        stick = physical[4:7]
        handle = physical[11:14]
        container = handle + np.asarray([0.05, 0.0, 0.0], dtype=np.float32)
        target = np.asarray(base_env._target_pos, dtype=np.float32)
        end_of_stick = np.asarray(base_env._get_site_pos("stick_end"), dtype=np.float32)

        is_inserted = bool(base_env._stick_is_inserted(handle, end_of_stick))
        target_distance = float(np.linalg.norm(handle - target))
        target_reached = target_distance <= 0.12
        picked = bool(
            stick[2]
            >= float(unwrap_attribute(env, "height_target")) - 0.01
        )

        rewards.append(float(reward))
        near_object.append(float(info.get("near_object", 0.0)) > 0.0)
        grasp_success.append(float(info.get("grasp_success", 0.0)) > 0.0)
        pick_condition.append(picked)
        inserted.append(is_inserted)
        handle_target.append(target_reached)
        joint_terminal.append(is_inserted and target_reached)
        hand_stick_distance.append(float(np.linalg.norm(hand - stick)))
        stick_lift.append(float(stick[2] - initial_stick[2]))
        stick_container_distance.append(float(np.linalg.norm(stick - container)))
        handle_target_distance.append(target_distance)
        end_handle_distance.append(float(np.linalg.norm(end_of_stick - handle)))
        container_displacement.append(
            float(np.linalg.norm(container[:2] - initial_container[:2]))
        )
        succeeded = succeeded or extract_success(info) > 0.0
        if terminated or truncated:
            break

    def fraction(values: list[bool]) -> float:
        return float(np.mean(np.asarray(values, dtype=np.float32)))

    return {
        "condition": condition,
        "episode": episode,
        "episode_return": float(np.sum(rewards)),
        "success": int(succeeded),
        "near_object_any": int(any(near_object)),
        "grasp_success_any": int(any(grasp_success)),
        "pick_condition_any": int(any(pick_condition)),
        "inserted_any": int(any(inserted)),
        "handle_target_any": int(any(handle_target)),
        "joint_terminal_any": int(any(joint_terminal)),
        "near_object_fraction": fraction(near_object),
        "grasp_success_fraction": fraction(grasp_success),
        "pick_condition_fraction": fraction(pick_condition),
        "inserted_fraction": fraction(inserted),
        "handle_target_fraction": fraction(handle_target),
        "joint_terminal_fraction": fraction(joint_terminal),
        "min_hand_stick_distance": float(np.min(hand_stick_distance)),
        "max_stick_lift": float(np.max(stick_lift)),
        "min_stick_container_distance": float(np.min(stick_container_distance)),
        "min_handle_target_distance": float(np.min(handle_target_distance)),
        "min_end_handle_distance": float(np.min(end_handle_distance)),
        "max_container_displacement": float(np.max(container_displacement)),
    }


def summarize(condition: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    mean_fields = [field for field in EPISODE_FIELDS if field not in {"condition", "episode"}]
    summary: dict[str, Any] = {"condition": condition, "episodes": len(rows)}
    for field in mean_fields:
        summary[f"mean_{field}"] = float(np.mean([float(row[field]) for row in rows]))
    summary["best_episode_return"] = float(
        max(float(row["episode_return"]) for row in rows)
    )
    return summary


def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive.")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    episode_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    expected_source: Path | None = None
    for run_dir_value in args.run_dirs:
        run_dir = Path(run_dir_value).expanduser().resolve()
        causal_config = load_json(run_dir / "config.json")
        source_run = Path(causal_config["source_run_dir"]).expanduser().resolve()
        if expected_source is None:
            expected_source = source_run
        elif source_run != expected_source:
            raise ValueError("All conditions must originate from the same source run.")
        source_config = load_json(source_run / "config.json")
        tasks = list(source_config["tasks"])
        task_index = int(causal_config["new_task_index"])
        if tasks[task_index] != "stick-pull-v3":
            raise ValueError("This diagnostic only supports stick-pull-v3.")

        agent = build_agent_from_config(config=source_config, device=args.device)
        load_sac_checkpoint(
            agent=agent,
            path=run_dir / "checkpoints" / "final.pt",
            map_location=args.device,
            load_optimizer=False,
        )
        agent.eval()
        method = get_method(str(source_config["method"]))
        max_episode_steps = int(
            source_config.get("max_episode_steps", DEFAULT_EPISODE_LENGTH)
        )
        env = make_cw_env(
            "stick-pull-v3",
            seed=args.seed,
            max_episode_steps=max_episode_steps,
            append_task_id=bool(method.append_task_id),
            env_version=str(source_config["env_version"]),
            reward_function_version=str(source_config["reward_function_version"]),
            num_task_ids=len(tasks),
        )
        condition = f"{causal_config['memory_mode']}+{causal_config['integration']}"
        condition_rows: list[dict[str, Any]] = []
        try:
            for episode in range(args.episodes):
                row = rollout_episode(
                    condition=condition,
                    episode=episode,
                    episode_seed=args.seed + episode,
                    env=env,
                    agent=agent,
                    task_index=task_index,
                    max_episode_steps=max_episode_steps,
                )
                condition_rows.append(row)
                episode_rows.append(row)
        finally:
            env.close()
        summary = summarize(condition, condition_rows)
        summaries.append(summary)
        print(
            f"[stickpull-behavior] condition={condition} "
            f"success={summary['mean_success']:.3f} "
            f"grasp={summary['mean_grasp_success_any']:.3f} "
            f"pick={summary['mean_pick_condition_any']:.3f} "
            f"inserted={summary['mean_inserted_any']:.3f} "
            f"target={summary['mean_handle_target_any']:.3f}",
            flush=True,
        )

    write_csv(output_dir / "episodes.csv", EPISODE_FIELDS, episode_rows)
    write_json(output_dir / "summary.json", {"conditions": summaries})
    summary_fields = tuple(summaries[0].keys())
    write_csv(output_dir / "summary.csv", summary_fields, summaries)
    print(json.dumps({"output_dir": str(output_dir), "conditions": summaries}, indent=2))


if __name__ == "__main__":
    main()
