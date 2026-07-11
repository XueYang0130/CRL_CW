"""Run a short random-action smoke test on one CW10 environment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from crl_cw.envs import extract_success, make_cw_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        default="hammer-v1",
        help="Original ClonEx-SAC CW10 task name.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.steps <= 0:
        raise ValueError("--steps must be positive.")

    env = make_cw_env(task_name=args.task, seed=args.seed)

    try:
        observation, reset_info = env.reset(seed=args.seed)

        print(f"Task: {args.task}")
        print(f"Observation shape: {np.asarray(observation).shape}")
        print(f"Observation dtype: {np.asarray(observation).dtype}")
        print(f"Action space: {env.action_space}")
        print(f"Action shape: {env.action_space.shape}")
        print(f"Episode limit: {env.spec.max_episode_steps}")
        print(f"Reset info keys: {sorted(reset_info.keys())}")

        for step in range(1, args.steps + 1):
            action = env.action_space.sample()

            observation, reward, terminated, truncated, info = env.step(action)

            success = extract_success(info)

            print(
                f"step={step:02d} "
                f"reward={float(reward):.4f} "
                f"success={success:.0f} "
                f"terminated={terminated} "
                f"truncated={truncated}"
            )

            if terminated or truncated:
                observation, reset_info = env.reset()

        print("Environment smoke test passed.")

    finally:
        env.close()


if __name__ == "__main__":
    main()