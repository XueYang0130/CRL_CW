from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents import SACAgent
from envs import DEFAULT_EPISODE_LENGTH, extract_success, make_cw_env
from methods import (
    bc_gradient_strategy_for_method,
    get_method,
    method_defaults,
)
from training.sac_trainer import ExplorationHeadSelector
from utils import load_sac_checkpoint, write_csv, write_json


EPISODE_FIELDS = (
    "method",
    "seed",
    "run_name",
    "rollout_kind",
    "repeat",
    "episode",
    "head_index",
    "head_task",
    "episode_return",
    "success",
    "episode_length",
    "min_hand_stick_distance",
    "max_stick_lift",
    "max_container_xy_displacement",
    "min_container_target_xy_distance",
    "mean_action_norm",
    "action_saturation_fraction",
)


SUMMARY_FIELDS = (
    "method",
    "seed",
    "run_name",
    "rollout_kind",
    "head_index",
    "head_task",
    "episodes",
    "states",
    "mean_return",
    "std_return",
    "success_rate",
    "mean_episode_length",
    "physical_state_covariance_trace",
    "physical_state_effective_rank",
    "mean_distance_from_episode_initial_state",
    "mean_hand_stick_distance",
    "min_hand_stick_distance",
    "reach_fraction_010",
    "reach_fraction_005",
    "mean_stick_lift",
    "max_stick_lift",
    "lift_fraction_005",
    "mean_stick_container_distance",
    "min_stick_container_distance",
    "mean_container_target_xy_distance",
    "min_container_target_xy_distance",
    "mean_container_xy_displacement",
    "max_container_xy_displacement",
    "mean_action_norm",
    "action_saturation_fraction",
    "mean_policy_log_std",
    "mean_current_q",
    "mean_current_q_disagreement",
)


INITIALIZATION_FIELDS = (
    "method",
    "seed",
    "run_name",
    "current_task_index",
    "current_task_name",
    "checkpoint",
    "protocol_selected_head",
    "protocol_selected_task",
    "robust_best_head",
    "robust_best_task",
    "robust_best_return",
    "current_head_mean_action_norm",
    "current_head_mean_log_std",
    "current_head_q_mean",
    "current_head_q_std",
    "current_head_q_disagreement",
)


PAIR_FIELDS = (
    "seed",
    "left_method",
    "right_method",
    "left_protocol_selected_head",
    "right_protocol_selected_head",
    "left_robust_best_head",
    "right_robust_best_head",
    "protocol_state_centroid_distance",
    "protocol_state_covariance_distance",
    "protocol_state_mmd_rbf",
    "current_head_symmetric_gaussian_kl",
    "current_head_mean_action_l2",
    "current_head_mean_action_mae",
    "actor_backbone_linear_cka",
    "critic1_backbone_linear_cka",
    "actor_backbone_relative_parameter_distance",
    "current_actor_head_relative_parameter_distance",
    "critic1_backbone_relative_parameter_distance",
    "current_critic1_head_relative_parameter_distance",
)


@dataclass
class EpisodeTrace:
    states: np.ndarray
    actions: np.ndarray
    log_stds: np.ndarray
    q1_values: np.ndarray
    q2_values: np.ndarray
    rewards: np.ndarray
    success: bool
    initial_physical_state: np.ndarray

    @property
    def episode_return(self) -> float:
        return float(self.rewards.sum())

    @property
    def episode_length(self) -> int:
        return int(self.states.shape[0])


@dataclass
class RunDiagnostic:
    method: str
    seed: int
    run_name: str
    task_names: tuple[str, ...]
    protocol_selected_head: int
    robust_best_head: int
    protocol_states: np.ndarray
    reset_actor_means: np.ndarray
    reset_actor_log_stds: np.ndarray
    reset_actor_actions: np.ndarray
    actor_features: np.ndarray
    critic1_features: np.ndarray
    actor_backbone_parameters: np.ndarray
    current_actor_head_parameters: np.ndarray
    critic1_backbone_parameters: np.ndarray
    current_critic1_head_parameters: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare pre-stick-pull checkpoints under the exact old-head "
            "best-return exploration rule without performing learning updates."
        )
    )
    parser.add_argument("--run-dirs", nargs="+", required=True)
    parser.add_argument("--new-task-index", type=int, default=4)
    parser.add_argument("--episodes-per-head", type=int, default=10)
    parser.add_argument("--current-head-episodes", type=int, default=10)
    parser.add_argument("--protocol-episodes", type=int, default=50)
    parser.add_argument("--reset-states", type=int, default=100)
    parser.add_argument("--diagnostic-seed", type=int, default=840_000)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/diagnostics",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="stickpull_initialization_comparison",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object at {path}.")
    return payload


def build_agent(config: dict[str, Any], device: str) -> SACAgent:
    method_id = str(config["method"])
    method = get_method(method_id)
    task_names = tuple(str(value) for value in config["tasks"])
    resolved_config: dict[str, Any] = {
        "bc_gradient_strategy": bc_gradient_strategy_for_method(method_id),
        "bc_max_norm_ratio": 1.0,
        "bc_combination_strategy": "average",
        "bc_adaptive_target_ratio": 0.2,
        "bc_adaptive_conflict_ratio": 0.05,
        "bc_cagrad_alpha": 0.5,
        "gradient_diagnostics": False,
        "gradient_diagnostics_interval": 500,
        "gradient_diagnostics_source_batch_size": 128,
        **method_defaults(method_id),
        **config,
        "device": device,
    }
    env = make_cw_env(
        task_names[0],
        seed=int(resolved_config["seed"]),
        max_episode_steps=int(
            resolved_config.get("max_episode_steps", DEFAULT_EPISODE_LENGTH)
        ),
        append_task_id=method.append_task_id,
        env_version=str(resolved_config["env_version"]),
        reward_function_version=str(resolved_config["reward_function_version"]),
        num_task_ids=len(task_names),
    )
    try:
        agent = method.build_agent(
            args=SimpleNamespace(**resolved_config),
            observation_dim=int(env.observation_space.shape[0]),
            action_dim=int(env.action_space.shape[0]),
            action_low=env.action_space.low,
            action_high=env.action_space.high,
            total_tasks=len(task_names),
        )
    finally:
        env.close()
    return agent.to(device)


def seed_episode(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def actor_observation(
    agent: SACAgent,
    observation: np.ndarray,
    head_index: int,
) -> np.ndarray:
    return agent._observation_for_actor_head(  # noqa: SLF001
        observation=observation,
        head_index=head_index,
    )


@torch.inference_mode()
def policy_values(
    agent: SACAgent,
    observation: np.ndarray,
    head_index: int,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    overridden = actor_observation(agent, observation, head_index)
    actor_tensor = torch.as_tensor(
        overridden,
        dtype=torch.float32,
        device=agent.device,
    ).unsqueeze(0)
    mean, log_std = agent.actor.distribution_parameters(actor_tensor)
    action = agent.select_action_with_head(
        observation,
        head_index=head_index,
        deterministic=False,
    )
    state_tensor = torch.as_tensor(
        observation,
        dtype=torch.float32,
        device=agent.device,
    ).unsqueeze(0)
    action_tensor = torch.as_tensor(
        action,
        dtype=torch.float32,
        device=agent.device,
    ).unsqueeze(0)
    q1 = float(agent.critic1(state_tensor, action_tensor).item())
    q2 = float(agent.critic2(state_tensor, action_tensor).item())
    return (
        np.asarray(action, dtype=np.float32),
        log_std.squeeze(0).cpu().numpy(),
        q1,
        q2,
    )


def rollout_episode(
    *,
    env: Any,
    agent: SACAgent,
    head_index: int,
    seed: int,
    max_episode_steps: int,
) -> EpisodeTrace:
    seed_episode(seed)
    reset_result = env.reset(seed=seed)
    observation = reset_result[0] if isinstance(reset_result, tuple) else reset_result
    observation = np.asarray(observation, dtype=np.float32)
    initial_physical_state = observation[:-agent.task_id_dim].copy()

    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    log_stds: list[np.ndarray] = []
    q1_values: list[float] = []
    q2_values: list[float] = []
    rewards: list[float] = []
    succeeded = False

    for _ in range(max_episode_steps):
        action, log_std, q1, q2 = policy_values(
            agent,
            observation,
            head_index,
        )
        states.append(observation.copy())
        actions.append(action.copy())
        log_stds.append(log_std.copy())
        q1_values.append(q1)
        q2_values.append(q2)

        step_result = env.step(action)
        if len(step_result) != 5:
            raise RuntimeError("This diagnostic requires a Gymnasium five-value step.")
        next_observation, reward, terminated, truncated, info = step_result
        rewards.append(float(reward))
        succeeded = succeeded or extract_success(info) > 0.0
        observation = np.asarray(next_observation, dtype=np.float32)
        if terminated or truncated:
            break

    return EpisodeTrace(
        states=np.stack(states),
        actions=np.stack(actions),
        log_stds=np.stack(log_stds),
        q1_values=np.asarray(q1_values, dtype=np.float32),
        q2_values=np.asarray(q2_values, dtype=np.float32),
        rewards=np.asarray(rewards, dtype=np.float32),
        success=bool(succeeded),
        initial_physical_state=initial_physical_state,
    )


def physical_states(trace: EpisodeTrace, task_id_dim: int) -> np.ndarray:
    if task_id_dim <= 0:
        return trace.states
    return trace.states[:, :-task_id_dim]


def trace_features(trace: EpisodeTrace, task_id_dim: int) -> dict[str, np.ndarray]:
    states = physical_states(trace, task_id_dim)
    hand = states[:, 0:3]
    stick = states[:, 4:7]
    handle = states[:, 11:14]
    target = states[:, -3:]
    container = handle + np.asarray([0.05, 0.0, 0.0], dtype=np.float32)

    initial = trace.initial_physical_state
    initial_hand = initial[0:3]
    initial_stick = initial[4:7]
    initial_container = initial[11:14] + np.asarray(
        [0.05, 0.0, 0.0], dtype=np.float32
    )
    return {
        "distance_from_initial": np.linalg.norm(states - initial, axis=1),
        "hand_stick_distance": np.linalg.norm(hand - stick, axis=1),
        "stick_lift": stick[:, 2] - initial_stick[2],
        "stick_container_distance": np.linalg.norm(stick - container, axis=1),
        "container_target_xy_distance": np.linalg.norm(
            container[:, :2] - target[:, :2], axis=1
        ),
        "container_xy_displacement": np.linalg.norm(
            container[:, :2] - initial_container[:2], axis=1
        ),
        "hand_displacement": np.linalg.norm(hand - initial_hand, axis=1),
    }


def episode_row(
    *,
    method: str,
    seed: int,
    run_name: str,
    rollout_kind: str,
    repeat: int,
    episode: int,
    head_index: int,
    head_task: str,
    trace: EpisodeTrace,
    task_id_dim: int,
) -> dict[str, Any]:
    features = trace_features(trace, task_id_dim)
    action_norm = np.linalg.norm(trace.actions, axis=1)
    return {
        "method": method,
        "seed": seed,
        "run_name": run_name,
        "rollout_kind": rollout_kind,
        "repeat": repeat,
        "episode": episode,
        "head_index": head_index,
        "head_task": head_task,
        "episode_return": trace.episode_return,
        "success": int(trace.success),
        "episode_length": trace.episode_length,
        "min_hand_stick_distance": float(features["hand_stick_distance"].min()),
        "max_stick_lift": float(features["stick_lift"].max()),
        "max_container_xy_displacement": float(
            features["container_xy_displacement"].max()
        ),
        "min_container_target_xy_distance": float(
            features["container_target_xy_distance"].min()
        ),
        "mean_action_norm": float(action_norm.mean()),
        "action_saturation_fraction": float(
            np.mean(np.abs(trace.actions) >= 0.95)
        ),
    }


def effective_rank(states: np.ndarray) -> float:
    if states.shape[0] <= 1:
        return 0.0
    covariance = np.cov(states, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(covariance)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    total = float(eigenvalues.sum())
    if total <= 1e-12:
        return 0.0
    probabilities = eigenvalues / total
    probabilities = probabilities[probabilities > 0.0]
    return float(np.exp(-(probabilities * np.log(probabilities)).sum()))


def summarize_traces(
    *,
    method: str,
    seed: int,
    run_name: str,
    rollout_kind: str,
    head_index: int,
    head_task: str,
    traces: list[EpisodeTrace],
    task_id_dim: int,
) -> dict[str, Any]:
    states = np.concatenate(
        [physical_states(trace, task_id_dim) for trace in traces], axis=0
    )
    actions = np.concatenate([trace.actions for trace in traces], axis=0)
    log_stds = np.concatenate([trace.log_stds for trace in traces], axis=0)
    q1_values = np.concatenate([trace.q1_values for trace in traces], axis=0)
    q2_values = np.concatenate([trace.q2_values for trace in traces], axis=0)
    features = {
        key: np.concatenate(values, axis=0)
        for key, values in _feature_lists(traces, task_id_dim).items()
    }
    returns = np.asarray([trace.episode_return for trace in traces])
    lengths = np.asarray([trace.episode_length for trace in traces])
    successes = np.asarray([trace.success for trace in traces], dtype=np.float32)
    covariance = np.cov(states, rowvar=False)
    minimum_q = np.minimum(q1_values, q2_values)
    return {
        "method": method,
        "seed": seed,
        "run_name": run_name,
        "rollout_kind": rollout_kind,
        "head_index": head_index,
        "head_task": head_task,
        "episodes": len(traces),
        "states": int(states.shape[0]),
        "mean_return": float(returns.mean()),
        "std_return": float(returns.std()),
        "success_rate": float(successes.mean()),
        "mean_episode_length": float(lengths.mean()),
        "physical_state_covariance_trace": float(np.trace(covariance)),
        "physical_state_effective_rank": effective_rank(states),
        "mean_distance_from_episode_initial_state": float(
            features["distance_from_initial"].mean()
        ),
        "mean_hand_stick_distance": float(features["hand_stick_distance"].mean()),
        "min_hand_stick_distance": float(features["hand_stick_distance"].min()),
        "reach_fraction_010": float(
            np.mean(features["hand_stick_distance"] < 0.10)
        ),
        "reach_fraction_005": float(
            np.mean(features["hand_stick_distance"] < 0.05)
        ),
        "mean_stick_lift": float(features["stick_lift"].mean()),
        "max_stick_lift": float(features["stick_lift"].max()),
        "lift_fraction_005": float(np.mean(features["stick_lift"] > 0.005)),
        "mean_stick_container_distance": float(
            features["stick_container_distance"].mean()
        ),
        "min_stick_container_distance": float(
            features["stick_container_distance"].min()
        ),
        "mean_container_target_xy_distance": float(
            features["container_target_xy_distance"].mean()
        ),
        "min_container_target_xy_distance": float(
            features["container_target_xy_distance"].min()
        ),
        "mean_container_xy_displacement": float(
            features["container_xy_displacement"].mean()
        ),
        "max_container_xy_displacement": float(
            features["container_xy_displacement"].max()
        ),
        "mean_action_norm": float(np.linalg.norm(actions, axis=1).mean()),
        "action_saturation_fraction": float(np.mean(np.abs(actions) >= 0.95)),
        "mean_policy_log_std": float(log_stds.mean()),
        "mean_current_q": float(minimum_q.mean()),
        "mean_current_q_disagreement": float(
            np.abs(q1_values - q2_values).mean()
        ),
    }


def _feature_lists(
    traces: list[EpisodeTrace], task_id_dim: int
) -> dict[str, list[np.ndarray]]:
    values: dict[str, list[np.ndarray]] = {}
    for trace in traces:
        for key, array in trace_features(trace, task_id_dim).items():
            values.setdefault(key, []).append(array)
    return values


def flatten_parameters(parameters: Any) -> np.ndarray:
    tensors = [value.detach().cpu().reshape(-1) for value in parameters]
    if not tensors:
        return np.empty((0,), dtype=np.float32)
    return torch.cat(tensors).numpy()


def current_actor_head_parameters(agent: SACAgent, task_index: int) -> np.ndarray:
    values: list[torch.Tensor] = []
    for layer in (agent.actor.mean_head, agent.actor.log_std_head):
        output_dim = layer.out_features // agent.num_tasks
        weight = layer.weight.reshape(
            output_dim, agent.num_tasks, layer.in_features
        )[:, task_index, :]
        bias = layer.bias.reshape(output_dim, agent.num_tasks)[:, task_index]
        values.extend((weight.reshape(-1), bias.reshape(-1)))
    return flatten_parameters(values)


def current_critic_head_parameters(
    critic: torch.nn.Module, task_index: int
) -> np.ndarray:
    return flatten_parameters(
        (
            critic.q_head.weight[task_index],
            critic.q_head.bias[task_index],
        )
    )


@torch.inference_mode()
def reset_state_diagnostics(
    *,
    agent: SACAgent,
    env: Any,
    task_index: int,
    reset_states: int,
    seed: int,
) -> dict[str, np.ndarray | float]:
    observations = []
    for index in range(reset_states):
        reset_result = env.reset(seed=seed + index)
        observation = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        observations.append(np.asarray(observation, dtype=np.float32))
    observation_array = np.stack(observations)
    observation_tensor = torch.as_tensor(
        observation_array,
        dtype=torch.float32,
        device=agent.device,
    )
    means, log_stds = agent.actor.distribution_parameters(observation_tensor)
    mean_actions = torch.tanh(means) * agent.actor.action_scale + agent.actor.action_bias
    q1 = agent.critic1(observation_tensor, mean_actions)
    q2 = agent.critic2(observation_tensor, mean_actions)
    actor_inputs = observation_tensor[:, :-agent.task_id_dim]
    actor_features = agent.actor.backbone(actor_inputs)
    critic_inputs = torch.cat((actor_inputs, mean_actions), dim=-1)
    critic1_features = agent.critic1.backbone(critic_inputs)
    return {
        "observations": observation_array,
        "means": means.cpu().numpy(),
        "log_stds": log_stds.cpu().numpy(),
        "mean_actions": mean_actions.cpu().numpy(),
        "q1": q1.cpu().numpy(),
        "q2": q2.cpu().numpy(),
        "actor_features": actor_features.cpu().numpy(),
        "critic1_features": critic1_features.cpu().numpy(),
    }


def normalized_parameter_distance(left: np.ndarray, right: np.ndarray) -> float:
    denominator = np.linalg.norm(left) + np.linalg.norm(right) + 1e-12
    return float(2.0 * np.linalg.norm(left - right) / denominator)


def linear_cka(left: np.ndarray, right: np.ndarray) -> float:
    left_centered = left - left.mean(axis=0, keepdims=True)
    right_centered = right - right.mean(axis=0, keepdims=True)
    cross = left_centered.T @ right_centered
    numerator = float(np.square(cross).sum())
    left_norm = math.sqrt(float(np.square(left_centered.T @ left_centered).sum()))
    right_norm = math.sqrt(float(np.square(right_centered.T @ right_centered).sum()))
    return numerator / (left_norm * right_norm + 1e-12)


def symmetric_gaussian_kl(
    left_mean: np.ndarray,
    left_log_std: np.ndarray,
    right_mean: np.ndarray,
    right_log_std: np.ndarray,
) -> float:
    def kl(
        first_mean: np.ndarray,
        first_log_std: np.ndarray,
        second_mean: np.ndarray,
        second_log_std: np.ndarray,
    ) -> np.ndarray:
        first_var = np.exp(2.0 * first_log_std) + 1e-6
        second_var = np.exp(2.0 * second_log_std) + 1e-6
        return np.sum(
            second_log_std
            - first_log_std
            + (first_var + np.square(first_mean - second_mean))
            / (2.0 * second_var)
            - 0.5,
            axis=1,
        )

    return float(
        0.5
        * (
            kl(left_mean, left_log_std, right_mean, right_log_std).mean()
            + kl(right_mean, right_log_std, left_mean, left_log_std).mean()
        )
    )


def standardized_state_distances(
    left: np.ndarray,
    right: np.ndarray,
    seed: int,
) -> tuple[float, float, float]:
    pooled = np.concatenate((left, right), axis=0)
    scale = pooled.std(axis=0) + 1e-6
    left_scaled = left / scale
    right_scaled = right / scale
    centroid = float(np.linalg.norm(left_scaled.mean(axis=0) - right_scaled.mean(axis=0)))
    left_cov = np.cov(left_scaled, rowvar=False)
    right_cov = np.cov(right_scaled, rowvar=False)
    covariance = float(
        np.linalg.norm(left_cov - right_cov, ord="fro")
        / (np.linalg.norm(left_cov, ord="fro") + np.linalg.norm(right_cov, ord="fro") + 1e-12)
    )

    rng = np.random.default_rng(seed)
    sample_size = min(500, left_scaled.shape[0], right_scaled.shape[0])
    left_sample = left_scaled[
        rng.choice(left_scaled.shape[0], size=sample_size, replace=False)
    ]
    right_sample = right_scaled[
        rng.choice(right_scaled.shape[0], size=sample_size, replace=False)
    ]
    combined = np.concatenate((left_sample, right_sample), axis=0)
    squared = np.sum(
        np.square(combined[:, None, :] - combined[None, :, :]), axis=2
    )
    positive = squared[squared > 0.0]
    bandwidth = float(np.median(positive)) if positive.size else 1.0
    bandwidth = max(bandwidth, 1e-6)
    kernel = np.exp(-squared / (2.0 * bandwidth))
    n = sample_size
    mmd = float(
        kernel[:n, :n].mean()
        + kernel[n:, n:].mean()
        - 2.0 * kernel[:n, n:].mean()
    )
    return centroid, covariance, mmd


def run_one(
    *,
    run_dir: Path,
    args: argparse.Namespace,
    episode_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    initialization_rows: list[dict[str, Any]],
    npz_payload: dict[str, np.ndarray],
) -> RunDiagnostic:
    config = load_json(run_dir / "config.json")
    method = str(config["method"])
    seed = int(config["seed"])
    task_names = tuple(str(value) for value in config["tasks"])
    task_index = int(args.new_task_index)
    if not 0 < task_index < len(task_names):
        raise ValueError("new-task-index must refer to a task after task zero.")
    if task_names[task_index] != "stick-pull-v3":
        raise ValueError(
            f"Expected stick-pull-v3 at index {task_index}, got {task_names[task_index]}."
        )
    agent = build_agent(config, args.device)
    checkpoint = run_dir / "checkpoints" / f"task_{task_index - 1}.pt"
    load_sac_checkpoint(
        agent=agent,
        path=checkpoint,
        map_location=args.device,
        load_optimizer=False,
    )
    agent.eval()
    env = make_cw_env(
        task_names[task_index],
        seed=args.diagnostic_seed,
        max_episode_steps=int(
            config.get("max_episode_steps", DEFAULT_EPISODE_LENGTH)
        ),
        append_task_id=bool(get_method(method).append_task_id),
        env_version=str(config["env_version"]),
        reward_function_version=str(config["reward_function_version"]),
        num_task_ids=len(task_names),
    )
    max_episode_steps = int(config.get("max_episode_steps", DEFAULT_EPISODE_LENGTH))
    run_name = run_dir.name

    robust_traces: dict[int, list[EpisodeTrace]] = {}
    protocol_traces: list[EpisodeTrace] = []
    current_traces: list[EpisodeTrace] = []
    try:
        for head_index in range(task_index):
            traces = []
            for episode in range(args.episodes_per_head):
                trace = rollout_episode(
                    env=env,
                    agent=agent,
                    head_index=head_index,
                    seed=args.diagnostic_seed + 10_000 * seed + episode,
                    max_episode_steps=max_episode_steps,
                )
                traces.append(trace)
                episode_rows.append(
                    episode_row(
                        method=method,
                        seed=seed,
                        run_name=run_name,
                        rollout_kind="old_head_sweep",
                        repeat=0,
                        episode=episode,
                        head_index=head_index,
                        head_task=task_names[head_index],
                        trace=trace,
                        task_id_dim=agent.task_id_dim,
                    )
                )
            robust_traces[head_index] = traces
            summary_rows.append(
                summarize_traces(
                    method=method,
                    seed=seed,
                    run_name=run_name,
                    rollout_kind="old_head_sweep",
                    head_index=head_index,
                    head_task=task_names[head_index],
                    traces=traces,
                    task_id_dim=agent.task_id_dim,
                )
            )

        robust_best_head = max(
            robust_traces,
            key=lambda index: np.mean(
                [trace.episode_return for trace in robust_traces[index]]
            ),
        )

        selector = ExplorationHeadSelector("best_return", task_index + 1)
        selected_head = selector.start_new_episode()
        for episode in range(args.protocol_episodes):
            trace = rollout_episode(
                env=env,
                agent=agent,
                head_index=selected_head,
                seed=args.diagnostic_seed + 100_000 + 10_000 * seed + episode,
                max_episode_steps=max_episode_steps,
            )
            for reward in trace.rewards:
                selector.tell_results(float(reward), float(trace.success))
            protocol_traces.append(trace)
            episode_rows.append(
                episode_row(
                    method=method,
                    seed=seed,
                    run_name=run_name,
                    rollout_kind="protocol_best_return",
                    repeat=0,
                    episode=episode,
                    head_index=selected_head,
                    head_task=task_names[selected_head],
                    trace=trace,
                    task_id_dim=agent.task_id_dim,
                )
            )
            selected_head = selector.start_new_episode()
        protocol_selected_head = int(selected_head)
        summary_rows.append(
            summarize_traces(
                method=method,
                seed=seed,
                run_name=run_name,
                rollout_kind="protocol_best_return",
                head_index=-1,
                head_task="mixed_protocol",
                traces=protocol_traces,
                task_id_dim=agent.task_id_dim,
            )
        )

        for episode in range(args.current_head_episodes):
            trace = rollout_episode(
                env=env,
                agent=agent,
                head_index=task_index,
                seed=args.diagnostic_seed + 200_000 + 10_000 * seed + episode,
                max_episode_steps=max_episode_steps,
            )
            current_traces.append(trace)
            episode_rows.append(
                episode_row(
                    method=method,
                    seed=seed,
                    run_name=run_name,
                    rollout_kind="untrained_current_head",
                    repeat=0,
                    episode=episode,
                    head_index=task_index,
                    head_task=task_names[task_index],
                    trace=trace,
                    task_id_dim=agent.task_id_dim,
                )
            )
        summary_rows.append(
            summarize_traces(
                method=method,
                seed=seed,
                run_name=run_name,
                rollout_kind="untrained_current_head",
                head_index=task_index,
                head_task=task_names[task_index],
                traces=current_traces,
                task_id_dim=agent.task_id_dim,
            )
        )

        reset = reset_state_diagnostics(
            agent=agent,
            env=env,
            task_index=task_index,
            reset_states=args.reset_states,
            seed=args.diagnostic_seed + 300_000 + 10_000 * seed,
        )
    finally:
        env.close()

    q1 = np.asarray(reset["q1"])
    q2 = np.asarray(reset["q2"])
    initialization_rows.append(
        {
            "method": method,
            "seed": seed,
            "run_name": run_name,
            "current_task_index": task_index,
            "current_task_name": task_names[task_index],
            "checkpoint": str(checkpoint),
            "protocol_selected_head": protocol_selected_head,
            "protocol_selected_task": task_names[protocol_selected_head],
            "robust_best_head": robust_best_head,
            "robust_best_task": task_names[robust_best_head],
            "robust_best_return": float(
                np.mean(
                    [trace.episode_return for trace in robust_traces[robust_best_head]]
                )
            ),
            "current_head_mean_action_norm": float(
                np.linalg.norm(np.asarray(reset["mean_actions"]), axis=1).mean()
            ),
            "current_head_mean_log_std": float(
                np.asarray(reset["log_stds"]).mean()
            ),
            "current_head_q_mean": float(np.minimum(q1, q2).mean()),
            "current_head_q_std": float(np.minimum(q1, q2).std()),
            "current_head_q_disagreement": float(np.abs(q1 - q2).mean()),
        }
    )

    key = f"{method}_seed{seed}"
    protocol_states = np.concatenate(
        [physical_states(trace, agent.task_id_dim) for trace in protocol_traces],
        axis=0,
    )
    npz_payload[f"{key}_protocol_states"] = protocol_states
    npz_payload[f"{key}_protocol_actions"] = np.concatenate(
        [trace.actions for trace in protocol_traces], axis=0
    )
    return RunDiagnostic(
        method=method,
        seed=seed,
        run_name=run_name,
        task_names=task_names,
        protocol_selected_head=protocol_selected_head,
        robust_best_head=robust_best_head,
        protocol_states=protocol_states,
        reset_actor_means=np.asarray(reset["means"]),
        reset_actor_log_stds=np.asarray(reset["log_stds"]),
        reset_actor_actions=np.asarray(reset["mean_actions"]),
        actor_features=np.asarray(reset["actor_features"]),
        critic1_features=np.asarray(reset["critic1_features"]),
        actor_backbone_parameters=flatten_parameters(agent.actor.backbone.parameters()),
        current_actor_head_parameters=current_actor_head_parameters(agent, task_index),
        critic1_backbone_parameters=flatten_parameters(agent.critic1.backbone.parameters()),
        current_critic1_head_parameters=current_critic_head_parameters(
            agent.critic1, task_index
        ),
    )


def pair_runs(
    diagnostics: list[RunDiagnostic], diagnostic_seed: int
) -> list[dict[str, Any]]:
    rows = []
    by_seed: dict[int, list[RunDiagnostic]] = {}
    for diagnostic in diagnostics:
        by_seed.setdefault(diagnostic.seed, []).append(diagnostic)
    for seed, candidates in sorted(by_seed.items()):
        if len(candidates) != 2:
            continue
        left, right = sorted(candidates, key=lambda value: value.method)
        centroid, covariance, mmd = standardized_state_distances(
            left.protocol_states,
            right.protocol_states,
            diagnostic_seed + seed,
        )
        rows.append(
            {
                "seed": seed,
                "left_method": left.method,
                "right_method": right.method,
                "left_protocol_selected_head": left.protocol_selected_head,
                "right_protocol_selected_head": right.protocol_selected_head,
                "left_robust_best_head": left.robust_best_head,
                "right_robust_best_head": right.robust_best_head,
                "protocol_state_centroid_distance": centroid,
                "protocol_state_covariance_distance": covariance,
                "protocol_state_mmd_rbf": mmd,
                "current_head_symmetric_gaussian_kl": symmetric_gaussian_kl(
                    left.reset_actor_means,
                    left.reset_actor_log_stds,
                    right.reset_actor_means,
                    right.reset_actor_log_stds,
                ),
                "current_head_mean_action_l2": float(
                    np.linalg.norm(
                        left.reset_actor_actions - right.reset_actor_actions,
                        axis=1,
                    ).mean()
                ),
                "current_head_mean_action_mae": float(
                    np.abs(
                        left.reset_actor_actions - right.reset_actor_actions
                    ).mean()
                ),
                "actor_backbone_linear_cka": linear_cka(
                    left.actor_features, right.actor_features
                ),
                "critic1_backbone_linear_cka": linear_cka(
                    left.critic1_features, right.critic1_features
                ),
                "actor_backbone_relative_parameter_distance": normalized_parameter_distance(
                    left.actor_backbone_parameters,
                    right.actor_backbone_parameters,
                ),
                "current_actor_head_relative_parameter_distance": normalized_parameter_distance(
                    left.current_actor_head_parameters,
                    right.current_actor_head_parameters,
                ),
                "critic1_backbone_relative_parameter_distance": normalized_parameter_distance(
                    left.critic1_backbone_parameters,
                    right.critic1_backbone_parameters,
                ),
                "current_critic1_head_relative_parameter_distance": normalized_parameter_distance(
                    left.current_critic1_head_parameters,
                    right.current_critic1_head_parameters,
                ),
            }
        )
    return rows


def validate_args(args: argparse.Namespace) -> None:
    for name in (
        "episodes_per_head",
        "current_head_episodes",
        "protocol_episodes",
        "reset_states",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    run_dirs = [Path(value).expanduser().resolve() for value in args.run_dirs]
    missing = [str(path) for path in run_dirs if not path.is_dir()]
    if missing:
        raise FileNotFoundError("Missing run directories: " + ", ".join(missing))


def main() -> None:
    args = parse_args()
    validate_args(args)
    output_dir = Path(args.output_dir).expanduser().resolve() / args.run_name
    output_dir.mkdir(parents=True, exist_ok=False)

    episode_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    initialization_rows: list[dict[str, Any]] = []
    npz_payload: dict[str, np.ndarray] = {}
    diagnostics = []
    for value in args.run_dirs:
        run_dir = Path(value).expanduser().resolve()
        diagnostic = run_one(
            run_dir=run_dir,
            args=args,
            episode_rows=episode_rows,
            summary_rows=summary_rows,
            initialization_rows=initialization_rows,
            npz_payload=npz_payload,
        )
        diagnostics.append(diagnostic)
        print(
            f"[stickpull-init] method={diagnostic.method} seed={diagnostic.seed} "
            f"protocol_head={diagnostic.protocol_selected_head} "
            f"robust_head={diagnostic.robust_best_head}"
        )

    pair_rows = pair_runs(diagnostics, args.diagnostic_seed)
    write_csv(output_dir / "episodes.csv", EPISODE_FIELDS, episode_rows)
    write_csv(output_dir / "policy_summaries.csv", SUMMARY_FIELDS, summary_rows)
    write_csv(
        output_dir / "initialization_summaries.csv",
        INITIALIZATION_FIELDS,
        initialization_rows,
    )
    write_csv(output_dir / "pairwise_comparisons.csv", PAIR_FIELDS, pair_rows)
    np.savez_compressed(output_dir / "state_samples.npz", **npz_payload)
    write_json(
        output_dir / "summary.json",
        {
            "output_dir": str(output_dir),
            "new_task_index": args.new_task_index,
            "episodes_per_head": args.episodes_per_head,
            "current_head_episodes": args.current_head_episodes,
            "protocol_episodes": args.protocol_episodes,
            "reset_states": args.reset_states,
            "diagnostic_seed": args.diagnostic_seed,
            "runs": initialization_rows,
            "pairwise_comparisons": pair_rows,
        },
    )
    print(json.dumps({"output_dir": str(output_dir), "runs": len(diagnostics)}, indent=2))


if __name__ == "__main__":
    main()
