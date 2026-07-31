from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


SEGMENT_ORDER: tuple[str, ...] = (
    "approach",
    "contact_or_alignment",
    "manipulation",
    "finish_or_stabilize",
)

STICKPULL_TRANSFER_SEGMENT_ORDER: tuple[str, ...] = (
    "approach_tool",
    "grasp_or_lift_tool",
    "align_tool_to_handle",
    "pull_with_contact",
    "terminal_stabilize",
)


@dataclass(frozen=True)
class FeatureSnapshot:
    task_name: str
    tcp_to_obj: float
    obj_to_target: float
    progress_ratio: float
    object_displacement: float
    success: float
    object_motion: float = 0.0
    gripper_open: float = 0.0
    action_motion: float = 0.0
    gripper_action: float = 0.0
    near_object: float = 0.0
    grasp_success: float = 0.0
    grasp_reward: float = 0.0
    in_place_reward: float = 0.0


@dataclass(frozen=True)
class SemanticLabel:
    general: str
    task_specific: str


def _progress_ratio(current: float, initial: float) -> float:
    if initial <= 1e-8:
        return 1.0
    return float(np.clip(1.0 - current / initial, 0.0, 1.0))


def _cache_episode_initial_object_position(
    env: Any,
    task_name: str,
    object_position: np.ndarray,
) -> np.ndarray:
    base_env = env.unwrapped
    cache_name = f"_semantic_initial_object_position__{task_name.replace('-', '_')}"
    first_obs_name = f"_semantic_first_observation__{task_name.replace('-', '_')}"
    elapsed_name = f"_semantic_last_elapsed_steps__{task_name.replace('-', '_')}"
    try:
        elapsed_steps = env.get_wrapper_attr("_elapsed_steps")
    except AttributeError:
        elapsed_steps = getattr(env, "_elapsed_steps", None)
    previous_elapsed = getattr(base_env, elapsed_name, None)
    needs_refresh = (
        not hasattr(base_env, cache_name)
        or elapsed_steps == 0
        or (
            elapsed_steps is not None
            and previous_elapsed is not None
            and elapsed_steps < previous_elapsed
        )
    )

    if needs_refresh:
        if task_name == "window-close-v3" and hasattr(base_env, "window_handle_pos_init"):
            initial = np.asarray(base_env.window_handle_pos_init, dtype=np.float64)[:3]
        elif task_name == "handle-press-side-v3" and hasattr(base_env, "_handle_init_pos"):
            initial = np.asarray(base_env._handle_init_pos, dtype=np.float64)[:3]
        elif task_name == "hammer-v3" and hasattr(base_env, "hammer_init_pos"):
            initial = np.asarray(base_env.hammer_init_pos, dtype=np.float64)[:3]
        elif task_name in {"push-wall-v3", "push-back-v3", "push-v3", "shelf-place-v3", "peg-unplug-side-v3"} and hasattr(base_env, "obj_init_pos"):
            initial = np.asarray(base_env.obj_init_pos, dtype=np.float64)[:3]
        else:
            # Some Meta-World tasks expose geometry attributes that do not
            # correspond to the tracked object position. For those tasks, the
            # safest baseline is the actual object position observed at reset.
            initial = object_position.copy()
        setattr(base_env, cache_name, initial)
        setattr(base_env, first_obs_name, True)
    if elapsed_steps is not None:
        setattr(base_env, elapsed_name, int(elapsed_steps))

    cached = np.asarray(getattr(base_env, cache_name), dtype=np.float64)
    return cached[: object_position.shape[0]]


def _consume_first_observation_flag(env: Any, task_name: str) -> bool:
    base_env = env.unwrapped
    name = f"_semantic_first_observation__{task_name.replace('-', '_')}"
    is_first = bool(getattr(base_env, name, False))
    if is_first:
        setattr(base_env, name, False)
    return is_first


def _distance_to_target(
    task_name: str,
    object_position: np.ndarray,
    target_position: np.ndarray,
) -> float:
    if task_name == "window-close-v3":
        return float(abs(object_position[0] - target_position[0]))
    if task_name == "handle-press-side-v3":
        return float(abs(object_position[2] - target_position[2]))
    if task_name in {"push-wall-v3", "push-back-v3", "push-v3"}:
        return float(np.linalg.norm(object_position[:2] - target_position[:2]))
    if task_name == "peg-unplug-side-v3":
        return float(abs(object_position[0] - target_position[0]))
    return float(np.linalg.norm(object_position - target_position))


def _object_displacement(
    task_name: str,
    object_position: np.ndarray,
    initial_object_position: np.ndarray,
) -> float:
    if task_name == "window-close-v3":
        return float(abs(object_position[0] - initial_object_position[0]))
    if task_name == "handle-press-side-v3":
        return float(abs(object_position[2] - initial_object_position[2]))
    if task_name in {"push-wall-v3", "push-back-v3", "push-v3"}:
        return float(np.linalg.norm(object_position[:2] - initial_object_position[:2]))
    if task_name == "peg-unplug-side-v3":
        return float(abs(object_position[0] - initial_object_position[0]))
    return float(np.linalg.norm(object_position - initial_object_position))


def extract_features(
    env: Any,
    task_name: str,
    success_so_far: bool,
    *,
    previous_object_position: np.ndarray | None = None,
    action: np.ndarray | None = None,
    info: dict[str, Any] | None = None,
) -> FeatureSnapshot:
    base_env = env.unwrapped
    tcp = np.asarray(base_env.tcp_center, dtype=np.float64)
    obj = np.asarray(base_env._get_pos_objects(), dtype=np.float64).reshape(-1)[:3]
    target = np.asarray(base_env._target_pos, dtype=np.float64)
    target = target.reshape(-1)[: obj.shape[0]]
    initial_obj = _cache_episode_initial_object_position(env, task_name, obj)

    tcp_to_obj = float(np.linalg.norm(obj - tcp))
    obj_to_target = _distance_to_target(task_name, obj, target)
    initial_obj_to_target = _distance_to_target(task_name, initial_obj, target)
    object_displacement = _object_displacement(task_name, obj, initial_obj)
    object_motion = (
        0.0
        if previous_object_position is None
        else float(np.linalg.norm(obj - np.asarray(previous_object_position)[: obj.shape[0]]))
    )
    observation = np.asarray(base_env._get_obs(), dtype=np.float64).reshape(-1)
    gripper_open = float(observation[3]) if observation.size > 3 else 0.0
    action_array = (
        np.zeros(4, dtype=np.float64)
        if action is None
        else np.asarray(action, dtype=np.float64).reshape(-1)
    )
    action_motion = float(np.linalg.norm(action_array[:3])) if action_array.size >= 3 else 0.0
    gripper_action = float(action_array[3]) if action_array.size >= 4 else 0.0
    info = {} if info is None else info
    progress_ratio = _progress_ratio(obj_to_target, initial_obj_to_target)
    if _consume_first_observation_flag(env, task_name):
        object_displacement = 0.0
        progress_ratio = 0.0

    return FeatureSnapshot(
        task_name=task_name,
        tcp_to_obj=tcp_to_obj,
        obj_to_target=obj_to_target,
        progress_ratio=progress_ratio,
        object_displacement=object_displacement,
        success=float(success_so_far),
        object_motion=object_motion,
        gripper_open=gripper_open,
        action_motion=action_motion,
        gripper_action=gripper_action,
        near_object=float(info.get("near_object", tcp_to_obj <= 0.05)),
        grasp_success=float(info.get("grasp_success", 0.0)),
        grasp_reward=float(info.get("grasp_reward", 0.0)),
        in_place_reward=float(info.get("in_place_reward", progress_ratio)),
    )


def segment_label(features: FeatureSnapshot) -> str:
    if features.success > 0.0 or features.progress_ratio >= 0.85:
        return "finish_or_stabilize"
    if features.progress_ratio >= 0.20 or features.object_displacement >= 0.03:
        return "manipulation"
    if features.tcp_to_obj <= 0.05:
        return "contact_or_alignment"
    return "approach"


def task_aware_segment_label(features: FeatureSnapshot) -> str:
    """Return a task-family-aware four-stage semantic label."""
    if features.success > 0.0 or features.progress_ratio >= 0.88:
        return "finish_or_stabilize"

    task = features.task_name
    tool_or_lift = task in {"hammer-v3", "stick-pull-v3", "shelf-place-v3"}
    planar_push = task in {"push-wall-v3", "push-back-v3", "push-v3"}
    articulated = task in {
        "faucet-close-v3",
        "handle-press-side-v3",
        "window-close-v3",
        "peg-unplug-side-v3",
    }

    if tool_or_lift:
        manipulating = (
            features.object_displacement >= 0.018
            or features.object_motion >= 0.0015
            or features.progress_ratio >= 0.12
        )
        contact_distance = 0.065
    elif planar_push:
        manipulating = (
            features.object_displacement >= 0.012
            or features.object_motion >= 0.001
            or features.progress_ratio >= 0.08
        )
        contact_distance = 0.07
    elif articulated:
        manipulating = (
            features.object_displacement >= 0.008
            or features.object_motion >= 0.0008
            or features.progress_ratio >= 0.06
        )
        contact_distance = 0.06
    else:
        manipulating = (
            features.object_displacement >= 0.02
            or features.object_motion >= 0.001
            or features.progress_ratio >= 0.10
        )
        contact_distance = 0.06

    if manipulating:
        return "manipulation"
    if features.tcp_to_obj <= contact_distance:
        return "contact_or_alignment"
    return "approach"


class TaskAwareSegmenter:
    """Stateful task-aware segmenter with short temporal debounce."""

    def __init__(self, *, debounce_steps: int = 3) -> None:
        if debounce_steps <= 0:
            raise ValueError("debounce_steps must be positive.")
        self.debounce_steps = int(debounce_steps)
        self._label: str | None = None
        self._candidate: str | None = None
        self._candidate_count = 0

    def reset(self) -> None:
        self._label = None
        self._candidate = None
        self._candidate_count = 0

    def label(self, features: FeatureSnapshot) -> str:
        candidate = task_aware_segment_label(features)
        if self._label is None:
            self._label = candidate
            return candidate
        if candidate == "finish_or_stabilize" and features.success > 0.0:
            self._label = candidate
            self._candidate = None
            self._candidate_count = 0
            return candidate
        if candidate == self._label:
            self._candidate = None
            self._candidate_count = 0
            return self._label
        if candidate != self._candidate:
            self._candidate = candidate
            self._candidate_count = 1
        else:
            self._candidate_count += 1
        if self._candidate_count >= self.debounce_steps:
            self._label = candidate
            self._candidate = None
            self._candidate_count = 0
        return self._label


def task_aware_v3_label(features: FeatureSnapshot) -> SemanticLabel:
    """Map Meta-World task events to general and task-specific semantics."""
    if features.success > 0.0:
        return SemanticLabel("finish_or_stabilize", "complete_and_stabilize")

    task = features.task_name
    contact_distance = 0.10 if task == "hammer-v3" else 0.065
    geometric_near = features.tcp_to_obj <= contact_distance
    near = geometric_near or (
        features.near_object > 0.0 and features.tcp_to_obj <= 0.10
    )
    grasped = features.grasp_success > 0.0 or features.grasp_reward >= 0.5
    aligned = features.in_place_reward >= 0.65 or features.progress_ratio >= 0.65
    moving = features.object_motion >= 8e-4 or features.object_displacement >= 0.008

    if not near and features.object_displacement < 0.01:
        return SemanticLabel("approach", "approach_object")

    if task in {"hammer-v3", "stick-pull-v3", "shelf-place-v3", "peg-unplug-side-v3"}:
        if not near and not grasped:
            return SemanticLabel("approach", "approach_object")
        if (
            task == "hammer-v3"
            and near
            and features.progress_ratio < 0.05
            and features.object_displacement < 0.05
        ):
            return SemanticLabel("contact_or_alignment", "establish_hammer_contact")
        task_moving = moving
        if task == "peg-unplug-side-v3":
            # Contact jitter can exceed the generic one-step motion threshold
            # before the peg has meaningfully left the socket.
            task_moving = (
                features.object_displacement >= 0.008
                or features.progress_ratio >= 0.08
            )
        if aligned and task_moving:
            specific = {
                "hammer-v3": "align_hammer_to_nail",
                "stick-pull-v3": "align_tool_to_handle",
                "shelf-place-v3": "align_object_to_shelf",
                "peg-unplug-side-v3": "extract_peg_from_socket",
            }[task]
            return SemanticLabel("manipulation", specific)
        if grasped and task_moving:
            specific = {
                "hammer-v3": "transport_hammer",
                "stick-pull-v3": "lift_and_transport_tool",
                "shelf-place-v3": "lift_and_transport_object",
                "peg-unplug-side-v3": "grasp_and_pull_peg",
            }[task]
            return SemanticLabel("manipulation", specific)
        if task_moving:
            return SemanticLabel("manipulation", "move_engaged_object")
        if grasped:
            return SemanticLabel("contact_or_alignment", "grasp_or_engage_object")
        return SemanticLabel("contact_or_alignment", "establish_contact")

    if task in {"push-wall-v3", "push-back-v3", "push-v3"}:
        if moving:
            return SemanticLabel("manipulation", "push_object_toward_target")
        return SemanticLabel("contact_or_alignment", "align_tcp_for_push")

    if task == "faucet-close-v3":
        if moving:
            return SemanticLabel("manipulation", "rotate_faucet_closed")
        return SemanticLabel("contact_or_alignment", "engage_faucet_handle")
    if task == "handle-press-side-v3":
        if moving:
            return SemanticLabel("manipulation", "press_handle_down")
        return SemanticLabel("contact_or_alignment", "align_to_side_handle")
    if task == "window-close-v3":
        if moving:
            return SemanticLabel("manipulation", "slide_window_closed")
        return SemanticLabel("contact_or_alignment", "engage_window_handle")

    if aligned or moving:
        return SemanticLabel("manipulation", "move_object_toward_target")
    return SemanticLabel("contact_or_alignment", "establish_contact")


class TaskAwareV3Segmenter:
    """Event-based segmenter with asymmetric temporal stabilization."""

    def __init__(self, *, regression_debounce_steps: int = 3) -> None:
        if regression_debounce_steps <= 0:
            raise ValueError("regression_debounce_steps must be positive.")
        self.regression_debounce_steps = int(regression_debounce_steps)
        self._label: SemanticLabel | None = None
        self._regression_candidate: SemanticLabel | None = None
        self._regression_count = 0

    def reset(self) -> None:
        self._label = None
        self._regression_candidate = None
        self._regression_count = 0

    def label(self, features: FeatureSnapshot) -> SemanticLabel:
        candidate = task_aware_v3_label(features)
        if self._label is None:
            self._label = candidate
            return candidate
        ranks = {label: index for index, label in enumerate(SEGMENT_ORDER)}
        current_rank = ranks[self._label.general]
        candidate_rank = ranks[candidate.general]
        if candidate_rank >= current_rank:
            self._label = candidate
            self._regression_candidate = None
            self._regression_count = 0
            return candidate
        if candidate == self._regression_candidate:
            self._regression_count += 1
        else:
            self._regression_candidate = candidate
            self._regression_count = 1
        if self._regression_count >= self.regression_debounce_steps:
            self._label = candidate
            self._regression_candidate = None
            self._regression_count = 0
        return self._label


def stickpull_transfer_segment_label(features: FeatureSnapshot) -> str:
    # This scheme is not meant to describe the source task literally.
    # It describes which abstract transfer skill a source-state most closely
    # resembles for downstream stick-pull learning.
    if features.success > 0.0 or features.progress_ratio >= 0.90:
        return "terminal_stabilize"
    if (
        features.tcp_to_obj <= 0.06
        and (features.progress_ratio >= 0.35 or features.object_displacement >= 0.06)
    ):
        return "pull_with_contact"
    if (
        features.tcp_to_obj <= 0.05
        and (features.progress_ratio >= 0.12 or features.object_displacement >= 0.02)
    ):
        return "align_tool_to_handle"
    if features.tcp_to_obj <= 0.07:
        return "grasp_or_lift_tool"
    return "approach_tool"


def segment_label_for_scheme(
    features: FeatureSnapshot,
    *,
    scheme: str = "default",
) -> str:
    if scheme == "default":
        return segment_label(features)
    if scheme == "task_aware_v2":
        return task_aware_segment_label(features)
    if scheme == "stickpull_transfer":
        return stickpull_transfer_segment_label(features)
    raise ValueError(f"Unsupported segmentation scheme: {scheme}")
