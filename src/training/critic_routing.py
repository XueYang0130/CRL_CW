from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


VALID_CRITIC_ROUTES = frozenset({"transfer", "reset"})


@dataclass(frozen=True)
class CriticRoute:
    task_index: int
    task_name: str
    route: str
    reason: str
    source: str


def load_critic_route_manifest(
    path: str | Path,
    *,
    tasks: list[str],
) -> dict[int, CriticRoute]:
    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Critic route manifest does not exist: {manifest_path}"
        )
    payload: Any = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("routes"), list):
        raise ValueError("Critic route manifest must contain a 'routes' list.")

    routes: dict[int, CriticRoute] = {}
    for raw_route in payload["routes"]:
        if not isinstance(raw_route, dict):
            raise ValueError("Every critic route must be an object.")
        task_index = int(raw_route["task_index"])
        if task_index >= len(tasks):
            # A full-sequence manifest may be reused for a prefix smoke run.
            continue
        if task_index < 0:
            raise ValueError(f"Critic route task index is out of range: {task_index}.")
        if task_index in routes:
            raise ValueError(f"Duplicate critic route for task index {task_index}.")
        task_name = str(raw_route["task_name"])
        if task_name != tasks[task_index]:
            raise ValueError(
                "Critic route task name does not match the active sequence: "
                f"index={task_index}, manifest={task_name}, sequence={tasks[task_index]}."
            )
        route = str(raw_route["route"])
        if route not in VALID_CRITIC_ROUTES:
            raise ValueError(
                f"Unsupported critic route '{route}' for task {task_name}."
            )
        if task_index == 0 and route != "transfer":
            raise ValueError("The first task must use the transfer critic route.")
        routes[task_index] = CriticRoute(
            task_index=task_index,
            task_name=task_name,
            route=route,
            reason=str(raw_route.get("reason", "")),
            source=str(raw_route.get("source", payload.get("source", "manifest"))),
        )

    missing = sorted(set(range(len(tasks))) - set(routes))
    if missing:
        raise ValueError(f"Critic route manifest is missing task indices: {missing}.")
    return routes
