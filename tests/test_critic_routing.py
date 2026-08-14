import json
import tempfile
import unittest
from pathlib import Path

from training.critic_routing import load_critic_route_manifest


class CriticRoutingTests(unittest.TestCase):
    def write_manifest(self, payload: dict) -> Path:
        directory = Path(tempfile.mkdtemp())
        path = directory / "routes.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_loads_complete_frozen_routes(self) -> None:
        path = self.write_manifest(
            {
                "source": "test",
                "routes": [
                    {"task_index": 0, "task_name": "a", "route": "transfer"},
                    {"task_index": 1, "task_name": "b", "route": "reset"},
                ],
            }
        )
        routes = load_critic_route_manifest(path, tasks=["a", "b"])
        self.assertEqual(routes[1].route, "reset")
        self.assertEqual(routes[1].source, "test")

    def test_rejects_missing_route(self) -> None:
        path = self.write_manifest(
            {
                "routes": [
                    {"task_index": 0, "task_name": "a", "route": "transfer"}
                ]
            }
        )
        with self.assertRaisesRegex(ValueError, "missing task indices"):
            load_critic_route_manifest(path, tasks=["a", "b"])

    def test_rejects_reset_on_first_task(self) -> None:
        path = self.write_manifest(
            {
                "routes": [
                    {"task_index": 0, "task_name": "a", "route": "reset"}
                ]
            }
        )
        with self.assertRaisesRegex(ValueError, "first task"):
            load_critic_route_manifest(path, tasks=["a"])

    def test_full_manifest_can_drive_sequence_prefix(self) -> None:
        path = self.write_manifest(
            {
                "routes": [
                    {"task_index": 0, "task_name": "a", "route": "transfer"},
                    {"task_index": 1, "task_name": "future", "route": "reset"},
                ]
            }
        )
        routes = load_critic_route_manifest(path, tasks=["a"])
        self.assertEqual(set(routes), {0})
