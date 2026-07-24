"""Tests for the fixed CW10 task sequence."""

import sys
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from envs import CW10_TASKS, get_cw10_tasks


EXPECTED_CW10_TASKS = (
    "hammer-v3",
    "push-wall-v3",
    "faucet-close-v3",
    "push-back-v3",
    "stick-pull-v3",
    "handle-press-side-v3",
    "push-v3",
    "shelf-place-v3",
    "window-close-v3",
    "peg-unplug-side-v3",
)


class TestCW10Tasks(unittest.TestCase):
    def test_cw10_contains_ten_tasks(self) -> None:
        self.assertEqual(len(CW10_TASKS), 10)

    def test_cw10_order_matches_reference(self) -> None:
        self.assertEqual(CW10_TASKS, EXPECTED_CW10_TASKS)

    def test_get_cw10_tasks_returns_copy(self) -> None:
        first_result = get_cw10_tasks()
        first_result.append("fake-task")

        second_result = get_cw10_tasks()

        self.assertEqual(len(second_result), 10)
        self.assertNotIn("fake-task", second_result)

    def test_task_names_are_unique(self) -> None:
        self.assertEqual(len(CW10_TASKS), len(set(CW10_TASKS)))


if __name__ == "__main__":
    unittest.main()
