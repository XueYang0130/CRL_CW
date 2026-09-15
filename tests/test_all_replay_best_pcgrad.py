import unittest

from methods import get_method


class AllReplayBestPCGradMethodTests(unittest.TestCase):
    def test_is_an_isolated_state_selection_ablation(self) -> None:
        method = get_method("all_replay_best_pcgrad")

        self.assertIsNone(method.success_replay_teacher)
        self.assertEqual(method.all_replay_teacher, "best")
        self.assertEqual(method.defaults["bc_gradient_strategy"], "pcgrad_sac_priority")
        self.assertEqual(method.defaults["bc_combination_strategy"], "average")
        self.assertEqual(method.defaults["episodic_memory_per_task"], 10_000)


if __name__ == "__main__":
    unittest.main()
