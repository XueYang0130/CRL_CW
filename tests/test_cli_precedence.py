import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch

from agents import FullBehaviorCloningSACAgent
from scripts.run_copy import parse_args
from methods import get_method


class TestCliPrecedence(unittest.TestCase):
    def test_success_replay_cagrad_preset(self) -> None:
        with patch.object(
            sys,
            "argv",
            [
                "run.py",
                "--mode",
                "continual",
                "--method",
                "success_replay_best_cagrad",
            ],
        ):
            args = parse_args()
        self.assertEqual(args.bc_gradient_strategy, "standard")
        self.assertEqual(args.bc_combination_strategy, "cagrad")
        self.assertEqual(args.bc_cagrad_alpha, 0.5)
        self.assertEqual(args.episodic_memory_per_task, 10_000)

    def test_success_replay_cagrad_rejects_combination_override(self) -> None:
        with patch.object(
            sys,
            "argv",
            [
                "run.py",
                "--mode",
                "continual",
                "--method",
                "success_replay_best_cagrad",
                "--bc-combination-strategy",
                "average",
            ],
        ), self.assertRaises(SystemExit):
            parse_args()

    def test_gradient_method_presets_select_expected_strategy(self) -> None:
        expected = {
            "full_bc": "standard",
            "full_bc_norm_balanced": "norm_balanced",
            "full_bc_pcgrad": "pcgrad_sac_priority",
        }
        for method, strategy in expected.items():
            with self.subTest(method=method), patch.object(
                sys,
                "argv",
                ["run.py", "--mode", "continual", "--method", method],
            ):
                args = parse_args()
            self.assertEqual(args.bc_gradient_strategy, strategy)
            self.assertTrue(args.gradient_diagnostics)
            self.assertTrue(get_method(method).complete_reference_memory)

            agent = get_method(method).build_agent(
                args=args,
                observation_dim=7,
                action_dim=2,
                action_low=np.full(2, -1.0, dtype=np.float32),
                action_high=np.full(2, 1.0, dtype=np.float32),
                total_tasks=3,
            )
            self.assertIsInstance(agent, FullBehaviorCloningSACAgent)
            self.assertEqual(agent.bc_gradient_strategy, strategy)
            self.assertEqual(agent.num_tasks, 3)
            self.assertEqual(agent.task_id_dim, 3)

    def test_gradient_method_rejects_mislabeled_strategy_override(self) -> None:
        invalid_combinations = (
            ("full_bc", "pcgrad_sac_priority"),
            ("clonex_sac", "norm_balanced"),
            ("full_bc_pcgrad", "standard"),
            ("full_bc_norm_balanced", "standard"),
            ("fine_tuning", "pcgrad_sac_priority"),
        )
        for method, strategy in invalid_combinations:
            with self.subTest(method=method, strategy=strategy), patch.object(
                sys,
                "argv",
                [
                    "run.py",
                    "--mode",
                    "continual",
                    "--method",
                    method,
                    "--bc-gradient-strategy",
                    strategy,
                ],
            ), self.assertRaises(SystemExit):
                parse_args()

    def test_reference_collection_configuration_fails_before_training(self) -> None:
        invalid_arguments = (
            ["--best-return-eval-episodes", "0"],
            ["--full-bc-reference-episodes", "0"],
            [
                "--full-bc-reference-episodes",
                "20",
                "--full-bc-reference-max-attempts",
                "19",
            ],
        )
        for arguments in invalid_arguments:
            with self.subTest(arguments=arguments), patch.object(
                sys,
                "argv",
                [
                    "run.py",
                    "--mode",
                    "continual",
                    "--method",
                    "full_bc",
                    *arguments,
                ],
            ), self.assertRaises(SystemExit):
                parse_args()

    def test_rejects_invalid_gradient_control_values(self) -> None:
        invalid_arguments = (
            ["--gradient-clip-norm", "-0.1"],
            ["--gradient-clip-norm", "nan"],
            ["--actor-cloning-coefficient", "nan"],
            ["--bc-max-norm-ratio", "inf"],
            ["--bc-cagrad-alpha", "1.0"],
            ["--background-segment-ratio", "nan"],
            ["--task-specific-segment-ratio", "inf"],
        )
        for extra_arguments in invalid_arguments:
            with self.subTest(arguments=extra_arguments), patch.object(
                sys,
                "argv",
                ["run.py", "--mode", "continual", *extra_arguments],
            ), self.assertRaises(SystemExit):
                parse_args()

    def test_rejects_invalid_training_values_before_running(self) -> None:
        invalid_arguments = (
            ["--target-output-std", "0"],
            ["--learning-rate", "nan"],
            ["--replay-size", "0"],
            ["--batch-size", "0"],
            ["--update-every", "0"],
            ["--llm-controller-max-output-tokens", "0"],
        )
        for extra_arguments in invalid_arguments:
            with self.subTest(arguments=extra_arguments), patch.object(
                sys,
                "argv",
                ["run.py", "--mode", "continual", *extra_arguments],
            ), self.assertRaises(SystemExit):
                parse_args()

    def test_rejects_invalid_jsrl_schedule_before_task_one(self) -> None:
        invalid_arguments = (
            ["--jsrl-initial-guide-steps", "200"],
            ["--jsrl-curriculum-stages", "1"],
            ["--jsrl-evaluation-interval", "30000"],
            ["--jsrl-stage-tolerance", "nan"],
            ["--jsrl-moving-average-window", "6"],
        )
        for extra_arguments in invalid_arguments:
            with self.subTest(arguments=extra_arguments), patch.object(
                sys,
                "argv",
                [
                    "run.py",
                    "--mode",
                    "continual",
                    "--method",
                    "jsrl_continual",
                    *extra_arguments,
                ],
            ), self.assertRaises(SystemExit):
                parse_args()

    def test_multi_head_method_rejects_persistent_online_replay(self) -> None:
        with patch.object(
            sys,
            "argv",
            [
                "run.py",
                "--mode",
                "continual",
                "--method",
                "full_bc",
                "--no-reset-buffer-on-task-change",
            ],
        ), self.assertRaises(SystemExit):
            parse_args()

    def test_rejects_task_count_option_for_the_wrong_mode(self) -> None:
        invalid_argv = (
            ["run.py", "--mode", "continual", "--num-tasks", "2"],
            ["run.py", "--mode", "single-batch", "--sequence-task-count", "2"],
            ["run.py", "--mode", "single", "--sequence-task-count", "2"],
        )
        for argv in invalid_argv:
            with self.subTest(argv=argv), patch.object(
                sys, "argv", argv
            ), self.assertRaises(SystemExit):
                parse_args()

    def test_gradient_method_presets_execute_bc_on_old_task_memory(self) -> None:
        expected = {
            "full_bc": "standard",
            "full_bc_norm_balanced": "norm_balanced",
            "full_bc_pcgrad": "pcgrad_sac_priority",
        }
        for method, strategy in expected.items():
            argv = [
                "run.py",
                "--mode",
                "continual",
                "--method",
                method,
                "--gradient-diagnostics-interval",
                "1",
            ]
            with self.subTest(method=method), patch.object(sys, "argv", argv):
                args = parse_args()
                agent = get_method(method).build_agent(
                    args=args,
                    observation_dim=7,
                    action_dim=2,
                    action_low=np.full(2, -1.0, dtype=np.float32),
                    action_high=np.full(2, 1.0, dtype=np.float32),
                    total_tasks=3,
                )

                old_observations = torch.randn(8, 7)
                old_observations[:, -3:] = 0.0
                old_observations[:, 4] = 1.0
                agent.add_reference_memory(observations=old_observations)

                observations = torch.randn(8, 7)
                observations[:, -3:] = 0.0
                observations[:, 5] = 1.0
                next_observations = observations.clone()
                agent.update_batch(
                    observations=observations,
                    actions=torch.rand(8, 2) * 2.0 - 1.0,
                    rewards=torch.randn(8, 1),
                    next_observations=next_observations,
                    dones=torch.zeros(8, 1),
                )

                rows = agent.drain_gradient_diagnostics()["windows"]
                self.assertEqual(len(rows), 2)
                self.assertEqual(rows[0]["gradient_strategy"], strategy)
                self.assertEqual(
                    agent.gradient_diagnostics_summary()["bc_updates_seen"],
                    1,
                )

    def test_llm_hybrid_uses_fine_grained_task_specific_store_without_manifest(self) -> None:
        argv = [
            "run.py",
            "--mode",
            "continual",
            "--method",
            "semantic_hybrid_bc",
            "--segment-selection-mode",
            "llm_online",
            "--task-specific-segment-ratio",
            "0.2",
        ]
        with patch.object(sys, "argv", argv):
            args = parse_args()
        self.assertIsNone(args.task_specific_segment_manifest)
        self.assertEqual(args.task_specific_segment_ratio, 0.2)

    def test_rejects_semantic_controller_for_unsupported_method(self) -> None:
        for method in ("fine_tuning", "full_bc", "stage_aware_semantic_bc"):
            with self.subTest(method=method), patch.object(
                sys,
                "argv",
                [
                    "run.py",
                    "--mode",
                    "continual",
                    "--method",
                    method,
                    "--segment-selection-mode",
                    "llm_online",
                ],
            ), self.assertRaises(SystemExit):
                parse_args()

    def test_semantic_methods_default_to_validated_v3_segmenter(self) -> None:
        for method in ("semantic_local_bc", "semantic_hybrid_bc"):
            argv = ["run.py", "--mode", "continual", "--method", method]
            if method == "semantic_hybrid_bc":
                argv.extend(
                    [
                        "--task-specific-segment-manifest",
                        "configs/segment_selection/cw10_v3_task_specific_example.json",
                    ]
                )
            with self.subTest(method=method), patch.object(sys, "argv", argv):
                args = parse_args()
            self.assertEqual(args.semantic_segment_scheme, "task_aware_v3")

    def test_explicit_segment_scheme_overrides_method_preset(self) -> None:
        argv = [
            "run.py",
            "--mode",
            "continual",
            "--method",
            "semantic_local_bc",
            "--semantic-segment-scheme",
            "task_aware_v2",
        ]
        with patch.object(sys, "argv", argv):
            args = parse_args()
        self.assertEqual(args.semantic_segment_scheme, "task_aware_v2")

    def test_explicit_values_override_method_presets(self) -> None:
        argv = [
            "run.py",
            "--mode",
            "continual",
            "--method",
            "semantic_hybrid_bc",
            "--exploration-strategy",
            "random",
            "--background-segment-ratio",
            "0",
            "--task-specific-segment-manifest",
            "configs/segment_selection/cw10_v3_task_specific_example.json",
        ]
        with patch.object(sys, "argv", argv):
            args = parse_args()
        self.assertEqual(args.exploration_strategy, "random")
        self.assertEqual(args.background_segment_ratio, 0.0)

    def test_semantic_segment_weights_are_parsed_explicitly(self) -> None:
        argv = [
            "run.py",
            "--mode",
            "continual",
            "--method",
            "semantic_local_bc",
            "--semantic-segments",
            "contact_or_alignment",
            "manipulation",
            "finish_or_stabilize",
            "--semantic-segment-weights",
            "contact_or_alignment=1",
            "manipulation=1",
            "finish_or_stabilize=0.2",
        ]
        with patch.object(sys, "argv", argv):
            args = parse_args()
        self.assertAlmostEqual(sum(args.semantic_segment_weights.values()), 1.0)
        self.assertAlmostEqual(
            args.semantic_segment_weights["contact_or_alignment"], 1.0 / 2.2
        )
        self.assertAlmostEqual(
            args.semantic_segment_weights["finish_or_stabilize"], 0.2 / 2.2
        )


if __name__ == "__main__":
    unittest.main()
