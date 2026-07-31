import unittest

from envs import get_cw10_tasks, make_cw_env
from training.semantic_segments import (
    FeatureSnapshot,
    extract_features,
    segment_label,
    segment_label_for_scheme,
    TaskAwareSegmenter,
    task_aware_segment_label,
    TaskAwareV3Segmenter,
)


class TestSemanticSegments(unittest.TestCase):
    def test_task_aware_segmenter_detects_planar_manipulation(self) -> None:
        features = FeatureSnapshot(
            task_name="push-back-v3",
            tcp_to_obj=0.04,
            obj_to_target=0.20,
            progress_ratio=0.02,
            object_displacement=0.02,
            success=0.0,
            object_motion=0.002,
        )
        self.assertEqual(task_aware_segment_label(features), "manipulation")

    def test_task_aware_segmenter_debounces_one_step_flicker(self) -> None:
        segmenter = TaskAwareSegmenter(debounce_steps=3)
        approach = FeatureSnapshot(
            task_name="window-close-v3",
            tcp_to_obj=0.20,
            obj_to_target=0.20,
            progress_ratio=0.0,
            object_displacement=0.0,
            success=0.0,
        )
        contact = FeatureSnapshot(
            task_name="window-close-v3",
            tcp_to_obj=0.03,
            obj_to_target=0.20,
            progress_ratio=0.0,
            object_displacement=0.0,
            success=0.0,
        )
        self.assertEqual(segmenter.label(approach), "approach")
        self.assertEqual(segmenter.label(contact), "approach")
        self.assertEqual(segmenter.label(approach), "approach")

    def test_v3_uses_environment_events_for_tool_task(self) -> None:
        segmenter = TaskAwareV3Segmenter()
        grasp = FeatureSnapshot(
            task_name="stick-pull-v3",
            tcp_to_obj=0.02,
            obj_to_target=0.30,
            progress_ratio=0.05,
            object_displacement=0.0,
            success=0.0,
            near_object=1.0,
            grasp_success=1.0,
            grasp_reward=1.0,
        )
        label = segmenter.label(grasp)
        self.assertEqual(label.general, "contact_or_alignment")
        self.assertEqual(label.task_specific, "grasp_or_engage_object")

    def test_v3_hammer_separates_geometric_contact_from_manipulation(self) -> None:
        segmenter = TaskAwareV3Segmenter()
        contact = FeatureSnapshot(
            task_name="hammer-v3",
            tcp_to_obj=0.096,
            obj_to_target=0.30,
            progress_ratio=0.02,
            object_displacement=0.03,
            object_motion=0.002,
            success=0.0,
        )
        manipulation = FeatureSnapshot(
            task_name="hammer-v3",
            tcp_to_obj=0.06,
            obj_to_target=0.20,
            progress_ratio=0.10,
            object_displacement=0.08,
            object_motion=0.003,
            success=0.0,
        )
        contact_label = segmenter.label(contact)
        self.assertEqual(contact_label.general, "contact_or_alignment")
        self.assertEqual(contact_label.task_specific, "establish_hammer_contact")
        self.assertEqual(segmenter.label(manipulation).general, "manipulation")

    def test_v3_peg_ignores_contact_jitter_before_extraction(self) -> None:
        jitter = FeatureSnapshot(
            task_name="peg-unplug-side-v3",
            tcp_to_obj=0.04,
            obj_to_target=0.30,
            progress_ratio=0.01,
            object_displacement=0.002,
            object_motion=0.001,
            success=0.0,
            near_object=1.0,
        )
        extracted = FeatureSnapshot(
            task_name="peg-unplug-side-v3",
            tcp_to_obj=0.04,
            obj_to_target=0.20,
            progress_ratio=0.10,
            object_displacement=0.01,
            object_motion=0.001,
            success=0.0,
            near_object=1.0,
        )
        segmenter = TaskAwareV3Segmenter()
        self.assertEqual(segmenter.label(jitter).general, "contact_or_alignment")
        self.assertEqual(segmenter.label(extracted).general, "manipulation")

    def test_v3_advances_immediately_but_debounces_regression(self) -> None:
        segmenter = TaskAwareV3Segmenter(regression_debounce_steps=3)
        approach = FeatureSnapshot(
            task_name="push-v3",
            tcp_to_obj=0.2,
            obj_to_target=0.3,
            progress_ratio=0.0,
            object_displacement=0.0,
            success=0.0,
        )
        manipulation = FeatureSnapshot(
            task_name="push-v3",
            tcp_to_obj=0.03,
            obj_to_target=0.2,
            progress_ratio=0.2,
            object_displacement=0.03,
            success=0.0,
            near_object=1.0,
            object_motion=0.002,
        )
        self.assertEqual(segmenter.label(approach).general, "approach")
        self.assertEqual(segmenter.label(manipulation).general, "manipulation")
        self.assertEqual(segmenter.label(approach).general, "manipulation")

    def test_window_close_reset_is_not_finish_or_stabilize(self) -> None:
        env = make_cw_env(
            "window-close-v3",
            seed=0,
            max_episode_steps=200,
            append_task_id=False,
            env_version="v3",
            reward_function_version="cw10_v1",
        )
        try:
            env.reset(seed=0)
            features = extract_features(env, "window-close-v3", False)
            self.assertLess(features.progress_ratio, 0.85)
            self.assertNotEqual(segment_label(features), "finish_or_stabilize")
        finally:
            env.close()

    def test_cw10_reset_progress_is_not_saturated(self) -> None:
        for task_name in get_cw10_tasks("v3"):
            env = make_cw_env(
                task_name,
                seed=0,
                max_episode_steps=200,
                append_task_id=False,
                env_version="v3",
                reward_function_version="cw10_v1",
            )
            try:
                env.reset(seed=0)
                features = extract_features(env, task_name, False)
                self.assertLess(
                    features.progress_ratio,
                    0.85,
                    msg=f"{task_name} has saturated reset progress_ratio={features.progress_ratio}",
                )
            finally:
                env.close()

    def test_cw10_initial_features_refresh_after_every_reset(self) -> None:
        for task_name in get_cw10_tasks("v3"):
            env = make_cw_env(
                task_name,
                seed=0,
                max_episode_steps=200,
                append_task_id=False,
                env_version="v3",
                reward_function_version="cw10_v1",
            )
            try:
                for episode_seed in (0, 1, 2):
                    env.reset(seed=episode_seed)
                    features = extract_features(env, task_name, False)
                    self.assertAlmostEqual(
                        features.progress_ratio,
                        0.0,
                        places=6,
                        msg=f"{task_name} retained progress after reset {episode_seed}",
                    )
                    self.assertAlmostEqual(
                        features.object_displacement,
                        0.0,
                        places=6,
                        msg=f"{task_name} retained displacement after reset {episode_seed}",
                    )
                    env.step(env.action_space.sample())
            finally:
                env.close()

    def test_feature_cache_refreshes_when_first_extraction_follows_step(self) -> None:
        env = make_cw_env(
            "faucet-close-v3",
            seed=0,
            max_episode_steps=200,
            append_task_id=False,
            env_version="v3",
            reward_function_version="cw10_v1",
        )
        try:
            env.reset(seed=0)
            extract_features(env, "faucet-close-v3", False)
            for _ in range(5):
                env.step(env.action_space.sample())
                extract_features(env, "faucet-close-v3", False)

            env.reset(seed=1)
            env.step(env.action_space.sample())
            features = extract_features(env, "faucet-close-v3", False)
            self.assertAlmostEqual(features.progress_ratio, 0.0, places=6)
            self.assertAlmostEqual(features.object_displacement, 0.0, places=6)
        finally:
            env.close()

    def test_stickpull_transfer_scheme_returns_expected_labels(self) -> None:
        finish = FeatureSnapshot(
            task_name="push-back-v3",
            tcp_to_obj=0.02,
            obj_to_target=0.01,
            progress_ratio=0.95,
            object_displacement=0.10,
            success=1.0,
        )
        pull = FeatureSnapshot(
            task_name="push-back-v3",
            tcp_to_obj=0.03,
            obj_to_target=0.10,
            progress_ratio=0.50,
            object_displacement=0.08,
            success=0.0,
        )
        align = FeatureSnapshot(
            task_name="faucet-close-v3",
            tcp_to_obj=0.03,
            obj_to_target=0.20,
            progress_ratio=0.15,
            object_displacement=0.03,
            success=0.0,
        )
        grasp = FeatureSnapshot(
            task_name="hammer-v3",
            tcp_to_obj=0.05,
            obj_to_target=0.40,
            progress_ratio=0.05,
            object_displacement=0.01,
            success=0.0,
        )
        approach = FeatureSnapshot(
            task_name="hammer-v3",
            tcp_to_obj=0.20,
            obj_to_target=0.50,
            progress_ratio=0.0,
            object_displacement=0.0,
            success=0.0,
        )

        self.assertEqual(
            segment_label_for_scheme(finish, scheme="stickpull_transfer"),
            "terminal_stabilize",
        )
        self.assertEqual(
            segment_label_for_scheme(pull, scheme="stickpull_transfer"),
            "pull_with_contact",
        )
        self.assertEqual(
            segment_label_for_scheme(align, scheme="stickpull_transfer"),
            "align_tool_to_handle",
        )
        self.assertEqual(
            segment_label_for_scheme(grasp, scheme="stickpull_transfer"),
            "grasp_or_lift_tool",
        )
        self.assertEqual(
            segment_label_for_scheme(approach, scheme="stickpull_transfer"),
            "approach_tool",
        )


if __name__ == "__main__":
    unittest.main()
