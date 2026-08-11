from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from agents import SACAgent
from methods import get_method
from training.implicit_policy_prior import initialize_actor_head_from_source
from training.llm_policy_prior_controller import (
    LLMPolicyPriorDecision,
    build_policy_prior_prompt,
    call_policy_prior_controller,
    resolve_prior_exploration,
)


class _FakeResponse:
    def __init__(self, decision: dict[str, object]) -> None:
        self.payload = {
            "output": [{"content": [{"type": "output_text", "text": json.dumps(decision)}]}]
        }

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


class ImplicitPolicyPriorTests(unittest.TestCase):
    def make_agent(self) -> SACAgent:
        return SACAgent(
            observation_dim=9,
            action_dim=2,
            action_low=np.full(2, -1.0, dtype=np.float32),
            action_high=np.full(2, 1.0, dtype=np.float32),
            num_tasks=3,
            task_id_dim=3,
            hide_task_id=True,
            device="cpu",
        )

    def test_initialization_changes_only_target_actor_head(self) -> None:
        agent = self.make_agent()
        observations = np.random.default_rng(3).normal(size=(32, 9)).astype(np.float32)
        observations[:, -3:] = (0.0, 1.0, 0.0)
        before = {name: value.detach().clone() for name, value in agent.state_dict().items()}

        result = initialize_actor_head_from_source(
            agent,
            observations=observations,
            source_task_index=0,
            target_task_index=1,
            updates=20,
            learning_rate=1e-2,
        )

        self.assertLess(result.final_kl, result.initial_kl)
        after = agent.state_dict()
        changed = {name for name in before if not torch.equal(before[name], after[name])}
        self.assertEqual(changed, {
            "actor.mean_head.weight",
            "actor.mean_head.bias",
            "actor.log_std_head.weight",
            "actor.log_std_head.bias",
        })
        for name in changed:
            old = before[name]
            new = after[name]
            self.assertTrue(torch.equal(old[0::3], new[0::3]))
            self.assertFalse(torch.equal(old[1::3], new[1::3]))
            self.assertTrue(torch.equal(old[2::3], new[2::3]))

    def test_method_explores_with_initialized_current_head(self) -> None:
        method = get_method("success_replay_best_llm_prior_pcgrad")
        strategy, heads = method.exploration_config(task_index=2, strategy="current")
        self.assertEqual(strategy, "current")
        self.assertEqual(heads, 3)

    def test_prior_exploration_uses_random_actions_when_no_source_is_selected(self) -> None:
        decision = LLMPolicyPriorDecision(None, 0.9, "No transfer.", "llm", "{}")
        strategy, heads, label = resolve_prior_exploration(task_index=4, decision=decision)
        self.assertIsNone(strategy)
        self.assertIsNone(heads)
        self.assertEqual(label, "random")

    def test_prior_exploration_uses_only_initialized_current_head(self) -> None:
        decision = LLMPolicyPriorDecision("push-back-v3", 0.9, "Transfer.", "llm", "{}")
        strategy, heads, label = resolve_prior_exploration(task_index=4, decision=decision)
        self.assertEqual(strategy, "current")
        self.assertEqual(heads, 5)
        self.assertEqual(label, "current_initialized_prior")

    def test_controller_rejects_future_or_unknown_task(self) -> None:
        with patch(
            "training.llm_policy_prior_controller.urllib.request.urlopen",
            return_value=_FakeResponse(
                {"source_task_name": "stick-pull-v3", "confidence": 0.9, "reason": "bad"}
            ),
        ):
            with self.assertRaisesRegex(ValueError, "unavailable or future"):
                call_policy_prior_controller(
                    api_key="test",
                    model="test",
                    prompt="test",
                    max_output_tokens=512,
                    previous_task_names=["hammer-v3"],
                    minimum_confidence=0.7,
                )

    def test_low_confidence_transfer_falls_back_to_none(self) -> None:
        with patch(
            "training.llm_policy_prior_controller.urllib.request.urlopen",
            return_value=_FakeResponse(
                {"source_task_name": "hammer-v3", "confidence": 0.5, "reason": "weak"}
            ),
        ):
            decision = call_policy_prior_controller(
                api_key="test",
                model="test",
                prompt="test",
                max_output_tokens=512,
                previous_task_names=["hammer-v3"],
                minimum_confidence=0.7,
            )
        self.assertIsNone(decision.source_task_name)
        self.assertEqual(decision.source, "confidence_fallback")
        self.assertEqual(decision.confidence, 0.5)
        self.assertTrue(decision.raw_response_text)

    def test_prompt_contains_only_visible_previous_tasks(self) -> None:
        source_dir = Path("prompts/llm_policy_prior")
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            (target / "head_selector.txt").write_text(
                (source_dir / "head_selector.txt").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            (target / "cw10_v3_task_descriptions.json").write_text(
                (source_dir / "cw10_v3_task_descriptions.json").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            prompt = build_policy_prior_prompt(
                current_task_name="push-wall-v3",
                previous_task_names=["hammer-v3"],
                prompt_dir=target,
            )
        self.assertIn("hammer-v3", prompt)
        self.assertIn("push-wall-v3", prompt)
        self.assertNotIn("stick-pull-v3", prompt)


if __name__ == "__main__":
    unittest.main()
