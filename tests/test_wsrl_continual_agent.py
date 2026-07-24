import numpy as np
import pytest
import torch

from agents import WSRLContinualAgent


def make_agent(*, backbone_source: str, head_source: str) -> WSRLContinualAgent:
    return WSRLContinualAgent(
        observation_dim=7,
        action_dim=2,
        action_low=np.full(2, -1.0, dtype=np.float32),
        action_high=np.full(2, 1.0, dtype=np.float32),
        num_tasks=3,
        task_id_dim=3,
        hide_task_id=True,
        backbone_source=backbone_source,
        head_source=head_source,
    )


def head_rows(layer: torch.nn.Linear, head_index: int, num_heads: int) -> torch.Tensor:
    return layer.weight[head_index::num_heads].detach().clone()


def test_current_head_copies_previous_task_head() -> None:
    agent = make_agent(backbone_source="current", head_source="current")
    source = head_rows(agent.actor.mean_head, 0, agent.num_tasks)
    agent.on_task_end(task_index=0, replay_buffer=object(), batch_size=1)

    agent.initialize_task_from_guide(task_index=1, guide_task_index=0)

    assert torch.equal(
        head_rows(agent.actor.mean_head, 1, agent.num_tasks),
        source,
    )


def test_best_return_restores_matching_backbone_and_head() -> None:
    agent = make_agent(backbone_source="best_return", head_source="best_return")
    agent.on_task_end(task_index=0, replay_buffer=object(), batch_size=1)
    expected_backbone = {
        key: value.detach().clone()
        for key, value in agent.actor.backbone.state_dict().items()
    }
    expected_head = head_rows(agent.actor.mean_head, 0, agent.num_tasks)
    with torch.no_grad():
        for parameter in agent.actor.parameters():
            parameter.add_(1.0)

    agent.initialize_task_from_guide(task_index=1, guide_task_index=0)

    for key, value in agent.actor.backbone.state_dict().items():
        assert torch.equal(value, expected_backbone[key])
    assert torch.equal(
        head_rows(agent.actor.mean_head, 1, agent.num_tasks),
        expected_head,
    )


def test_guide_policy_is_frozen_after_task_end() -> None:
    agent = make_agent(backbone_source="current", head_source="reset")
    observation = np.zeros(7, dtype=np.float32)
    observation[-3] = 1.0
    agent.on_task_end(task_index=0, replay_buffer=object(), batch_size=1)
    expected = agent.select_guide_action(
        observation,
        guide_task_index=0,
        deterministic=True,
    )

    with torch.no_grad():
        for parameter in agent.actor.parameters():
            parameter.add_(1.0)

    actual = agent.select_guide_action(
        observation,
        guide_task_index=0,
        deterministic=True,
    )
    assert np.array_equal(actual, expected)


def test_guide_policy_requires_completed_task_snapshot() -> None:
    agent = make_agent(backbone_source="current", head_source="reset")
    observation = np.zeros(7, dtype=np.float32)
    observation[-3] = 1.0

    with pytest.raises(RuntimeError, match="Missing frozen guide policy"):
        agent.select_guide_action(
            observation,
            guide_task_index=0,
        )


@pytest.mark.parametrize(
    ("backbone_source", "head_source"),
    (("current", "best_return"), ("best_return", "current")),
)
def test_rejects_mismatched_sources(backbone_source: str, head_source: str) -> None:
    with pytest.raises(ValueError):
        make_agent(
            backbone_source=backbone_source,
            head_source=head_source,
        )
