from __future__ import annotations

import numpy as np
import torch

from scripts.diagnose_success_cluster_balancing import (
    extract_bc_feature_gradients,
    kmeans_labels,
    sample_balanced_groups,
    sample_one_group_per_microbatch,
)
from scripts.run_stickpull_causal_ablation import GradientClusterRoutedSampler


class _TinyActor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = torch.nn.Linear(3, 4)
        self.mean_head = torch.nn.Linear(4, 2)
        self.log_std_head = torch.nn.Linear(4, 2)

    def distribution_parameters(
        self, observations: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(observations)
        return self.mean_head(features), self.log_std_head(features).clamp(-2.0, 2.0)


class _TinyAgent:
    def __init__(self) -> None:
        self.actor = _TinyActor()
        self.device = torch.device("cpu")

    @staticmethod
    def _gaussian_kl(
        first_mean: torch.Tensor,
        first_log_std: torch.Tensor,
        second_mean: torch.Tensor,
        second_log_std: torch.Tensor,
    ) -> torch.Tensor:
        first_variance = first_log_std.exp().square() + 1e-6
        second_variance = second_log_std.exp().square() + 1e-6
        return (
            second_log_std
            - first_log_std
            + (first_variance + (first_mean - second_mean).square())
            / (2.0 * second_variance)
            - 0.5
        ).sum(dim=-1)


def test_bc_feature_gradients_are_per_state_and_read_only() -> None:
    agent = _TinyAgent()
    observations = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    target_means = np.zeros((3, 2), dtype=np.float32)
    target_log_stds = np.zeros((3, 2), dtype=np.float32)

    signatures = extract_bc_feature_gradients(
        agent=agent,
        observations=observations,
        target_means=target_means,
        target_log_stds=target_log_stds,
        batch_size=2,
    )

    assert signatures.shape == (3, 4)
    assert np.all(np.isfinite(signatures))
    assert all(parameter.grad is None for parameter in agent.actor.parameters())
    assert agent.actor.training


def test_bc_feature_gradient_extraction_preserves_eval_mode() -> None:
    agent = _TinyAgent()
    agent.actor.eval()
    extract_bc_feature_gradients(
        agent=agent,
        observations=np.ones((2, 3), dtype=np.float32),
        target_means=np.zeros((2, 2), dtype=np.float32),
        target_log_stds=np.zeros((2, 2), dtype=np.float32),
        batch_size=2,
    )

    assert not agent.actor.training


def test_kmeans_recovers_separated_groups_without_empty_clusters() -> None:
    rng = np.random.default_rng(3)
    features = np.concatenate(
        (
            rng.normal(loc=-5.0, scale=0.1, size=(20, 3)),
            rng.normal(loc=0.0, scale=0.1, size=(20, 3)),
            rng.normal(loc=5.0, scale=0.1, size=(20, 3)),
        ),
        axis=0,
    ).astype(np.float32)
    labels, centers = kmeans_labels(
        features,
        clusters=3,
        iterations=20,
        rng=np.random.default_rng(11),
    )

    assert centers.shape == (3, 3)
    assert np.all(np.bincount(labels, minlength=3) > 0)
    assert len(set(labels[:20])) == 1
    assert len(set(labels[20:40])) == 1
    assert len(set(labels[40:])) == 1


def test_cluster_balanced_sampler_uses_every_group_equally() -> None:
    candidates = {
        (0, 0): np.arange(0, 10),
        (0, 1): np.arange(10, 20),
        (1, 0): np.arange(20, 30),
        (1, 1): np.arange(30, 40),
    }
    batches = sample_balanced_groups(
        candidates_by_group=candidates,
        microbatches=4,
        batch_size=12,
        rng=np.random.default_rng(5),
    )

    for batch in batches:
        counts = [
            np.count_nonzero((batch >= start) & (batch < start + 10))
            for start in (0, 10, 20, 30)
        ]
        assert counts == [3, 3, 3, 3]


def test_cluster_routed_sampler_uses_one_group_per_microbatch() -> None:
    candidates = {
        (0, 0): np.arange(0, 20),
        (0, 1): np.arange(20, 40),
        (1, 0): np.arange(40, 60),
        (1, 1): np.arange(60, 80),
    }
    batches = sample_one_group_per_microbatch(
        candidates_by_group=candidates,
        microbatches=4,
        batch_size=8,
        rng=np.random.default_rng(13),
    )

    represented_groups = set()
    for batch in batches:
        group = int(batch.min() // 20)
        assert np.all((batch // 20) == group)
        represented_groups.add(group)
    assert represented_groups == {0, 1, 2, 3}


def test_training_routed_sampler_cycles_groups_uniformly() -> None:
    candidates = {
        (0, 0): np.arange(0, 20),
        (0, 1): np.arange(20, 40),
        (1, 0): np.arange(40, 60),
        (1, 1): np.arange(60, 80),
    }
    sampler = GradientClusterRoutedSampler(
        candidates_by_group=candidates,
        batch_size=8,
        seed=17,
    )

    sampled_groups = []
    for _ in range(9):
        indices, group = sampler.sample_indices()
        sampled_groups.append(group)
        assert indices.shape == (8,)
        assert np.all(np.isin(indices, candidates[group]))
        assert np.unique(indices).size == 8

    assert set(sampled_groups[:4]) == set(candidates)
    assert set(sampled_groups[4:8]) == set(candidates)
    counts = list(sampler.draw_counts.values())
    assert max(counts) - min(counts) == 1


def test_training_routed_sampler_rejects_underfilled_cluster() -> None:
    with np.testing.assert_raises_regex(ValueError, "BC batch size"):
        GradientClusterRoutedSampler(
            candidates_by_group={(0, 0): np.arange(7)},
            batch_size=8,
            seed=0,
        )
