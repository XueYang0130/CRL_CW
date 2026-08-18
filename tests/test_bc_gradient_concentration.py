from __future__ import annotations

import numpy as np

from scripts.diagnose_bc_gradient_concentration import (
    gradient_concentration_metrics,
    sample_microbatch_indices,
)


def test_identical_gradients_are_maximally_concentrated() -> None:
    gradients = np.tile(np.asarray([[1.0, 2.0, 3.0]]), (4, 1))
    metrics = gradient_concentration_metrics(gradients)

    assert np.isclose(metrics["mean_pairwise_cosine"], 1.0)
    assert np.isclose(metrics["directional_coherence"], 1.0)
    assert np.isclose(metrics["effective_rank"], 1.0)
    assert np.isclose(metrics["top_eigenvalue_fraction"], 1.0)


def test_orthogonal_gradients_have_high_effective_rank() -> None:
    gradients = np.eye(4, dtype=np.float64)
    metrics = gradient_concentration_metrics(gradients)

    assert np.isclose(metrics["mean_pairwise_cosine"], 0.0)
    assert np.isclose(metrics["directional_coherence"], 0.5)
    assert np.isclose(metrics["effective_rank"], 4.0)
    assert np.isclose(metrics["stable_rank"], 4.0)


def test_balanced_microbatches_include_every_source_task() -> None:
    candidates = {
        0: np.arange(0, 20),
        1: np.arange(20, 40),
        2: np.arange(40, 60),
        3: np.arange(60, 80),
    }
    batches = sample_microbatch_indices(
        candidates_by_task=candidates,
        microbatches=5,
        batch_size=12,
        rng=np.random.default_rng(7),
    )

    assert len(batches) == 5
    for batch in batches:
        assert batch.shape == (12,)
        task_counts = [
            np.count_nonzero((batch >= start) & (batch < start + 20))
            for start in (0, 20, 40, 60)
        ]
        assert task_counts == [3, 3, 3, 3]
