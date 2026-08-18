from __future__ import annotations

import numpy as np

from scripts.diagnose_memory_criticality_coverage import (
    pearson_correlation,
    spectral_geometry_metrics,
    unique_union_indices,
)


def test_spectral_geometry_separates_critical_and_candidate_coverage() -> None:
    critical = np.tile(np.asarray([[1.0, 0.0]], dtype=np.float32), (20, 1))
    coverage = np.concatenate(
        (
            critical,
            np.tile(np.asarray([[0.0, 1.0]], dtype=np.float32), (20, 1)),
        ),
        axis=0,
    )
    metrics = spectral_geometry_metrics(
        memory_signatures=critical,
        critical_signatures=critical,
        coverage_signatures=coverage,
        rank=1,
    )

    assert np.isclose(metrics["critical_energy_coverage"], 1.0)
    assert np.isclose(metrics["candidate_energy_coverage"], 0.5)
    assert np.isclose(metrics["effective_rank"], 1.0)
    assert metrics["critical_covariance_error"] == 0.0
    assert metrics["candidate_covariance_error"] > 0.0


def test_unique_union_indices_keep_first_payload_precedence() -> None:
    first = np.asarray([[1.0, 0.0], [2.0, 0.0]], dtype=np.float32)
    second = np.asarray([[2.0, 0.0], [3.0, 0.0]], dtype=np.float32)

    first_indices, second_indices = unique_union_indices(first, second)

    np.testing.assert_array_equal(first_indices, [0, 1])
    np.testing.assert_array_equal(second_indices, [1])


def test_pearson_correlation_handles_signal_and_degenerate_inputs() -> None:
    assert np.isclose(pearson_correlation([1, 2, 3], [2, 4, 6]), 1.0)
    assert pearson_correlation([1, 1, 1], [1, 2, 3]) is None
    assert pearson_correlation([1, 2], [1, 2]) is None
