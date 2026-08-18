from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from scripts.build_stickpull_matched_memory import (
    sample_stratified_broad_memory,
    sample_without_replacement,
)
from scripts.run_stickpull_causal_ablation import (
    build_dynamic_success_dominant_memory,
    build_nonoverlapping_mixed_memory,
    configure_integration,
)


def test_matched_memory_sampling_is_bounded_and_without_duplicates() -> None:
    observations = np.arange(60, dtype=np.float32).reshape(20, 3)
    sampled = sample_without_replacement(
        observations,
        capacity=7,
        rng=np.random.default_rng(123),
    )

    assert sampled.shape == (7, 3)
    assert np.unique(sampled, axis=0).shape[0] == 7
    assert all(any(np.array_equal(row, source) for source in observations) for row in sampled)


def test_matched_memory_sampling_keeps_small_payload_intact() -> None:
    observations = np.arange(12, dtype=np.float32).reshape(4, 3)
    sampled = sample_without_replacement(
        observations,
        capacity=10,
        rng=np.random.default_rng(123),
    )

    np.testing.assert_array_equal(sampled, observations)
    assert sampled is not observations


def test_broad_memory_has_exact_random_quota() -> None:
    policy = np.column_stack(
        (np.zeros(20, dtype=np.float32), np.arange(20, dtype=np.float32))
    )
    random = np.column_stack(
        (np.ones(10, dtype=np.float32), np.arange(10, dtype=np.float32))
    )

    sampled, random_count = sample_stratified_broad_memory(
        policy_pool=policy,
        random_pool=random,
        capacity=10,
        random_ratio=0.2,
        rng=np.random.default_rng(9),
    )

    assert sampled.shape == (10, 2)
    assert random_count == 2
    assert int(sampled[:, 0].sum()) == 2


def test_causal_ablation_integration_modes_are_explicit() -> None:
    agent = SimpleNamespace(
        bc_gradient_strategy="unexpected",
        bc_combination_strategy="unexpected",
    )
    agent.configure_bc_update_interval = lambda interval: setattr(
        agent, "bc_update_interval", interval
    )

    configure_integration(
        agent,
        integration="adaptive_pcgrad",
        bc_update_interval=4,
        adaptive_target_ratio=0.2,
        adaptive_conflict_ratio=0.05,
    )
    assert agent.bc_gradient_strategy == "pcgrad_sac_priority"
    assert agent.bc_combination_strategy == "adaptive_additive"
    assert agent.bc_update_interval == 4
    assert agent.bc_adaptive_target_ratio == 0.2
    assert agent.bc_adaptive_conflict_ratio == 0.05

    configure_integration(
        agent,
        integration="plain",
        bc_update_interval=1,
        adaptive_target_ratio=0.05,
        adaptive_conflict_ratio=0.0125,
    )
    assert agent.bc_gradient_strategy == "standard"
    assert agent.bc_combination_strategy == "average"


def test_mixed50_memory_is_task_balanced_and_nonoverlapping() -> None:
    def payload(offset: int) -> dict[str, np.ndarray]:
        rows = []
        for task_index in range(2):
            for state_index in range(12):
                task_id = [0.0, 0.0]
                task_id[task_index] = 1.0
                rows.append([offset + 100 * task_index + state_index, *task_id])
        observations = np.asarray(rows, dtype=np.float32)
        return {
            "observations": observations,
            "target_means": observations[:, :1].copy(),
            "target_log_stds": np.zeros((observations.shape[0], 1), dtype=np.float32),
        }

    success = payload(0)
    broad = payload(1_000)
    observations, means, log_stds, rows = build_nonoverlapping_mixed_memory(
        success_payload=success,
        broad_payload=broad,
        task_id_dim=2,
        expected_source_tasks=["task-a", "task-b"],
        states_per_task=10,
        seed=7,
    )

    assert observations.shape == (20, 3)
    assert means.shape == log_stds.shape == (20, 1)
    assert len({row.tobytes() for row in observations}) == 20
    assert {(row["source_task_name"], row["memory_source"], row["states"]) for row in rows} == {
        ("task-a", "success", 5),
        ("task-a", "broad", 5),
        ("task-b", "success", 5),
        ("task-b", "broad", 5),
    }
    task_indices = np.argmax(observations[:, -2:], axis=1)
    np.testing.assert_array_equal(np.bincount(task_indices, minlength=2), [10, 10])


def test_dynamic_mixed_memory_uses_broad_to_fill_success_shortfall() -> None:
    def payload(rows: list[list[float]]) -> dict[str, np.ndarray]:
        observations = np.asarray(rows, dtype=np.float32)
        return {
            "observations": observations,
            "target_means": observations[:, :1].copy(),
            "target_log_stds": np.zeros((observations.shape[0], 1), dtype=np.float32),
        }

    success = payload([[float(index), 1.0] for index in range(3)])
    broad = payload([[100.0 + index, 1.0] for index in range(10)])
    observations, means, log_stds, rows = build_dynamic_success_dominant_memory(
        success_payload=success,
        broad_payload=broad,
        task_id_dim=1,
        expected_source_tasks=["task-a"],
        states_per_task=10,
        success_fraction=0.8,
        seed=4,
    )

    assert observations.shape == (10, 2)
    assert means.shape == log_stds.shape == (10, 1)
    assert len({row.tobytes() for row in observations}) == 10
    assert {(row["memory_source"], row["states"]) for row in rows} == {
        ("success", 3),
        ("broad", 7),
    }


def test_dynamic_mixed_memory_keeps_nominal_eighty_twenty_split() -> None:
    def payload(offset: int) -> dict[str, np.ndarray]:
        observations = np.asarray(
            [[offset + float(index), 1.0] for index in range(12)],
            dtype=np.float32,
        )
        return {
            "observations": observations,
            "target_means": observations[:, :1].copy(),
            "target_log_stds": np.zeros((observations.shape[0], 1), dtype=np.float32),
        }

    _, _, _, rows = build_dynamic_success_dominant_memory(
        success_payload=payload(0),
        broad_payload=payload(100),
        task_id_dim=1,
        expected_source_tasks=["task-a"],
        states_per_task=10,
        success_fraction=0.8,
        seed=9,
    )

    assert {(row["memory_source"], row["states"]) for row in rows} == {
        ("success", 8),
        ("broad", 2),
    }
