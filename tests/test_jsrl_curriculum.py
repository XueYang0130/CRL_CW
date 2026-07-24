from training.continual_experiment import (
    jsrl_advance_reason,
    jsrl_horizons,
    reached_relative_threshold,
)


def test_jsrl_horizons_include_autonomous_final_stage() -> None:
    assert jsrl_horizons(initial_steps=180, stages=10) == [
        180,
        160,
        140,
        120,
        100,
        80,
        60,
        40,
        20,
        0,
    ]


def test_relative_threshold_supports_positive_and_negative_returns() -> None:
    assert reached_relative_threshold(current=95.0, reference=100.0, tolerance=0.05)
    assert reached_relative_threshold(current=-105.0, reference=-100.0, tolerance=0.05)
    assert not reached_relative_threshold(current=-106.0, reference=-100.0, tolerance=0.05)


def test_zero_reference_requires_strict_improvement() -> None:
    assert not reached_relative_threshold(current=0.0, reference=0.0, tolerance=0.05)
    assert reached_relative_threshold(current=1.0, reference=0.0, tolerance=0.05)


def test_jsrl_advances_when_performance_reaches_threshold() -> None:
    assert jsrl_advance_reason(
        moving_average=90.0,
        reference=100.0,
        tolerance=0.10,
        stage_evaluations=1,
        min_evaluations=1,
        max_evaluations_without_advance=5,
        has_next_stage=True,
    ) == "performance"


def test_jsrl_patience_forces_fifth_evaluation_advance() -> None:
    common = {
        "moving_average": 50.0,
        "reference": 100.0,
        "tolerance": 0.10,
        "min_evaluations": 1,
        "max_evaluations_without_advance": 5,
        "has_next_stage": True,
    }
    assert jsrl_advance_reason(stage_evaluations=4, **common) is None
    assert jsrl_advance_reason(stage_evaluations=5, **common) == "patience"


def test_jsrl_does_not_advance_past_final_stage() -> None:
    assert jsrl_advance_reason(
        moving_average=100.0,
        reference=100.0,
        tolerance=0.10,
        stage_evaluations=5,
        min_evaluations=1,
        max_evaluations_without_advance=5,
        has_next_stage=False,
    ) is None
