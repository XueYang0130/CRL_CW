from methods.success_replay import make_success_replay_method


METHOD = make_success_replay_method(
    "success_replay_best_adaptive_scaling",
    teacher="best",
    gradient_strategy="standard",
    defaults={
        "bc_combination_strategy": "adaptive_additive",
        "bc_adaptive_target_ratio": 0.2,
        "bc_adaptive_conflict_ratio": 0.05,
    },
)
