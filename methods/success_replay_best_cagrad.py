from methods.success_replay import make_success_replay_method


METHOD = make_success_replay_method(
    "success_replay_best_cagrad",
    teacher="best",
    defaults={
        "bc_combination_strategy": "cagrad",
        "bc_cagrad_alpha": 0.5,
    },
)
