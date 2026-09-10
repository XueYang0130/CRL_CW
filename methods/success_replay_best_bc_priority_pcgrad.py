from methods.success_replay import make_success_replay_method


METHOD = make_success_replay_method(
    "success_replay_best_bc_priority_pcgrad",
    teacher="best",
    gradient_strategy="pcgrad_bc_priority",
)
