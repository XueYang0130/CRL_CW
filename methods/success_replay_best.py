from methods.success_replay import make_success_replay_method


METHOD = make_success_replay_method(
    "success_replay_best",
    teacher="best",
)
