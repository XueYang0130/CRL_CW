from methods.success_replay import make_success_replay_method


METHOD = make_success_replay_method(
    "success_replay_final_pcgrad",
    teacher="final",
    gradient_strategy="pcgrad_sac_priority",
)
