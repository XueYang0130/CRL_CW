from methods.success_replay import make_success_replay_method


METHOD = make_success_replay_method(
    "success_replay_best_llm_schedule_pcgrad",
    teacher="best",
    gradient_strategy="pcgrad_sac_priority",
    defaults={
        "bc_combination_strategy": "adaptive_additive",
        "bc_adaptive_target_ratio": 0.2,
        "bc_adaptive_conflict_ratio": 0.05,
        "llm_task_bc_schedule": True,
    },
)
