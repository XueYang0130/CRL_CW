from dataclasses import replace

from methods.success_replay import make_success_replay_method


_BASE = make_success_replay_method(
    "success_replay_best_llm_prior_pcgrad",
    teacher="best",
    gradient_strategy="pcgrad_sac_priority",
    defaults={
        "bc_combination_strategy": "adaptive_additive",
        "bc_adaptive_target_ratio": 0.2,
        "bc_adaptive_conflict_ratio": 0.05,
        "exploration_strategy": "current",
    },
)

METHOD = replace(
    _BASE,
    reference_exploration=True,
    llm_prior_initialization=True,
)
