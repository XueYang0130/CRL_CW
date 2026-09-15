from methods.base import MethodSpec
from methods.success_replay import build_agent


METHOD = MethodSpec(
    method_id="all_replay_best_pcgrad",
    modes=frozenset({"continual"}),
    append_task_id=True,
    multi_head=True,
    hide_task_id=True,
    reference_exploration=True,
    all_replay_teacher="best",
    defaults={
        "exploration_strategy": "best_return",
        "reset_buffer_on_task_change": True,
        "reset_optimizer_on_task_change": True,
        "episodic_memory_per_task": 10_000,
        "episodic_batch_size": 128,
        "actor_cloning_coefficient": 100.0,
        "bc_gradient_strategy": "pcgrad_sac_priority",
        "bc_max_norm_ratio": 1.0,
        "bc_combination_strategy": "average",
        "gradient_clip_norm": 0.1,
        "gradient_diagnostics": True,
    },
    agent_factory=build_agent,
)
