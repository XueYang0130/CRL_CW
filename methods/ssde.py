from typing import Any

from agents import SACAgent, SSDEFullSACAgent, SSDESACAgent
from methods.base import MethodSpec


def build_approx_agent(
    agent_kwargs: dict[str, Any],
    args: Any,
    total_tasks: int,
) -> SACAgent:
    del total_tasks
    return SSDESACAgent(
        **agent_kwargs,
        dormancy_threshold=args.ssde_dormancy_threshold,
        dormancy_check_every=args.ssde_dormancy_check_every,
        retrain_steps=args.ssde_retrain_steps,
    )


APPROX_METHOD = MethodSpec(
    # This is deliberately not registered as "ssde": it is a lightweight
    # compatibility approximation and does not implement SSDE's Sentence-BERT
    # sparse-prompt co-allocation or task-specific inference mechanism.
    method_id="ssde_approx",
    modes=frozenset({"continual"}),
    append_task_id=True,
    multi_head=True,
    hide_task_id=True,
    reference_exploration=True,
    defaults={
        "ssde_dormancy_threshold": 0.01,
        "ssde_dormancy_check_every": 10_000,
        "ssde_retrain_steps": 0,
        "gradient_clip_norm": 0.1,
    },
    agent_factory=build_approx_agent,
)


def build_agent(
    agent_kwargs: dict[str, Any],
    args: Any,
    total_tasks: int,
) -> SACAgent:
    del total_tasks
    return SSDEFullSACAgent(
        **agent_kwargs,
        ssde_seed=args.seed,
        hidden_size=args.ssde_hidden_size,
        descriptor_dim=args.ssde_descriptor_dim,
        fixed_lasso_alpha=args.ssde_fixed_lasso_alpha,
        random_lasso_alpha_start=args.ssde_random_lasso_alpha_start,
        random_lasso_alpha_end=args.ssde_random_lasso_alpha_end,
        default_beta=args.ssde_default_beta,
        use_adaptive_beta=args.ssde_adaptive_beta,
        beta_lambda=args.ssde_beta_lambda,
        sensitivity_interval=args.ssde_sensitivity_interval,
        sensitivity_batch_size=args.ssde_sensitivity_batch_size,
        sensitivity_threshold=args.ssde_sensitivity_threshold,
        stop_reset_after_steps=args.ssde_stop_reset_after_steps,
        ssde_target_entropy=args.ssde_target_entropy,
        random_distillation=args.ssde_random_distillation,
        distillation_steps=args.ssde_distillation_steps,
        distillation_batch_size=args.batch_size,
    )


METHOD = MethodSpec(
    method_id="ssde",
    modes=frozenset({"continual"}),
    append_task_id=True,
    multi_head=True,
    hide_task_id=True,
    reference_exploration=True,
    defaults={
        "learning_rate": 3e-4,
        "batch_size": 256,
        "initial_log_alpha": 0.0,
        "update_after": 10_000,
        "gradient_clip_norm": None,
        "exploration_strategy": "uniform_previous",
        "reset_buffer_on_task_change": True,
        "reset_optimizer_on_task_change": True,
        "reset_critic_on_task_change": True,
    },
    agent_factory=build_agent,
)
