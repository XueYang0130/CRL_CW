from methods.base import MethodSpec
from methods.adaptive_semantic_bc import METHOD as ADAPTIVE_SEMANTIC_BC
from methods.clonex_sac import METHOD as CLONEX_SAC
from methods.full_bc import METHOD as FULL_BC
from methods.full_bc_norm_balanced import METHOD as FULL_BC_NORM_BALANCED
from methods.full_bc_pcgrad import METHOD as FULL_BC_PCGRAD
from methods.general_task_specific_bc import METHOD as GENERAL_TASK_SPECIFIC_BC
from methods.jsrl_continual import METHOD as JSRL_CONTINUAL
from methods.fine_tuning import METHOD as FINE_TUNING
from methods.packnet import METHOD as PACKNET
from methods.semantic_hybrid_bc import METHOD as SEMANTIC_HYBRID_BC
from methods.semantic_local_bc import METHOD as SEMANTIC_LOCAL_BC
from methods.stage_aware_semantic_bc import METHOD as STAGE_AWARE_SEMANTIC_BC
from methods.success_replay_best_adaptive_pcgrad import (
    METHOD as SUCCESS_REPLAY_BEST_ADAPTIVE_PCGRAD,
)
from methods.success_replay_best import METHOD as SUCCESS_REPLAY_BEST
from methods.success_replay_best_pcgrad import METHOD as SUCCESS_REPLAY_BEST_PCGRAD
from methods.success_replay_final import METHOD as SUCCESS_REPLAY_FINAL
from methods.task_conditioned import METHOD as TASK_CONDITIONED
from methods.wsrl_continual import METHOD as WSRL_CONTINUAL


SINGLE_TASK_BASELINE = MethodSpec(
    method_id="single_task_baseline",
    modes=frozenset({"single", "single-batch"}),
)
METHOD_REGISTRY: dict[str, MethodSpec] = {
    method.method_id: method
    for method in (
        SINGLE_TASK_BASELINE,
        FINE_TUNING,
        TASK_CONDITIONED,
        PACKNET,
        CLONEX_SAC,
        FULL_BC,
        FULL_BC_NORM_BALANCED,
        FULL_BC_PCGRAD,
        SEMANTIC_LOCAL_BC,
        SEMANTIC_HYBRID_BC,
        STAGE_AWARE_SEMANTIC_BC,
        SUCCESS_REPLAY_FINAL,
        SUCCESS_REPLAY_BEST,
        SUCCESS_REPLAY_BEST_PCGRAD,
        SUCCESS_REPLAY_BEST_ADAPTIVE_PCGRAD,
        ADAPTIVE_SEMANTIC_BC,
        GENERAL_TASK_SPECIFIC_BC,
        JSRL_CONTINUAL,
        WSRL_CONTINUAL,
    )
}

BC_GRADIENT_STRATEGY_BY_METHOD = {
    "clonex_sac": "standard",
    "full_bc": "standard",
    "full_bc_norm_balanced": "norm_balanced",
    "full_bc_pcgrad": "pcgrad_sac_priority",
    "success_replay_final": "standard",
    "success_replay_best": "standard",
    "success_replay_best_pcgrad": "pcgrad_sac_priority",
    "success_replay_best_adaptive_pcgrad": "pcgrad_sac_priority",
}


def get_method(method_id: str) -> MethodSpec:
    try:
        return METHOD_REGISTRY[method_id]
    except KeyError as error:
        raise ValueError(f"Unsupported method: {method_id}") from error


def available_method_ids() -> list[str]:
    return list(METHOD_REGISTRY)


def default_method_for_mode(mode: str) -> str:
    if mode in {"single", "single-batch"}:
        return "single_task_baseline"
    if mode == "continual":
        return "fine_tuning"
    raise ValueError(f"Unsupported mode: {mode}")


def method_defaults(method_id: str) -> dict[str, object]:
    return dict(get_method(method_id).defaults)


def bc_gradient_strategy_for_method(method_id: str) -> str:
    get_method(method_id)
    return BC_GRADIENT_STRATEGY_BY_METHOD.get(method_id, "standard")


def is_method_compatible(mode: str, method_id: str) -> bool:
    return mode in get_method(method_id).modes


__all__ = [
    "METHOD_REGISTRY",
    "BC_GRADIENT_STRATEGY_BY_METHOD",
    "MethodSpec",
    "available_method_ids",
    "bc_gradient_strategy_for_method",
    "default_method_for_mode",
    "get_method",
    "is_method_compatible",
    "method_defaults",
]
