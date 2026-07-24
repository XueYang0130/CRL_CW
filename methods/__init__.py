from methods.base import MethodSpec
from methods.clonex_sac import METHOD as CLONEX_SAC
from methods.jsrl_continual import METHOD as JSRL_CONTINUAL
from methods.fine_tuning import METHOD as FINE_TUNING
from methods.packnet import METHOD as PACKNET
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
        JSRL_CONTINUAL,
        WSRL_CONTINUAL,
    )
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


def is_method_compatible(mode: str, method_id: str) -> bool:
    return mode in get_method(method_id).modes


__all__ = [
    "METHOD_REGISTRY",
    "MethodSpec",
    "available_method_ids",
    "default_method_for_mode",
    "get_method",
    "is_method_compatible",
    "method_defaults",
]
