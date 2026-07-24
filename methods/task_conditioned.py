from methods.base import MethodSpec


METHOD = MethodSpec(
    method_id="task_conditioned",
    modes=frozenset({"continual"}),
    append_task_id=True,
)
