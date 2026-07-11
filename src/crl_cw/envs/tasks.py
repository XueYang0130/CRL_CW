"""Task definitions for the Continual World experiments.

The CW10 order follows the ClonEx-SAC supplementary implementation.
Keep this order fixed because continual-learning results depend on the
task sequence.
"""

CW10_TASKS: tuple[str, ...] = (
    "hammer-v1",
    "push-wall-v1",
    "faucet-close-v1",
    "push-back-v1",
    "stick-pull-v1",
    "handle-press-side-v1",
    "push-v1",
    "shelf-place-v1",
    "window-close-v1",
    "peg-unplug-side-v1",
)


def get_cw10_tasks() -> list[str]:
    """Return a new list containing the fixed CW10 task sequence."""
    return list(CW10_TASKS)