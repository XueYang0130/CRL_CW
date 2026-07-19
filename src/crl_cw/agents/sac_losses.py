"""Loss functions for the Soft Actor-Critic baseline.

This module contains only the mathematical SAC objectives:

1. entropy-regularized Bellman target;
2. two critic losses;
3. stochastic actor loss;
4. automatic entropy-temperature loss.

It deliberately contains no neural-network construction, optimizer step,
replay-buffer sampling, environment interaction, logging, or continual-
learning logic.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import functional as F


def compute_q_target(
    rewards: Tensor,
    dones: Tensor,
    next_q1: Tensor,
    next_q2: Tensor,
    next_log_prob: Tensor,
    alpha: Tensor,
    *,
    gamma: float,
) -> Tensor:
    """Compute the entropy-regularized Bellman target.

    The SAC target is:

        target_q =
            reward
            + gamma
            * (1 - done)
            * (
                min(target_q1, target_q2)
                - alpha * next_log_prob
            )

    Args:
        rewards:
            Reward batch. Shape must be `(batch_size,)` or
            `(batch_size, 1)`.

        dones:
            Episode-ending indicator. Values should be zero or one.
            Boolean tensors are also accepted.

        next_q1:
            Q values predicted by target critic 1 for the next-state
            policy action.

        next_q2:
            Q values predicted by target critic 2 for the next-state
            policy action.

        next_log_prob:
            Log probability of the next-state policy action.

        alpha:
            Current entropy coefficient. This should normally be
            `exp(log_alpha)`.

        gamma:
            Discount factor in `[0, 1]`.

    Returns:
        Detached target-Q tensor with shape `(batch_size,)`.

        The result is detached because a Bellman target must not send
        gradients into the target critics, next action, or alpha.
    """
    if not math.isfinite(gamma):
        raise ValueError("gamma must be finite.")

    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1].")

    _validate_scalar_tensor(alpha, name="alpha")

    _validate_batch_tensors(
        rewards=rewards,
        dones=dones,
        next_q1=next_q1,
        next_q2=next_q2,
        next_log_prob=next_log_prob,
    )

    reward_values = _as_batch_vector(
        rewards,
        name="rewards",
    )

    # Support both float dones and boolean dones.
    done_values = _as_batch_vector(
        dones,
        name="dones",
    ).to(dtype=reward_values.dtype)

    next_q1_values = _as_batch_vector(
        next_q1,
        name="next_q1",
    )

    next_q2_values = _as_batch_vector(
        next_q2,
        name="next_q2",
    )

    next_log_prob_values = _as_batch_vector(
        next_log_prob,
        name="next_log_prob",
    )

    min_next_q = torch.minimum(
        next_q1_values,
        next_q2_values,
    )

    q_target = (
        reward_values
        + float(gamma)
        * (1.0 - done_values)
        * (
            min_next_q
            - alpha.detach() * next_log_prob_values
        )
    )

    # Equivalent to TensorFlow's tf.stop_gradient used by
    # the Continual World implementation.
    return q_target.detach()


def compute_critic_losses(
    q1_predictions: Tensor,
    q2_predictions: Tensor,
    q_targets: Tensor,
) -> tuple[Tensor, Tensor]:
    """Compute the two SAC critic losses.

    CloneX/Continual World uses:

        q1_loss = 0.5 * mean((target_q - q1) ** 2)
        q2_loss = 0.5 * mean((target_q - q2) ** 2)

    The critics share the same Bellman target but remain independent
    neural networks.

    Args:
        q1_predictions:
            Current critic-1 predictions for replay-buffer actions.

        q2_predictions:
            Current critic-2 predictions for replay-buffer actions.

        q_targets:
            Bellman targets produced by `compute_q_target()`.

    Returns:
        Tuple `(q1_loss, q2_loss)`, where each item is a scalar tensor.
    """
    _validate_batch_tensors(
        q1_predictions=q1_predictions,
        q2_predictions=q2_predictions,
        q_targets=q_targets,
    )

    q1_values = _as_batch_vector(
        q1_predictions,
        name="q1_predictions",
    )

    q2_values = _as_batch_vector(
        q2_predictions,
        name="q2_predictions",
    )

    target_values = _as_batch_vector(
        q_targets,
        name="q_targets",
    ).detach()

    q1_loss = 0.5 * F.mse_loss(
        q1_values,
        target_values,
    )

    q2_loss = 0.5 * F.mse_loss(
        q2_values,
        target_values,
    )

    return q1_loss, q2_loss


def compute_actor_loss(
    log_prob: Tensor,
    q1_policy: Tensor,
    q2_policy: Tensor,
    alpha: Tensor,
) -> Tensor:
    """Compute the stochastic SAC actor loss.

    The objective minimized by the actor is:

        actor_loss =
            mean(
                alpha * log_probability
                - min(q1_policy, q2_policy)
            )

    Minimizing this objective encourages:

    - actions with high predicted Q values;
    - sufficient policy entropy.

    Args:
        log_prob:
            Log probability of reparameterized actions sampled from
            the current actor.

        q1_policy:
            Critic-1 values for the current actor's sampled actions.

        q2_policy:
            Critic-2 values for the current actor's sampled actions.

        alpha:
            Current entropy coefficient.

    Returns:
        Scalar actor-loss tensor.
    """
    _validate_scalar_tensor(alpha, name="alpha")

    _validate_batch_tensors(
        log_prob=log_prob,
        q1_policy=q1_policy,
        q2_policy=q2_policy,
    )

    log_prob_values = _as_batch_vector(
        log_prob,
        name="log_prob",
    )

    q1_values = _as_batch_vector(
        q1_policy,
        name="q1_policy",
    )

    q2_values = _as_batch_vector(
        q2_policy,
        name="q2_policy",
    )

    min_policy_q = torch.minimum(
        q1_values,
        q2_values,
    )

    # The actor objective must not update log_alpha.
    #
    # In the TensorFlow reference, gradients of actor_loss are requested
    # only for actor variables. In PyTorch, detach alpha explicitly so
    # that the same separation is guaranteed.
    return (
        alpha.detach() * log_prob_values
        - min_policy_q
    ).mean()


def compute_alpha_loss(
    log_alpha: Tensor,
    log_prob: Tensor,
    *,
    target_entropy: float,
) -> Tensor:
    """Compute the automatic entropy-temperature loss.

    The objective is:

        alpha_loss =
            -mean(
                log_alpha
                * stop_gradient(
                    log_probability + target_entropy
                )
            )

    Args:
        log_alpha:
            Trainable logarithm of the entropy coefficient.

        log_prob:
            Log probability of actions sampled from the current actor.

        target_entropy:
            Desired policy entropy.

    Returns:
        Scalar alpha-loss tensor.
    """
    _validate_scalar_tensor(
        log_alpha,
        name="log_alpha",
    )

    if not math.isfinite(target_entropy):
        raise ValueError(
            "target_entropy must be finite."
        )

    log_prob_values = _as_batch_vector(
        log_prob,
        name="log_prob",
    )

    if log_prob_values.shape[0] == 0:
        raise ValueError(
            "Batch tensors must not be empty."
        )

    # log_prob must be treated as a fixed target during the alpha update.
    # Otherwise alpha_loss would incorrectly update the actor.
    entropy_error = (
        log_prob_values.detach()
        + float(target_entropy)
    )

    return -(
        log_alpha * entropy_error
    ).mean()


def _as_batch_vector(
    value: Tensor,
    *,
    name: str,
) -> Tensor:
    """Normalize `(batch,)` and `(batch, 1)` tensors to `(batch,)`.

    This avoids accidental broadcasting such as:

        `(batch, 1) - (batch,) -> (batch, batch)`

    which can silently produce a completely incorrect loss.
    """
    if not isinstance(value, Tensor):
        raise TypeError(
            f"{name} must be a torch.Tensor."
        )

    if value.ndim == 1:
        return value

    if (
        value.ndim == 2
        and value.shape[1] == 1
    ):
        return value.squeeze(1)

    raise ValueError(
        f"{name} must have shape "
        "(batch_size,) or (batch_size, 1), "
        f"got {tuple(value.shape)}."
    )


def _validate_batch_tensors(
    **values: Tensor,
) -> None:
    """Validate batch size and device consistency."""
    vectors = {
        name: _as_batch_vector(
            value,
            name=name,
        )
        for name, value in values.items()
    }

    batch_sizes = {
        name: value.shape[0]
        for name, value in vectors.items()
    }

    if len(set(batch_sizes.values())) != 1:
        details = ", ".join(
            f"{name}={size}"
            for name, size in batch_sizes.items()
        )

        raise ValueError(
            "All batch tensors must have the same "
            f"batch size; got {details}."
        )

    batch_size = next(
        iter(batch_sizes.values())
    )

    if batch_size == 0:
        raise ValueError(
            "Batch tensors must not be empty."
        )

    devices = {
        value.device
        for value in vectors.values()
    }

    if len(devices) != 1:
        raise ValueError(
            "All batch tensors must be on the same device."
        )


def _validate_scalar_tensor(
    value: Tensor,
    *,
    name: str,
) -> None:
    """Validate a scalar or one-element tensor."""
    if not isinstance(value, Tensor):
        raise TypeError(
            f"{name} must be a torch.Tensor."
        )

    if value.numel() != 1:
        raise ValueError(
            f"{name} must contain exactly one value."
        )