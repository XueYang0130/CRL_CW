"""Reinforcement-learning agents and supporting components."""
from .sac_agent import SACAgent

from crl_cw.agents.networks import (
    DEFAULT_HIDDEN_SIZES,
    LOG_STD_MAX,
    LOG_STD_MIN,
    ActorOutput,
    GaussianActor,
    QCritic,
)
from crl_cw.agents.replay_buffer import ReplayBatch, ReplayBuffer

__all__ = [
    "DEFAULT_HIDDEN_SIZES",
    "LOG_STD_MAX",
    "LOG_STD_MIN",
    "ActorOutput",
    "GaussianActor",
    "QCritic",
    "ReplayBatch",
    "ReplayBuffer",
    "SACAgent",
    "compute_actor_loss",
"compute_alpha_loss",
"compute_critic_losses",
"compute_q_target",
]

from .sac_losses import (
    compute_actor_loss,
    compute_alpha_loss,
    compute_critic_losses,
    compute_q_target,
)