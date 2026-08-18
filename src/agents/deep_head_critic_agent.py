from __future__ import annotations

from torch import nn

from agents.full_bc_agent import FullBehaviorCloningSACAgent
from agents.networks import DeepHeadQCritic


class DeepHeadCriticFullBCAgent(FullBehaviorCloningSACAgent):
    """Adaptive-PCGrad actor with nonlinear task-specific critic heads."""

    def __init__(self, *args, critic_head_hidden_size: int = 64, **kwargs) -> None:
        if critic_head_hidden_size <= 0:
            raise ValueError("critic_head_hidden_size must be positive.")
        self.critic_head_hidden_size = int(critic_head_hidden_size)
        super().__init__(*args, **kwargs)

    def _make_critic(self) -> nn.Module:
        return DeepHeadQCritic(
            observation_dim=self.observation_dim,
            action_dim=self.action_dim,
            task_id_dim=self.task_id_dim,
            num_heads=self.num_tasks,
            hide_task_id=self.hide_task_id,
            head_hidden_size=self.critic_head_hidden_size,
        )
