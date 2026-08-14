"""Adaptive critic initialization for continual RL.

At each task boundary, selects the best critic source from a bank of
historical critics (plus a random candidate) by running short probe
adaptation and picking the one with lowest final TD loss.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import torch

from agents import ReplayBuffer, SACAgent
from agents.sac_losses import compute_critic_losses, compute_q_target


@dataclass
class CriticSelectionResult:
    """Result of adaptive critic selection."""

    selected_source: str
    selected_task_index: int | None
    initial_td_loss: float
    final_td_loss: float
    all_candidates: list[dict[str, object]] = field(default_factory=list)


class CriticBank:
    """Stores critic state dicts after each task for later selection."""

    def __init__(self) -> None:
        self._states: list[dict[str, torch.Tensor]] = []

    def save(self, agent: SACAgent) -> None:
        """Save current critic + target critic state."""
        state = {
            "critic1": {k: v.cpu().clone() for k, v in agent.critic1.state_dict().items()},
            "critic2": {k: v.cpu().clone() for k, v in agent.critic2.state_dict().items()},
            "target_critic1": {k: v.cpu().clone() for k, v in agent.target_critic1.state_dict().items()},
            "target_critic2": {k: v.cpu().clone() for k, v in agent.target_critic2.state_dict().items()},
        }
        self._states.append(state)

    def __len__(self) -> int:
        return len(self._states)

    def load_into(self, agent: SACAgent, index: int) -> None:
        """Load historical critic state into agent."""
        state = self._states[index]
        agent.critic1.load_state_dict(state["critic1"], strict=True)
        agent.critic2.load_state_dict(state["critic2"], strict=True)
        agent.target_critic1.load_state_dict(state["target_critic1"], strict=True)
        agent.target_critic2.load_state_dict(state["target_critic2"], strict=True)


def _collect_probe_transitions(
    env,
    agent: SACAgent,
    num_transitions: int,
    max_episode_steps: int,
    seed: int,
) -> ReplayBuffer:
    """Collect transitions for probing critic adaptability."""
    buf = ReplayBuffer(
        observation_dim=agent.observation_dim,
        action_dim=agent.action_dim,
        capacity=num_transitions,
    )
    obs, _ = env.reset(seed=seed)
    ep_len = 0
    while len(buf) < num_transitions:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=agent.device).unsqueeze(0)
        with torch.no_grad():
            output = agent.actor.sample(obs_t)
        action = output.sampled_action.squeeze(0).cpu().numpy()
        next_obs, reward, terminated, truncated, info = env.step(action)
        ep_len += 1
        done = terminated or truncated or ep_len >= max_episode_steps
        buf.add(
            observation=obs,
            action=action,
            reward=reward,
            next_observation=next_obs,
            terminated=bool(terminated),
        )
        if done:
            obs, _ = env.reset()
            ep_len = 0
        else:
            obs = next_obs
    return buf


def _run_critic_warmup(
    agent: SACAgent,
    replay_buffer: ReplayBuffer,
    num_updates: int,
    batch_size: int,
) -> float:
    """Run critic-only TD updates, return final TD loss."""
    device = agent.device
    critic_params = list(agent.critic1.parameters()) + list(agent.critic2.parameters())
    optimizer = torch.optim.Adam(critic_params, lr=agent.learning_rate)
    final_loss = 0.0

    for _ in range(num_updates):
        batch = replay_buffer.sample(batch_size, device)
        obs = batch.observations
        actions = batch.actions
        rewards = batch.rewards
        next_obs = batch.next_observations
        dones = batch.terminated

        task_index = agent._task_index_from_observations(obs)
        alpha = agent.log_alpha[task_index].exp().detach()

        with torch.no_grad():
            next_out = agent.actor.sample(next_obs)
            next_q1 = agent.target_critic1(next_obs, next_out.sampled_action)
            next_q2 = agent.target_critic2(next_obs, next_out.sampled_action)
            q_targets = compute_q_target(
                rewards=rewards, dones=dones,
                next_q1=next_q1, next_q2=next_q2,
                next_log_prob=next_out.log_probability,
                alpha=alpha, gamma=agent.gamma,
            )

        q1 = agent.critic1(obs, actions)
        q2 = agent.critic2(obs, actions)
        q1_loss, q2_loss = compute_critic_losses(q1, q2, q_targets)
        loss = q1_loss + q2_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            for p, tp in zip(agent.critic1.parameters(), agent.target_critic1.parameters()):
                tp.data.mul_(agent.polyak).add_(p.data, alpha=1 - agent.polyak)
            for p, tp in zip(agent.critic2.parameters(), agent.target_critic2.parameters()):
                tp.data.mul_(agent.polyak).add_(p.data, alpha=1 - agent.polyak)

        final_loss = float(loss.item())

    return final_loss


def select_best_critic(
    agent: SACAgent,
    env,
    critic_bank: CriticBank,
    *,
    current_task_index: int,
    probe_transitions: int,
    warmup_updates: int,
    batch_size: int,
    max_episode_steps: int,
    seed: int,
) -> CriticSelectionResult:
    """Select the best critic source by probing adaptability.

    Candidates: each historical critic + a random initialization.
    Selection criterion: lowest final TD loss after K warmup updates.
    """
    # Save current state to restore actor after probing
    actor_state = {k: v.clone() for k, v in agent.actor.state_dict().items()}
    alpha_state = agent.log_alpha.data.clone()

    # Collect probe data
    probe_buffer = _collect_probe_transitions(
        env, agent, probe_transitions, max_episode_steps, seed=seed,
    )

    candidates: list[dict[str, object]] = []

    # Test each historical critic
    for hist_idx in range(len(critic_bank)):
        saved_critic1 = {k: v.clone() for k, v in agent.critic1.state_dict().items()}
        saved_critic2 = {k: v.clone() for k, v in agent.critic2.state_dict().items()}
        saved_tc1 = {k: v.clone() for k, v in agent.target_critic1.state_dict().items()}
        saved_tc2 = {k: v.clone() for k, v in agent.target_critic2.state_dict().items()}

        critic_bank.load_into(agent, hist_idx)
        final_loss = _run_critic_warmup(agent, probe_buffer, warmup_updates, batch_size)
        candidates.append({
            "source": f"historical_{hist_idx}",
            "task_index": hist_idx,
            "final_td_loss": final_loss,
        })

        # Restore critic for next candidate
        agent.critic1.load_state_dict(saved_critic1, strict=True)
        agent.critic2.load_state_dict(saved_critic2, strict=True)
        agent.target_critic1.load_state_dict(saved_tc1, strict=True)
        agent.target_critic2.load_state_dict(saved_tc2, strict=True)

    # Test random critic
    saved_critic1 = {k: v.clone() for k, v in agent.critic1.state_dict().items()}
    saved_critic2 = {k: v.clone() for k, v in agent.critic2.state_dict().items()}
    saved_tc1 = {k: v.clone() for k, v in agent.target_critic1.state_dict().items()}
    saved_tc2 = {k: v.clone() for k, v in agent.target_critic2.state_dict().items()}

    agent.reset_critics()
    final_loss_random = _run_critic_warmup(agent, probe_buffer, warmup_updates, batch_size)
    candidates.append({
        "source": "random",
        "task_index": None,
        "final_td_loss": final_loss_random,
    })

    # Restore critic state
    agent.critic1.load_state_dict(saved_critic1, strict=True)
    agent.critic2.load_state_dict(saved_critic2, strict=True)
    agent.target_critic1.load_state_dict(saved_tc1, strict=True)
    agent.target_critic2.load_state_dict(saved_tc2, strict=True)

    # Restore actor and alpha (unchanged by critic probing, but be safe)
    agent.actor.load_state_dict(actor_state, strict=True)
    agent.log_alpha.data.copy_(alpha_state)

    # Select best candidate
    best = min(candidates, key=lambda c: c["final_td_loss"])

    # Apply the winning critic
    if best["task_index"] is not None:
        critic_bank.load_into(agent, best["task_index"])
    else:
        agent.reset_critics()

    # Re-run warmup on the selected critic to leave it in adapted state
    initial_loss = best["final_td_loss"]
    adapted_loss = _run_critic_warmup(agent, probe_buffer, warmup_updates, batch_size)

    return CriticSelectionResult(
        selected_source=best["source"],
        selected_task_index=best["task_index"],
        initial_td_loss=initial_loss,
        final_td_loss=adapted_loss,
        all_candidates=candidates,
    )
