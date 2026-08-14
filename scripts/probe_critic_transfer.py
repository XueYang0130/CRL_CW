"""Probe critic transfer hypotheses using existing ClonEx checkpoints.

For each task transition T_{i-1} -> T_i, this script measures:
1. Critic adaptability: TD loss reduction after K critic-only updates
2. Q-gradient alignment: cos(∇_a Q_transferred, ∇_a Q_adapted)
3. Critic disagreement: |Q1 - Q2| as a readiness signal

Compares: previous critic vs random critic vs each historical critic.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents import ReplayBuffer, SACAgent
from agents.sac_losses import compute_critic_losses, compute_q_target
from envs import get_cw10_tasks, make_cw_env, extract_success
from utils.checkpoint import load_sac_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe critic transfer between tasks")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help="Path to checkpoints/ directory (containing task_0.pt ... task_9.pt)",
    )
    parser.add_argument("--env-version", default="v3")
    parser.add_argument("--reward-function-version", default="cw10_v1")
    parser.add_argument("--max-episode-steps", type=int, default=200)
    parser.add_argument("--probe-transitions", type=int, default=5000,
                        help="Transitions to collect for probing")
    parser.add_argument("--warmup-updates", type=int, default=500,
                        help="Critic-only TD updates for adaptability measurement")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--target-tasks", type=int, nargs="+", default=None,
                        help="Which task indices to probe (default: 1-9)")
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def collect_probe_data(
    env,
    agent: SACAgent,
    task_index: int,
    num_transitions: int,
    max_episode_steps: int,
) -> ReplayBuffer:
    """Collect transitions using the exploration policy (previous best head)."""
    buf = ReplayBuffer(
        observation_dim=agent.observation_dim,
        action_dim=agent.action_dim,
        capacity=num_transitions,
    )
    obs, _ = env.reset()
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


def compute_td_loss(
    agent: SACAgent,
    batch,
) -> float:
    """Compute TD loss without updating anything."""
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
    return float((q1_loss + q2_loss).item())


def critic_only_updates(
    agent: SACAgent,
    replay_buffer: ReplayBuffer,
    num_updates: int,
    batch_size: int,
) -> list[float]:
    """Run critic-only updates (actor frozen) and return loss curve."""
    critic_params = list(agent.critic1.parameters()) + list(agent.critic2.parameters())
    optimizer = torch.optim.Adam(critic_params, lr=agent.learning_rate)
    losses = []

    device = agent.device
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

        # Soft update targets
        with torch.no_grad():
            for p, tp in zip(agent.critic1.parameters(), agent.target_critic1.parameters()):
                tp.data.mul_(agent.polyak).add_(p.data, alpha=1 - agent.polyak)
            for p, tp in zip(agent.critic2.parameters(), agent.target_critic2.parameters()):
                tp.data.mul_(agent.polyak).add_(p.data, alpha=1 - agent.polyak)

        losses.append(float(loss.item()))

    return losses


def compute_action_gradient_alignment(
    agent_a: SACAgent,
    agent_b: SACAgent,
    observations: torch.Tensor,
    actions: torch.Tensor,
) -> float:
    """Compute cosine similarity between ∇_a Q of two critics."""
    actions_a = actions.clone().requires_grad_(True)
    q1_a = agent_a.critic1(observations, actions_a)
    q2_a = agent_a.critic2(observations, actions_a)
    q_a = torch.min(q1_a, q2_a).sum()
    grad_a = torch.autograd.grad(q_a, actions_a, create_graph=False)[0]

    actions_b = actions.clone().requires_grad_(True)
    q1_b = agent_b.critic1(observations, actions_b)
    q2_b = agent_b.critic2(observations, actions_b)
    q_b = torch.min(q1_b, q2_b).sum()
    grad_b = torch.autograd.grad(q_b, actions_b, create_graph=False)[0]

    grad_a_flat = grad_a.detach().reshape(-1)
    grad_b_flat = grad_b.detach().reshape(-1)
    cos = torch.nn.functional.cosine_similarity(
        grad_a_flat.unsqueeze(0), grad_b_flat.unsqueeze(0)
    )
    return float(cos.item())


def compute_critic_disagreement(
    agent: SACAgent,
    observations: torch.Tensor,
    actions: torch.Tensor,
) -> float:
    """Mean |Q1 - Q2| as uncertainty estimate."""
    with torch.no_grad():
        q1 = agent.critic1(observations, actions)
        q2 = agent.critic2(observations, actions)
    return float((q1 - q2).abs().mean().item())


def create_fresh_agent(reference_agent: SACAgent) -> SACAgent:
    """Create a randomly initialized agent with same architecture."""
    agent = SACAgent(
        observation_dim=reference_agent.observation_dim,
        action_dim=reference_agent.action_dim,
        action_low=[-1.0] * reference_agent.action_dim,
        action_high=[1.0] * reference_agent.action_dim,
        num_tasks=reference_agent.num_tasks,
        task_id_dim=reference_agent.task_id_dim,
        learning_rate=reference_agent.learning_rate,
        gamma=reference_agent.gamma,
        polyak=reference_agent.polyak,
        device=str(reference_agent.device),
        hide_task_id=reference_agent.hide_task_id,
        gradient_clip_norm=reference_agent.gradient_clip_norm,
    )
    return agent


def load_agent_from_checkpoint(checkpoint_path: Path, device: str) -> SACAgent:
    """Load checkpoint into a fresh agent."""
    payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    config = payload["agent_config"]
    agent = SACAgent(
        observation_dim=config["observation_dim"],
        action_dim=config["action_dim"],
        action_low=[-1.0] * config["action_dim"],
        action_high=[1.0] * config["action_dim"],
        num_tasks=config["num_tasks"],
        task_id_dim=config["task_id_dim"],
        learning_rate=config["learning_rate"],
        gamma=config["gamma"],
        polyak=config["polyak"],
        device=device,
        hide_task_id=True,
    )
    agent.load_state_dict(payload["agent_state_dict"], strict=True)
    return agent


def set_critic_from_source(target: SACAgent, source: SACAgent) -> None:
    """Copy critic (and target critic) weights from source to target."""
    target.critic1.load_state_dict(source.critic1.state_dict())
    target.critic2.load_state_dict(source.critic2.state_dict())
    target.target_critic1.load_state_dict(source.target_critic1.state_dict())
    target.target_critic2.load_state_dict(source.target_critic2.state_dict())


def main() -> None:
    args = parse_args()
    checkpoint_dir = Path(args.checkpoint_dir)
    tasks = get_cw10_tasks(args.env_version)
    target_tasks = args.target_tasks or list(range(1, 10))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    results = []

    for target_idx in target_tasks:
        task_name = tasks[target_idx]
        print(f"\n{'='*60}")
        print(f"Probing task {target_idx}: {task_name}")
        print(f"{'='*60}")

        # Load the agent state just before this task started (end of previous task)
        prev_ckpt = checkpoint_dir / f"task_{target_idx - 1}.pt"
        if not prev_ckpt.exists():
            print(f"  Skipping: {prev_ckpt} not found")
            continue

        # Create env for target task
        env = make_cw_env(
            task_name=task_name,
            seed=args.seed + target_idx,
            max_episode_steps=args.max_episode_steps,
            append_task_id=True,
            env_version=args.env_version,
            reward_function_version=args.reward_function_version,
        )

        # Load baseline agent (state at end of task i-1, as ClonEx would use)
        baseline_agent = load_agent_from_checkpoint(prev_ckpt, args.device)

        # Collect probe data using actor from previous task's best head
        print(f"  Collecting {args.probe_transitions} probe transitions...")
        probe_buffer = collect_probe_data(
            env, baseline_agent, target_idx,
            args.probe_transitions, args.max_episode_steps,
        )

        # Get a batch for gradient alignment measurement
        align_batch = probe_buffer.sample(min(512, len(probe_buffer)), args.device)
        align_obs = align_batch.observations
        align_actions = align_batch.actions

        # Measure initial disagreement
        init_disagreement = compute_critic_disagreement(
            baseline_agent, align_obs, align_actions
        )

        task_results = {
            "target_task_index": target_idx,
            "target_task_name": task_name,
            "initial_critic_disagreement": init_disagreement,
            "sources": [],
        }

        # Test each source critic: previous, random, and all historical
        source_configs = []
        # Previous task critic (what ClonEx actually does)
        source_configs.append(("previous", target_idx - 1))
        # Random critic
        source_configs.append(("random", -1))
        # All historical
        for hist_idx in range(target_idx):
            if hist_idx != target_idx - 1:
                source_configs.append((f"historical_{hist_idx}", hist_idx))

        for source_label, source_idx in source_configs:
            print(f"  Testing source: {source_label}...", end=" ", flush=True)

            # Create agent copy with actor from prev checkpoint (keep actor fixed)
            probe_agent = copy.deepcopy(baseline_agent)

            if source_idx == -1:
                # Random critic
                random_agent = create_fresh_agent(baseline_agent)
                set_critic_from_source(probe_agent, random_agent)
            elif source_idx != target_idx - 1:
                # Historical critic
                hist_ckpt = checkpoint_dir / f"task_{source_idx}.pt"
                if not hist_ckpt.exists():
                    print("SKIP (no checkpoint)")
                    continue
                hist_agent = load_agent_from_checkpoint(hist_ckpt, args.device)
                set_critic_from_source(probe_agent, hist_agent)

            # Measure initial TD loss
            eval_batch = probe_buffer.sample(min(1024, len(probe_buffer)), args.device)
            initial_loss = compute_td_loss(probe_agent, eval_batch)

            # Run critic-only warm-up
            loss_curve = critic_only_updates(
                probe_agent, probe_buffer,
                num_updates=args.warmup_updates,
                batch_size=args.batch_size,
            )
            final_loss = loss_curve[-1] if loss_curve else initial_loss

            # Adaptability score: relative improvement
            adaptability = (initial_loss - final_loss) / (initial_loss + 1e-8)

            # Q-gradient alignment with adapted critic vs baseline actor direction
            grad_alignment = compute_action_gradient_alignment(
                probe_agent, baseline_agent, align_obs, align_actions
            )

            # Post-warmup disagreement
            post_disagreement = compute_critic_disagreement(
                probe_agent, align_obs, align_actions
            )

            source_result = {
                "source_label": source_label,
                "source_task_index": source_idx,
                "initial_td_loss": initial_loss,
                "final_td_loss": final_loss,
                "adaptability": adaptability,
                "grad_alignment": grad_alignment,
                "post_warmup_disagreement": post_disagreement,
                "loss_at_50": loss_curve[49] if len(loss_curve) >= 50 else None,
                "loss_at_100": loss_curve[99] if len(loss_curve) >= 100 else None,
            }
            task_results["sources"].append(source_result)
            print(
                f"loss {initial_loss:.3f}->{final_loss:.3f} "
                f"adapt={adaptability:.3f} "
                f"∇Q_align={grad_alignment:.3f}"
            )

        # Summary: rank sources by adaptability
        sources_sorted = sorted(
            task_results["sources"],
            key=lambda x: x["final_td_loss"],
        )
        best_source = sources_sorted[0]["source_label"] if sources_sorted else "none"
        prev_source = next(
            (s for s in task_results["sources"] if s["source_label"] == "previous"),
            None,
        )
        task_results["best_source"] = best_source
        task_results["previous_final_loss"] = prev_source["final_td_loss"] if prev_source else None
        task_results["best_final_loss"] = sources_sorted[0]["final_td_loss"] if sources_sorted else None

        print(f"\n  Best source: {best_source}")
        if prev_source and sources_sorted:
            gap = prev_source["final_td_loss"] - sources_sorted[0]["final_td_loss"]
            print(f"  Gap vs previous: {gap:.4f}")

        results.append(task_results)
        env.close()

    # Print summary table
    print(f"\n{'='*80}")
    print("SUMMARY: Critic Transfer Probe")
    print(f"{'='*80}")
    print(f"{'Task':<20} {'Best Source':<15} {'Prev Loss':<12} {'Best Loss':<12} {'Gap':<10} {'Random better?'}")
    print("-" * 80)
    for r in results:
        prev_loss = r.get("previous_final_loss", 0) or 0
        best_loss = r.get("best_final_loss", 0) or 0
        gap = prev_loss - best_loss
        random_src = next(
            (s for s in r["sources"] if s["source_label"] == "random"), None
        )
        random_better = ""
        if random_src and prev_loss > 0:
            random_better = "YES" if random_src["final_td_loss"] < prev_loss else "no"
        print(
            f"{r['target_task_name']:<20} {r['best_source']:<15} "
            f"{prev_loss:<12.4f} {best_loss:<12.4f} {gap:<10.4f} {random_better}"
        )

    # Save results
    output_path = args.output or str(
        checkpoint_dir.parent / "critic_transfer_probe.json"
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
