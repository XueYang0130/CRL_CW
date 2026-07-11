from scipy.special import softmax
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf

from continualworld.sac.replay_buffers import EpisodicMemory, ReplayBuffer
from continualworld.sac.sac import SAC


class Episodic_SAC(SAC):
    def __init__(
        self,
        episodic_mem_per_task: int = 0, episodic_batch_size: int = 0,
        regularize_critic: bool = False, cl_reg_coef: float = 0.,
        episodic_memory_from_buffer: bool = True,
        **vanilla_sac_kwargs
    ):
        """Episodic replay.

        Args:
          episodic_mem_per_task: Number of examples to keep in additional memory per task.
          episodic_batch_size: Minibatch size to compute additional loss.
        """
        super().__init__(**vanilla_sac_kwargs)

        self.episodic_mem_per_task = episodic_mem_per_task
        self.episodic_batch_size = episodic_batch_size
        self.regularize_critic = regularize_critic
        self.cl_reg_coef = cl_reg_coef
        self.episodic_memory_from_buffer = episodic_memory_from_buffer

        episodic_mem_size = self.episodic_mem_per_task * self.env.num_envs
        self.episodic_memory = EpisodicMemory(
            obs_dim=self.obs_dim, act_dim=self.act_dim,
            size=episodic_mem_size, save_targets=True,
        )

    def kl_divergence(self, first_mu, first_logstd, second_mu, second_logstd):
        eps = 1e-6
        first_var = (tf.exp(first_logstd) + eps) ** 2
        second_var = (tf.exp(second_logstd) + eps) ** 2

        logstd_term = (second_logstd - first_logstd)
        mu_term = (first_var + (first_mu - second_mu) ** 2) / (2 * second_var)

        return tf.reduce_sum(logstd_term + mu_term - 0.5, -1)

    def behavioral_cloning_gradients(
            self,
            obs: tf.Tensor,
            actions: tf.Tensor,
            target_actor_dists: tf.Tensor,
            target_critic1_preds: tf.Tensor,
            target_critic2_preds: tf.Tensor,
            current_task_idx: int):

        target_mu = target_actor_dists[:, :self.act_dim]
        target_logstd = target_actor_dists[:, self.act_dim:]

        with tf.GradientTape(persistent=True) as g:
            mu, logstd, _, _ = self.actor(obs)

            actor_loss_per_example = self.kl_divergence(target_mu, target_logstd, mu, logstd)

            # TODO: weighting
            actor_loss = tf.reduce_mean(actor_loss_per_example)
            actor_loss *= self.cl_reg_coef

            if self.regularize_critic:
                critic1_pred = self.critic1(obs, actions)
                critic2_pred = self.critic2(obs, actions)

                critic1_loss_per_example = (critic1_pred - target_critic1_preds) ** 2
                critic2_loss_per_example = (critic2_pred - target_critic2_preds) ** 2

                critic1_loss = tf.reduce_mean(critic1_loss_per_example)
                critic2_loss = tf.reduce_mean(critic2_loss_per_example)
                critic_loss = critic1_loss + critic2_loss
                critic_loss *= self.cl_reg_coef

        actor_gradients = g.gradient(actor_loss, self.actor.trainable_variables)

        if self.regularize_critic:
            critic_gradients = g.gradient(critic_loss, self.critic_variables)
        else:
            critic_gradients = None

        return actor_gradients, critic_gradients

    def adjust_gradients(
        self,
        actor_gradients: List[tf.Tensor],
        critic_gradients: List[tf.Tensor],
        alpha_gradient: List[tf.Tensor],
        current_task_idx: int,
        metrics: dict,
        episodic_batch: Dict[str, tf.Tensor] = None,
    ) -> Tuple[List[tf.Tensor], List[tf.Tensor], List[tf.Tensor]]:
        if current_task_idx > 0:
            ref_actor_gradients, ref_critic_gradients = self.behavioral_cloning_gradients(
                obs=episodic_batch["obs"],
                actions=episodic_batch["actions"],
                target_actor_dists=episodic_batch["actor_dists"],
                target_critic1_preds=episodic_batch["critic1_preds"],
                target_critic2_preds=episodic_batch["critic2_preds"],
                current_task_idx=current_task_idx,
            )

            final_actor_gradients = self.merge_gradients(actor_gradients, ref_actor_gradients)
            final_critic_gradients = self.merge_gradients(critic_gradients, ref_critic_gradients)
        else:
            final_actor_gradients = actor_gradients
            final_critic_gradients = critic_gradients

        return final_actor_gradients, final_critic_gradients, alpha_gradient

    def merge_gradients(self, new_grads: List[tf.Tensor], ref_grads: Optional[List[tf.Tensor]]):
        if ref_grads is None:
            return new_grads
        final_grads = []
        for new_grad, ref_grad in zip(new_grads, ref_grads):
            final_grads += [(new_grad + ref_grad) / 2]
        return final_grads

    def gather_buffer(self, task_idx):
        tmp_replay_buffer = ReplayBuffer(
            obs_dim=self.obs_dim, act_dim=self.act_dim, size=self.episodic_mem_per_task
        )

        obs = self.env.envs[task_idx].reset()
        episode_len = 0
        for step_idx in range(self.episodic_mem_per_task):
            action = self.get_action(tf.convert_to_tensor(obs), deterministic=False)
            next_obs, reward, done, info = self.env.envs[task_idx].step(action)

            episode_len += 1
            done_to_store = done
            if episode_len == self.max_episode_len or info.get("TimeLimit.truncated"):
                done_to_store = False
            tmp_replay_buffer.store(obs, action, reward, next_obs, done_to_store)

            if done:
                obs = self.env.envs[task_idx].reset()
                episode_len = 0
            else:
                obs = next_obs
        return tmp_replay_buffer.sample_batch(self.episodic_mem_per_task)

    def on_task_start(self, current_task_idx: int) -> None:
        if current_task_idx > 0:
            if self.episodic_memory_from_buffer:
                new_episodic_mem = self.replay_buffer.sample_batch(self.episodic_mem_per_task)
            else:
                new_episodic_mem = self.gather_buffer(current_task_idx - 1)

            mu, log_std, _, _ = self.actor(new_episodic_mem["obs"])
            critic1_preds = self.critic1(new_episodic_mem["obs"], new_episodic_mem["actions"])
            critic2_preds = self.critic2(new_episodic_mem["obs"], new_episodic_mem["actions"])

            new_episodic_mem = {k: v.numpy() for k, v in new_episodic_mem.items()}
            new_episodic_mem["actor_dists"] = np.concatenate([mu.numpy(), log_std.numpy()], -1)
            new_episodic_mem["critic1_preds"] = critic1_preds.numpy()
            new_episodic_mem["critic2_preds"] = critic2_preds.numpy()

            self.episodic_memory.store_multiple(**new_episodic_mem)

    def get_episodic_batch(self, current_task_idx: int) -> Optional[Dict[str, tf.Tensor]]:
        if current_task_idx > 0:
            return self.episodic_memory.sample_batch(self.episodic_batch_size)
        return None
