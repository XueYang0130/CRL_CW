import random
from typing import Dict, Optional

import numpy as np
import tensorflow as tf

from continualworld.envs import MW_OBS_LEN


class ReplayBuffer:
    """A simple FIFO experience replay buffer for SAC agents."""

    def __init__(self, obs_dim: int, act_dim: int, size: int) -> None:
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.actions_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rewards_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(
        self, obs: np.ndarray, action: np.ndarray, reward: float, next_obs: np.ndarray, done: bool
    ) -> None:
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.actions_buf[self.ptr] = action
        self.rewards_buf[self.ptr] = reward
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size: int) -> Dict[str, tf.Tensor]:
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(
            obs=tf.convert_to_tensor(self.obs_buf[idxs]),
            next_obs=tf.convert_to_tensor(self.next_obs_buf[idxs]),
            actions=tf.convert_to_tensor(self.actions_buf[idxs]),
            rewards=tf.convert_to_tensor(self.rewards_buf[idxs]),
            done=tf.convert_to_tensor(self.done_buf[idxs]),
        )


class EpisodicMemory:
    """Buffer which does not support overwriting old samples."""

    def __init__(self, obs_dim: int, act_dim: int, size: int, save_targets: bool = False) -> None:
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.actions_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rewards_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.save_targets = save_targets
        self.task_id_buf = np.zeros([size], dtype=int)
        if self.save_targets:
            self.actor_dist_buf = np.zeros([size, act_dim * 2], dtype=np.float32)
            self.critic1_pred_buf = np.zeros([size], dtype=np.float32)
            self.critic2_pred_buf = np.zeros([size], dtype=np.float32)
        self.size, self.max_size = 0, size

    def store_multiple(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_obs: np.ndarray,
        done: np.ndarray,
        actor_dists: Optional[np.ndarray] = None,
        critic1_preds: Optional[np.ndarray] = None,
        critic2_preds: Optional[np.ndarray] = None,
    ) -> None:
        assert len(obs) == len(actions) == len(rewards) == len(next_obs) == len(done)
        assert self.size + len(obs) <= self.max_size

        range_start = self.size
        range_end = self.size + len(obs)
        self.obs_buf[range_start:range_end] = obs
        self.next_obs_buf[range_start:range_end] = next_obs
        self.actions_buf[range_start:range_end] = actions
        self.rewards_buf[range_start:range_end] = rewards
        self.done_buf[range_start:range_end] = done
        self.task_id_buf[range_start:range_end] = obs[:, MW_OBS_LEN:].argmax(-1)
        if self.save_targets:
            assert ((actor_dists is not None)
                    and (critic1_preds is not None)
                    and (critic2_preds is not None))
            self.actor_dist_buf[range_start:range_end] = actor_dists
            self.critic1_pred_buf[range_start:range_end] = critic1_preds
            self.critic2_pred_buf[range_start:range_end] = critic2_preds
        self.size = self.size + len(obs)

    def sample_batch(self, batch_size: int, task_weights: Optional[np.ndarray] = None) -> Dict[str, tf.Tensor]:
        batch_size = min(batch_size, self.size)
        if task_weights is not None:
            task_ids = self.task_id_buf[:self.size]
            example_weights = task_weights[task_ids]
            example_weights /= example_weights.sum()
            idxs = np.random.choice(self.size, size=batch_size, replace=False, p=example_weights)
        else:
            idxs = np.random.choice(self.size, size=batch_size, replace=False)

        batch_dict = dict(
            obs=tf.convert_to_tensor(self.obs_buf[idxs]),
            next_obs=tf.convert_to_tensor(self.next_obs_buf[idxs]),
            actions=tf.convert_to_tensor(self.actions_buf[idxs]),
            rewards=tf.convert_to_tensor(self.rewards_buf[idxs]),
            done=tf.convert_to_tensor(self.done_buf[idxs]),
        )

        if self.save_targets:
            batch_dict["actor_dists"] = tf.convert_to_tensor(self.actor_dist_buf[idxs])
            batch_dict["critic1_preds"] = tf.convert_to_tensor(self.critic1_pred_buf[idxs])
            batch_dict["critic2_preds"] = tf.convert_to_tensor(self.critic2_pred_buf[idxs])

        return batch_dict

class ReservoirReplayBuffer(ReplayBuffer):
    """Buffer for SAC agents implementing reservoir sampling."""

    def __init__(self, obs_dim: int, act_dim: int, size: int) -> None:
        super().__init__(obs_dim, act_dim, size)
        self.timestep = 0

    def store(
        self, obs: np.ndarray, action: np.ndarray, reward: float, next_obs: np.ndarray, done: bool
    ) -> None:
        current_t = self.timestep
        self.timestep += 1

        if current_t < self.max_size:
            buffer_idx = current_t
        else:
            buffer_idx = random.randint(0, current_t)
            if buffer_idx >= self.max_size:
                return

        self.obs_buf[buffer_idx] = obs
        self.next_obs_buf[buffer_idx] = next_obs
        self.actions_buf[buffer_idx] = action
        self.rewards_buf[buffer_idx] = reward
        self.done_buf[buffer_idx] = done
        self.size = min(self.size + 1, self.max_size)
