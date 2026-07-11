import random

import numpy as np
import scipy.special
import tensorflow as tf


class ExplorationHelper:
    def __init__(self, kind, num_available_heads):
        self.kind = kind
        self.num_available_heads = num_available_heads
        self.current_head_id = None
        self.current_rewards = []
        self.current_successes = []
        self.episode_returns = [[] for _ in range(self.num_available_heads)]
        self.episode_successes = [[] for _ in range(self.num_available_heads)]

    def tell_results(self, reward, success):
        # Pass relevant info from SAC algorithm after a step.
        assert self.current_head_id is not None
        self.current_rewards.append(reward)
        self.current_successes.append(success)

    def _get_one_hot(self, x):
        return tf.one_hot(x, self.num_available_heads).numpy()

    def select(self, head):
        self.current_head_id = head
        return self._get_one_hot(head)

    def get_exploration_head_one_hot(self):
        assert (self.current_head_id is None) == (len(self.current_rewards) == 0)

        if self.current_head_id is not None:
            # Previous exploration trajectory has finished, collect statistics
            self.episode_returns[self.current_head_id].append(sum(self.current_rewards))
            self.episode_successes[self.current_head_id].append(
                bool(np.any(self.current_successes))
            )
            self.current_rewards = []
            self.current_successes = []

        if self.kind == "current":
            return self.select(self.num_available_heads - 1)

        if self.kind == "previous":
            return self.select(self.num_available_heads - 2)

        if self.kind == "uniform_previous":
            return self.select(random.randint(0, self.num_available_heads - 2))

        if self.kind == "uniform_previous_or_current":
            return self.select(random.randint(0, self.num_available_heads - 1))

        # For other strategies: if some previous head is unused, return it
        for i in range(self.num_available_heads - 1):
            if len(self.episode_returns[i]) == 0:
                return self.select(i)

        if self.kind in ["best_success", "best_return"]:
            scores = []
            for i in range(self.num_available_heads - 1):
                if self.kind == "best_success":
                    score = float(np.mean(self.episode_successes[i]))
                elif self.kind == "best_return":
                    score = float(np.mean(self.episode_returns[i]))
                else:
                    assert False, "bad exploration kind!"
                scores.append(score)
            chosen = int(np.argmax(scores))
            return self.select(chosen)

        # TODO: clean up (separate parameters?)
        if self.kind.startswith("softmax_return_"):
            temperature = float(self.kind[len("softmax_return_") :])
            scores = np.array(
                [np.mean(self.episode_returns[i]) for i in range(self.num_available_heads - 1)]
            )

            # Normalize the scores to be within [0, 1]
            min_val, max_val = scores.min(), scores.max()
            scores = (scores - min_val) / (max_val - min_val + 1e-6)

            probs = scipy.special.softmax(scores / temperature)
            chosen = int(np.random.choice(range(self.num_available_heads - 1), p=probs))
            return self.select(chosen)

        assert False, "bad exploration kind!"

    @staticmethod
    def check_kind(kind):
        whole_names = [
            None,
            "current",
            "previous",
            "uniform_previous",
            "uniform_previous_or_current",
            "best_success",
            "best_return",
        ]
        assert kind in whole_names or kind.startswith("softmax_return_")
