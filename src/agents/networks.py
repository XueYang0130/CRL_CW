from __future__ import annotations

import math
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

DEFAULT_HIDDEN_SIZES: tuple[int, ...] = (256, 256, 256, 256)
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0
LEAKY_RELU_SLOPE = 0.2
LAYER_NORM_EPSILON = 1e-3


class ActorOutput(NamedTuple):
    """Outputs produced when sampling from the SAC actor."""

    mean_action: torch.Tensor
    sampled_action: torch.Tensor
    log_probability: torch.Tensor
    log_std: torch.Tensor


def initialize_linear_layer(layer: nn.Linear) -> None:
    """Initialize a linear layer like TensorFlow/Keras Dense."""
    nn.init.xavier_uniform_(layer.weight)
    nn.init.zeros_(layer.bias)


def choose_task_head(
    all_head_outputs: torch.Tensor,
    task_one_hot: torch.Tensor,
    num_heads: int,
) -> torch.Tensor:
    """Choose the output head indicated by a task one-hot vector.

    Args:
        all_head_outputs:
            Tensor with shape ``(batch_size, output_dim * num_heads)``.
        task_one_hot:
            Tensor with shape ``(batch_size, num_heads)``.
        num_heads:
            Number of task-specific heads.

    Returns:
        Tensor with shape ``(batch_size, output_dim)``.

    This follows the reference implementation's reshape:

        (batch, output_dim * heads) -> (batch, output_dim, heads)

    followed by multiplication with the task one-hot vector.
    """
    if all_head_outputs.ndim != 2:
        raise ValueError("all_head_outputs must be rank 2.")
    if task_one_hot.ndim != 2:
        raise ValueError("task_one_hot must be rank 2.")
    if task_one_hot.shape[1] != num_heads:
        raise ValueError(
            f"Expected task one-hot dimension {num_heads}, "
            f"received {task_one_hot.shape[1]}."
        )
    if all_head_outputs.shape[0] != task_one_hot.shape[0]:
        raise ValueError(
            "Head outputs and task one-hot must share batch size."
        )
    if all_head_outputs.shape[1] % num_heads != 0:
        raise ValueError(
            "Output dimension must be divisible by num_heads."
        )

    batch_size = all_head_outputs.shape[0]
    reshaped = all_head_outputs.reshape(
        batch_size,
        -1,
        num_heads,
    )
    return torch.einsum(
        "boh,bh->bo",
        reshaped,
        task_one_hot,
    )


class SharedMLP(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: Sequence[int] = DEFAULT_HIDDEN_SIZES,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()

        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if not hidden_sizes:
            raise ValueError("hidden_sizes must contain at least one layer.")
        if any(size <= 0 for size in hidden_sizes):
            raise ValueError("All hidden layer sizes must be positive.")

        self.input_dim = int(input_dim)
        self.hidden_sizes = tuple(int(size) for size in hidden_sizes)

        self.hidden_layers = nn.ModuleList()
        previous_dim = self.input_dim

        for hidden_dim in self.hidden_sizes:
            layer = nn.Linear(previous_dim, hidden_dim)
            initialize_linear_layer(layer)
            self.hidden_layers.append(layer)
            previous_dim = hidden_dim

        self.first_layer_norm: nn.LayerNorm | None
        if use_layer_norm:
            self.first_layer_norm = nn.LayerNorm(
                self.hidden_sizes[0],
                eps=LAYER_NORM_EPSILON,
            )
        else:
            self.first_layer_norm = None

        self.leaky_relu = nn.LeakyReLU(
            negative_slope=LEAKY_RELU_SLOPE,
        )

    @property
    def output_dim(self) -> int:
        """Return the final shared representation size."""
        return self.hidden_sizes[-1]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Encode a batch of vectors."""
        if inputs.ndim != 2:
            raise ValueError("SharedMLP expects shape (batch_size, input_dim).")
        if inputs.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected input dimension {self.input_dim}, "
                f"received {inputs.shape[-1]}."
            )

        x = inputs
        for layer_index, layer in enumerate(self.hidden_layers):
            x = layer(x)

            if layer_index == 0 and self.first_layer_norm is not None:
                x = self.first_layer_norm(x)
                x = torch.tanh(x)
            else:
                x = self.leaky_relu(x)

        return x


class GaussianActor(nn.Module):
    """Multi-head squashed-Gaussian actor."""

    def __init__(
        self,
        observation_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        hidden_sizes: Sequence[int] = DEFAULT_HIDDEN_SIZES,
        use_layer_norm: bool = True,
        task_id_dim: int = 0,
        num_heads: int = 1,
        hide_task_id: bool = False,
    ) -> None:
        super().__init__()

        if observation_dim <= 0:
            raise ValueError("observation_dim must be positive.")
        if task_id_dim < 0:
            raise ValueError("task_id_dim must be non-negative.")
        if task_id_dim >= observation_dim:
            raise ValueError(
                "task_id_dim must be smaller than observation_dim."
            )
        if num_heads <= 0:
            raise ValueError("num_heads must be positive.")
        if num_heads > 1 and task_id_dim != num_heads:
            raise ValueError(
                "For multi-head actor, task_id_dim must equal num_heads."
            )

        action_low_array = np.asarray(action_low, dtype=np.float32)
        action_high_array = np.asarray(action_high, dtype=np.float32)

        if action_low_array.ndim != 1:
            raise ValueError("action_low must be one-dimensional.")
        if action_high_array.shape != action_low_array.shape:
            raise ValueError(
                "action_low and action_high must have identical shapes."
            )
        if not np.all(np.isfinite(action_low_array)):
            raise ValueError("action_low must contain finite values.")
        if not np.all(np.isfinite(action_high_array)):
            raise ValueError("action_high must contain finite values.")
        if not np.all(action_high_array > action_low_array):
            raise ValueError(
                "Every action_high value must be greater than action_low."
            )

        self.observation_dim = int(observation_dim)
        self.task_id_dim = int(task_id_dim)
        self.num_heads = int(num_heads)
        self.action_dim = int(action_low_array.shape[0])
        self.hide_task_id = bool(hide_task_id)
        backbone_input_dim = (
            self.observation_dim - self.task_id_dim
            if self.hide_task_id
            else self.observation_dim
        )

        self.backbone = SharedMLP(
            input_dim=backbone_input_dim,
            hidden_sizes=hidden_sizes,
            use_layer_norm=use_layer_norm,
        )

        self.mean_head = nn.Linear(
            self.backbone.output_dim,
            self.action_dim * self.num_heads,
        )
        self.log_std_head = nn.Linear(
            self.backbone.output_dim,
            self.action_dim * self.num_heads,
        )
        initialize_linear_layer(self.mean_head)
        initialize_linear_layer(self.log_std_head)

        action_scale = (action_high_array - action_low_array) / 2.0
        action_bias = (action_high_array + action_low_array) / 2.0

        self.register_buffer(
            "action_scale",
            torch.as_tensor(action_scale, dtype=torch.float32),
        )
        self.register_buffer(
            "action_bias",
            torch.as_tensor(action_bias, dtype=torch.float32),
        )

    def _split_observation(
        self,
        observations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if observations.ndim != 2:
            raise ValueError("observations must be a rank-2 tensor.")
        if observations.shape[-1] != self.observation_dim:
            raise ValueError(
                f"Expected observation dimension {self.observation_dim}, "
                f"received {observations.shape[-1]}."
            )

        if self.task_id_dim == 0:
            return observations, None

        if self.hide_task_id:
            features_input = observations[:, :-self.task_id_dim]
        else:
            features_input = observations
        task_one_hot = observations[:, -self.task_id_dim:]
        return features_input, task_one_hot

    def distribution_parameters(
        self,
        observations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the active head's Gaussian parameters."""
        physical_observations, task_one_hot = (
            self._split_observation(observations)
        )
        features = self.backbone(physical_observations)

        all_means = self.mean_head(features)
        all_log_stds = self.log_std_head(features)

        if self.num_heads > 1:
            if task_one_hot is None:
                raise RuntimeError(
                    "Multi-head actor requires a task one-hot vector."
                )
            mean = choose_task_head(
                all_means,
                task_one_hot,
                self.num_heads,
            )
            log_std = choose_task_head(
                all_log_stds,
                task_one_hot,
                self.num_heads,
            )
        else:
            mean = all_means
            log_std = all_log_stds

        log_std = torch.clamp(
            log_std,
            min=LOG_STD_MIN,
            max=LOG_STD_MAX,
        )
        return mean, log_std

    def sample(
        self,
        observations: torch.Tensor,
    ) -> ActorOutput:
        """Sample differentiable actions and calculate log probability."""
        mean, log_std = self.distribution_parameters(observations)
        distribution = Normal(mean, log_std.exp())

        pre_tanh_action = distribution.rsample()
        squashed_sample = torch.tanh(pre_tanh_action)
        squashed_mean = torch.tanh(mean)

        sampled_action = (
            squashed_sample * self.action_scale + self.action_bias
        )
        mean_action = (
            squashed_mean * self.action_scale + self.action_bias
        )

        gaussian_log_probability = distribution.log_prob(
            pre_tanh_action
        )
        tanh_correction = 2.0 * (
            math.log(2.0)
            - pre_tanh_action
            - torch.nn.functional.softplus(-2.0 * pre_tanh_action)
        )

        log_probability = (
            gaussian_log_probability
            - tanh_correction
            - torch.log(self.action_scale)
        ).sum(dim=-1, keepdim=True)


        return ActorOutput(
            mean_action=mean_action,
            sampled_action=sampled_action,
            log_probability=log_probability,
            log_std=log_std,
        )

    def deterministic_action(
        self,
        observations: torch.Tensor,
    ) -> torch.Tensor:
        """Return Gaussian mean actions without sampling any noise."""
        mean, _ = self.distribution_parameters(observations)
        squashed_mean = torch.tanh(mean)

        return (
            squashed_mean * self.action_scale
            + self.action_bias
        )

    @torch.no_grad()
    def act(
        self,
        observation: np.ndarray,
        deterministic: bool,
        device: torch.device | str,
    ) -> np.ndarray:
        """Return one action for environment interaction."""
        observation_array = np.asarray(
            observation,
            dtype=np.float32,
        )
        expected_shape = (self.observation_dim,)

        if observation_array.shape != expected_shape:
            raise ValueError(
                f"Expected observation shape {expected_shape}, "
                f"received {observation_array.shape}."
            )

        observation_tensor = torch.as_tensor(
            observation_array,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)

        if deterministic:
            action = self.deterministic_action(
                observation_tensor
            )
        else:
            action = self.sample(
                observation_tensor
            ).sampled_action

        return action.squeeze(0).cpu().numpy()


class QCritic(nn.Module):
    """Multi-head Q-function used as one SAC critic."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int] = DEFAULT_HIDDEN_SIZES,
        use_layer_norm: bool = True,
        task_id_dim: int = 0,
        num_heads: int = 1,
        hide_task_id: bool = False,
    ) -> None:
        super().__init__()

        if observation_dim <= 0:
            raise ValueError("observation_dim must be positive.")
        if action_dim <= 0:
            raise ValueError("action_dim must be positive.")
        if task_id_dim < 0:
            raise ValueError("task_id_dim must be non-negative.")
        if task_id_dim >= observation_dim:
            raise ValueError(
                "task_id_dim must be smaller than observation_dim."
            )
        if num_heads <= 0:
            raise ValueError("num_heads must be positive.")
        if num_heads > 1 and task_id_dim != num_heads:
            raise ValueError(
                "For multi-head critic, task_id_dim must equal num_heads."
            )

        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self.task_id_dim = int(task_id_dim)
        self.num_heads = int(num_heads)
        self.hide_task_id = bool(hide_task_id)
        backbone_observation_dim = (
            self.observation_dim - self.task_id_dim
            if self.hide_task_id
            else self.observation_dim
        )

        self.backbone = SharedMLP(
            input_dim=backbone_observation_dim + self.action_dim,
            hidden_sizes=hidden_sizes,
            use_layer_norm=use_layer_norm,
        )

        self.q_head = nn.Linear(
            self.backbone.output_dim,
            self.num_heads,
        )
        initialize_linear_layer(self.q_head)

    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Estimate the active task head's Q(s, a)."""
        if observations.ndim != 2:
            raise ValueError("observations must be a rank-2 tensor.")
        if actions.ndim != 2:
            raise ValueError("actions must be a rank-2 tensor.")
        if observations.shape[0] != actions.shape[0]:
            raise ValueError(
                "observations and actions must have the same batch size."
            )
        if observations.shape[-1] != self.observation_dim:
            raise ValueError(
                f"Expected observation dimension {self.observation_dim}, "
                f"received {observations.shape[-1]}."
            )
        if actions.shape[-1] != self.action_dim:
            raise ValueError(
                f"Expected action dimension {self.action_dim}, "
                f"received {actions.shape[-1]}."
            )

        if self.task_id_dim == 0:
            backbone_observations = observations
            task_one_hot = None
        else:
            if self.hide_task_id:
                backbone_observations = observations[:, :-self.task_id_dim]
            else:
                backbone_observations = observations
            task_one_hot = observations[:, -self.task_id_dim:]

        features = self.backbone(
            torch.cat(
                (backbone_observations, actions),
                dim=-1,
            )
        )
        all_q_values = self.q_head(features)

        if self.num_heads > 1:
            if task_one_hot is None:
                raise RuntimeError(
                    "Multi-head critic requires a task one-hot vector."
                )
            q_values = choose_task_head(
                all_q_values,
                task_one_hot,
                self.num_heads,
            )
        else:
            q_values = all_q_values

        return q_values


class DeepHeadQCritic(nn.Module):
    """Multi-head Q-function with a nonlinear tower for each task."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int] = DEFAULT_HIDDEN_SIZES,
        head_hidden_size: int = 64,
        use_layer_norm: bool = True,
        task_id_dim: int = 0,
        num_heads: int = 1,
        hide_task_id: bool = False,
    ) -> None:
        super().__init__()
        if observation_dim <= 0:
            raise ValueError("observation_dim must be positive.")
        if action_dim <= 0:
            raise ValueError("action_dim must be positive.")
        if head_hidden_size <= 0:
            raise ValueError("head_hidden_size must be positive.")
        if task_id_dim < 0 or task_id_dim >= observation_dim:
            raise ValueError("task_id_dim must be in [0, observation_dim).")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive.")
        if num_heads > 1 and task_id_dim != num_heads:
            raise ValueError(
                "For multi-head critic, task_id_dim must equal num_heads."
            )

        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self.task_id_dim = int(task_id_dim)
        self.num_heads = int(num_heads)
        self.hide_task_id = bool(hide_task_id)
        self.head_hidden_size = int(head_hidden_size)
        backbone_observation_dim = (
            self.observation_dim - self.task_id_dim
            if self.hide_task_id
            else self.observation_dim
        )
        self.backbone = SharedMLP(
            input_dim=backbone_observation_dim + self.action_dim,
            hidden_sizes=hidden_sizes,
            use_layer_norm=use_layer_norm,
        )
        self.q_heads = nn.ModuleList(
            self._make_head(self.backbone.output_dim) for _ in range(self.num_heads)
        )

    def _make_head(self, input_dim: int) -> nn.Sequential:
        hidden = nn.Linear(input_dim, self.head_hidden_size)
        output = nn.Linear(self.head_hidden_size, 1)
        initialize_linear_layer(hidden)
        initialize_linear_layer(output)
        return nn.Sequential(hidden, nn.LeakyReLU(LEAKY_RELU_SLOPE), output)

    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        if observations.ndim != 2:
            raise ValueError("observations must be a rank-2 tensor.")
        if actions.ndim != 2:
            raise ValueError("actions must be a rank-2 tensor.")
        if observations.shape[0] != actions.shape[0]:
            raise ValueError("observations and actions must have the same batch size.")
        if observations.shape[-1] != self.observation_dim:
            raise ValueError(
                f"Expected observation dimension {self.observation_dim}, "
                f"received {observations.shape[-1]}."
            )
        if actions.shape[-1] != self.action_dim:
            raise ValueError(
                f"Expected action dimension {self.action_dim}, "
                f"received {actions.shape[-1]}."
            )

        if self.task_id_dim == 0:
            backbone_observations = observations
            task_indices = torch.zeros(
                observations.shape[0], dtype=torch.long, device=observations.device
            )
        else:
            task_one_hot = observations[:, -self.task_id_dim:]
            backbone_observations = (
                observations[:, :-self.task_id_dim]
                if self.hide_task_id
                else observations
            )
            task_indices = task_one_hot.argmax(dim=1)

        features = self.backbone(torch.cat((backbone_observations, actions), dim=-1))
        q_values = features.new_zeros((features.shape[0], 1))
        # Continual replay batches normally contain one task. Grouped dispatch also
        # keeps mixed-task batches correct without evaluating every task tower.
        for task_index_tensor in task_indices.unique():
            task_index = int(task_index_tensor.item())
            mask = task_indices == task_index_tensor
            q_values[mask] = self.q_heads[task_index](features[mask])

        # SAC requests gradients for every critic parameter in one call. Keep
        # inactive towers in that graph with exactly zero gradients, without
        # paying for their forward passes.
        zero_gradient_anchor = features.new_zeros(())
        for parameter in self.q_heads.parameters():
            zero_gradient_anchor = zero_gradient_anchor + parameter.reshape(-1)[0]
        return q_values + zero_gradient_anchor * 0.0
