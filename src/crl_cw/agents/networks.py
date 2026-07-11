"""Neural networks for the Soft Actor-Critic agent.

This module is a PyTorch implementation of the basic single-head actor
and critic architecture used by the ClonEx-SAC reference code.

"""

from __future__ import annotations

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
LOG_PROB_EPSILON = 1e-6


class ActorOutput(NamedTuple):
    """Outputs produced when sampling from the SAC actor."""

    mean_action: torch.Tensor
    sampled_action: torch.Tensor
    log_probability: torch.Tensor
    log_std: torch.Tensor


def initialize_linear_layer(layer: nn.Linear) -> None:
    """Initialize a linear layer like TensorFlow/Keras Dense.

    The ClonEx-SAC implementation uses Keras Dense layers, whose default
    kernel initializer is Glorot/Xavier uniform and whose bias initializer
    is zero.
    """
    nn.init.xavier_uniform_(layer.weight)
    nn.init.zeros_(layer.bias)


class ClonExMLP(nn.Module):
    """Shared MLP architecture matching the ClonEx-SAC implementation.

    When layer normalization is enabled, the original code applies:

        Dense -> LayerNorm -> tanh

    only to the first hidden layer. All remaining hidden layers use the
    configured activation directly after their Dense layer.

    For the ClonEx-SAC baseline, that activation is Leaky ReLU.
    """

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

        self.input_dim = input_dim
        self.hidden_sizes = tuple(hidden_sizes)
        self.use_layer_norm = use_layer_norm

        self.hidden_layers = nn.ModuleList()

        previous_dim = input_dim
        for hidden_dim in self.hidden_sizes:
            layer = nn.Linear(previous_dim, hidden_dim)
            initialize_linear_layer(layer)
            self.hidden_layers.append(layer)
            previous_dim = hidden_dim

        self.first_layer_norm: nn.LayerNorm | None

        if use_layer_norm:
            self.first_layer_norm = nn.LayerNorm(self.hidden_sizes[0])
        else:
            self.first_layer_norm = None

        self.leaky_relu = nn.LeakyReLU(
            negative_slope=LEAKY_RELU_SLOPE,
        )

    @property
    def output_dim(self) -> int:
        """Return the size of the final hidden representation."""
        return self.hidden_sizes[-1]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Encode a batch of input vectors."""
        if inputs.ndim != 2:
            raise ValueError(
                "ClonExMLP expects a rank-2 tensor with shape "
                f"(batch_size, input_dim), received {tuple(inputs.shape)}."
            )

        if inputs.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected input dimension {self.input_dim}, "
                f"received {inputs.shape[-1]}."
            )

        x = inputs

        for layer_index, layer in enumerate(self.hidden_layers):
            x = layer(x)

            if layer_index == 0 and self.first_layer_norm is not None:
                # This slightly unusual tanh after the first LayerNorm
                # exactly follows the ClonEx-SAC reference MLP.
                x = self.first_layer_norm(x)
                x = torch.tanh(x)
            else:
                x = self.leaky_relu(x)

        return x


class GaussianActor(nn.Module):
    """Squashed Gaussian actor used by SAC.

    The network produces a Gaussian mean and log standard deviation.
    Actions are sampled with the reparameterization trick, passed through
    tanh, and scaled to the environment action range.
    """

    def __init__(
        self,
        observation_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        hidden_sizes: Sequence[int] = DEFAULT_HIDDEN_SIZES,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()

        if observation_dim <= 0:
            raise ValueError("observation_dim must be positive.")

        action_low_array = np.asarray(action_low, dtype=np.float32)
        action_high_array = np.asarray(action_high, dtype=np.float32)

        if action_low_array.ndim != 1:
            raise ValueError("action_low must be a one-dimensional array.")

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

        self.observation_dim = observation_dim
        self.action_dim = int(action_low_array.shape[0])

        self.backbone = ClonExMLP(
            input_dim=observation_dim,
            hidden_sizes=hidden_sizes,
            use_layer_norm=use_layer_norm,
        )

        self.mean_head = nn.Linear(
            self.backbone.output_dim,
            self.action_dim,
        )
        self.log_std_head = nn.Linear(
            self.backbone.output_dim,
            self.action_dim,
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

    def distribution_parameters(
        self,
        observations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the Gaussian mean and bounded log standard deviation."""
        features = self.backbone(observations)

        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
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
        """Sample differentiable actions and calculate their log probability."""
        mean, log_std = self.distribution_parameters(observations)
        std = log_std.exp()

        distribution = Normal(mean, std)

        # rsample() uses the reparameterization trick required for the
        # SAC actor gradient.
        pre_tanh_action = distribution.rsample()

        squashed_sample = torch.tanh(pre_tanh_action)
        squashed_mean = torch.tanh(mean)

        sampled_action = (
            squashed_sample * self.action_scale
            + self.action_bias
        )
        mean_action = (
            squashed_mean * self.action_scale
            + self.action_bias
        )

        # Gaussian log probability before tanh transformation.
        log_probability = distribution.log_prob(pre_tanh_action)

        # Correct the density for tanh and action scaling.
        correction = torch.log(
            self.action_scale
            * (1.0 - squashed_sample.pow(2))
            + LOG_PROB_EPSILON
        )

        log_probability = log_probability - correction
        log_probability = log_probability.sum(
            dim=-1,
            keepdim=True,
        )

        return ActorOutput(
            mean_action=mean_action,
            sampled_action=sampled_action,
            log_probability=log_probability,
            log_std=log_std,
        )

    @torch.no_grad()
    def act(
        self,
        observation: np.ndarray,
        deterministic: bool,
        device: torch.device | str,
    ) -> np.ndarray:
        """Return one NumPy action for environment interaction."""
        observation_array = np.asarray(
            observation,
            dtype=np.float32,
        )

        if observation_array.shape != (self.observation_dim,):
            raise ValueError(
                "Invalid observation shape: expected "
                f"{(self.observation_dim,)}, "
                f"received {observation_array.shape}."
            )

        observation_tensor = torch.as_tensor(
            observation_array,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)

        actor_output = self.sample(observation_tensor)

        if deterministic:
            action = actor_output.mean_action
        else:
            action = actor_output.sampled_action

        return action.squeeze(0).cpu().numpy()


class QCritic(nn.Module):
    """Single Q-function for SAC.

    SAC will later instantiate two independent QCritic networks to form
    the twin-critic architecture.
    """

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int] = DEFAULT_HIDDEN_SIZES,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()

        if observation_dim <= 0:
            raise ValueError("observation_dim must be positive.")

        if action_dim <= 0:
            raise ValueError("action_dim must be positive.")

        self.observation_dim = observation_dim
        self.action_dim = action_dim

        self.backbone = ClonExMLP(
            input_dim=observation_dim + action_dim,
            hidden_sizes=hidden_sizes,
            use_layer_norm=use_layer_norm,
        )

        self.q_head = nn.Linear(
            self.backbone.output_dim,
            1,
        )
        initialize_linear_layer(self.q_head)

    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Estimate Q(s, a) for a batch of transitions."""
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

        inputs = torch.cat(
            (observations, actions),
            dim=-1,
        )

        features = self.backbone(inputs)
        q_values = self.q_head(features)

        return q_values