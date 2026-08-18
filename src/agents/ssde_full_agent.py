"""PyTorch SSDE reproduction adapted to the local continual SAC protocol.

The structural sparsity and dormant-exploration mechanisms follow the
official SSDE implementation. Environment handling, critics, replay, and
evaluation remain local so comparisons use the same CW10-v3 protocol.
"""

from __future__ import annotations

import copy
import math
from collections import deque
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from agents.networks import ActorOutput, initialize_linear_layer
from agents.sac_agent import SACAgent


_SENTENCE_ENCODER_CACHE: Any | None = None


CW10_TASK_DESCRIPTIONS: tuple[str, ...] = (
    "Hammer a screw on the wall.",
    "Bypass a wall and push a puck to a goal.",
    "Rotate the faucet clockwise.",
    "Pull a puck to a goal.",
    "Grasp a stick and pull a box with the stick.",
    "Press a handle down sideways.",
    "Push the puck to a goal.",
    "Pick and place a puck onto a shelf.",
    "Push and close a window.",
    "Unplug a peg sideways.",
)


class _StraightThroughStep(torch.autograd.Function):
    """Official SSDE Heaviside forward with clipped STE backward."""

    @staticmethod
    def forward(ctx, values: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(values)
        return (values > 0.0).to(values.dtype)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor) -> tuple[torch.Tensor]:
        (values,) = ctx.saved_tensors
        active = (values > 0.0) & (values < 1.0)
        return (gradient * active.to(gradient.dtype),)


def straight_through_step(values: torch.Tensor) -> torch.Tensor:
    return _StraightThroughStep.apply(values)


class FixedSparseDictionary:
    """Fixed random dictionary used to map a task descriptor to one gate."""

    def __init__(
        self,
        *,
        descriptor_dim: int,
        gate_dim: int,
        seed: int,
        atom_norm: float,
        lasso_alpha: float,
    ) -> None:
        if descriptor_dim <= 0 or gate_dim <= 0:
            raise ValueError("Dictionary dimensions must be positive.")
        if atom_norm <= 0.0 or lasso_alpha <= 0.0:
            raise ValueError("Dictionary norm and Lasso alpha must be positive.")
        rng = np.random.RandomState(seed)
        dictionary = rng.normal(size=(gate_dim, descriptor_dim))
        norms = np.linalg.norm(dictionary, axis=1, keepdims=True)
        scales = np.minimum(1.0, atom_norm / (norms + 1e-6))
        self.dictionary = np.asarray(dictionary * scales, dtype=np.float64)
        self.lasso_alpha = float(lasso_alpha)

    def encode(self, descriptor: np.ndarray) -> np.ndarray:
        from sklearn.decomposition import sparse_encode

        descriptor = np.asarray(descriptor, dtype=np.float64).reshape(1, -1)
        if descriptor.shape[1] != self.dictionary.shape[1]:
            raise ValueError("Task descriptor dimension does not match dictionary.")
        code = sparse_encode(
            descriptor,
            self.dictionary,
            algorithm="lasso_lars",
            alpha=self.lasso_alpha,
            max_iter=10_000,
            check_input=True,
        )
        if code.shape != (1, self.dictionary.shape[0]):
            raise RuntimeError("Sparse encoder returned an unexpected shape.")
        return code[0].astype(np.float32, copy=False)

    def state_dict(self) -> dict[str, Any]:
        return {
            "dictionary": self.dictionary.copy(),
            "lasso_alpha": self.lasso_alpha,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        dictionary = np.asarray(state["dictionary"], dtype=np.float64)
        if dictionary.shape != self.dictionary.shape:
            raise ValueError("Stored sparse dictionary has the wrong shape.")
        self.dictionary = dictionary.copy()
        self.lasso_alpha = float(state["lasso_alpha"])


class SSDEGaussianActor(nn.Module):
    """Task-gated Gaussian actor implementing SSDE's sparse MetaPolicy."""

    def __init__(
        self,
        *,
        observation_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        task_id_dim: int,
        num_heads: int,
        hide_task_id: bool,
        hidden_sizes: Sequence[int],
    ) -> None:
        super().__init__()
        if num_heads <= 0 or task_id_dim != num_heads:
            raise ValueError("SSDE requires one task-ID entry per actor head.")
        if len(hidden_sizes) != 4 or any(size <= 0 for size in hidden_sizes):
            raise ValueError("SSDE requires four positive hidden-layer sizes.")
        self.observation_dim = int(observation_dim)
        self.task_id_dim = int(task_id_dim)
        self.num_heads = int(num_heads)
        self.hide_task_id = bool(hide_task_id)
        self.action_dim = int(np.asarray(action_low).shape[0])
        self.hidden_sizes = tuple(int(size) for size in hidden_sizes)
        physical_dim = observation_dim - task_id_dim if hide_task_id else observation_dim

        dims = (physical_dim, *self.hidden_sizes)
        self.backbone_layers = nn.ModuleList(
            nn.Linear(dims[index], dims[index + 1])
            for index in range(len(self.hidden_sizes))
        )
        for layer in self.backbone_layers:
            initialize_linear_layer(layer)
        self.task_embeddings = nn.ParameterList(
            nn.Parameter(torch.empty(num_heads, size)) for size in self.hidden_sizes
        )
        self.random_task_embeddings = nn.ParameterList(
            nn.Parameter(torch.empty(num_heads, size))
            for size in self.hidden_sizes
        )
        for embedding, random_embedding in zip(
            self.task_embeddings, self.random_task_embeddings, strict=True
        ):
            nn.init.xavier_uniform_(embedding)
            nn.init.xavier_uniform_(random_embedding)

        self.mean_head = nn.Linear(self.hidden_sizes[-1], self.action_dim * num_heads, bias=False)
        self.log_std_head = nn.Linear(self.hidden_sizes[-1], self.action_dim * num_heads)
        nn.init.xavier_uniform_(self.mean_head.weight)
        initialize_linear_layer(self.log_std_head)
        with torch.no_grad():
            self.mean_head.weight.mul_(1e-4)
            self.log_std_head.weight.mul_(1e-4)

        low = np.asarray(action_low, dtype=np.float32)
        high = np.asarray(action_high, dtype=np.float32)
        self.register_buffer("action_scale", torch.as_tensor((high - low) / 2.0))
        self.register_buffer("action_bias", torch.as_tensor((high + low) / 2.0))

        self._overlap_masks: dict[int, tuple[torch.Tensor, ...]] = {}
        self._beta_by_task: dict[int, tuple[float, ...]] = {}

    def _split_observation(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if observations.ndim != 2 or observations.shape[1] != self.observation_dim:
            raise ValueError("SSDE actor received observations with the wrong shape.")
        task_one_hot = observations[:, -self.task_id_dim :]
        physical = observations[:, : -self.task_id_dim] if self.hide_task_id else observations
        return physical, task_one_hot

    def task_masks(self, task_index: int) -> tuple[torch.Tensor, ...]:
        if not 0 <= task_index < self.num_heads:
            raise ValueError("SSDE task index is out of range.")
        return tuple(
            torch.maximum(
                straight_through_step(embedding[task_index]),
                straight_through_step(random_embedding[task_index]),
            )
            for embedding, random_embedding in zip(
                self.task_embeddings, self.random_task_embeddings, strict=True
            )
        )

    def set_task_overlap(
        self,
        task_index: int,
        overlap_masks: tuple[torch.Tensor, ...],
        beta: tuple[float, ...],
    ) -> None:
        expected_masks = len(self.backbone_layers) * 2 + 1
        if len(overlap_masks) != expected_masks:
            raise ValueError("SSDE overlap must cover layer weights, biases, and mean head.")
        if len(beta) != len(self.backbone_layers) + 1:
            raise ValueError("SSDE beta must cover every layer and mean head.")
        self._overlap_masks[int(task_index)] = tuple(mask.detach().cpu().bool() for mask in overlap_masks)
        self._beta_by_task[int(task_index)] = tuple(float(value) for value in beta)

    def _linear_with_overlap(
        self,
        inputs: torch.Tensor,
        layer: nn.Linear,
        task_index: int,
        layer_index: int,
    ) -> torch.Tensor:
        masks = self._overlap_masks.get(task_index)
        beta = self._beta_by_task.get(task_index)
        if masks is None or beta is None:
            return layer(inputs)
        overlap = masks[layer_index * 2].to(device=layer.weight.device)
        scale = torch.where(overlap, beta[layer_index], 1.0).to(layer.weight.dtype)
        weight = layer.weight * scale
        bias = layer.bias
        if bias is not None:
            bias_overlap = masks[layer_index * 2 + 1].to(device=bias.device)
            bias_scale = torch.where(bias_overlap, beta[layer_index], 1.0).to(bias.dtype)
            bias = bias * bias_scale
        return torch.nn.functional.linear(inputs, weight, bias)

    def _features_for_task(self, physical: torch.Tensor, task_index: int) -> torch.Tensor:
        masks = self.task_masks(task_index)
        x = physical
        for index, (layer, mask) in enumerate(
            zip(self.backbone_layers, masks, strict=True)
        ):
            x = self._linear_with_overlap(x, layer, task_index, index)
            x = x * mask.to(x.dtype)
            if index == 0:
                active = mask.bool()
                if active.any():
                    active_x = x[:, active]
                    mean = active_x.mean(dim=-1, keepdim=True)
                    variance = active_x.var(dim=-1, unbiased=False, keepdim=True)
                    normalized = (active_x - mean) / torch.sqrt(variance + 1e-3)
                    x = x.clone()
                    x[:, active] = torch.tanh(normalized)
                else:
                    x = torch.zeros_like(x)
            else:
                x = torch.nn.functional.leaky_relu(x, negative_slope=0.2)
        return x

    def intermediate_features(
        self, physical: torch.Tensor, task_index: int
    ) -> tuple[torch.Tensor, ...]:
        """Return post-mask activations used by SSDE sensitivity scoring."""
        masks = self.task_masks(task_index)
        outputs: list[torch.Tensor] = []
        x = physical
        for index, (layer, mask) in enumerate(
            zip(self.backbone_layers, masks, strict=True)
        ):
            x = self._linear_with_overlap(x, layer, task_index, index)
            x = x * mask.to(x.dtype)
            outputs.append(x)
            if index == 0:
                active = mask.bool()
                if active.any():
                    active_x = x[:, active]
                    mean = active_x.mean(dim=-1, keepdim=True)
                    variance = active_x.var(dim=-1, unbiased=False, keepdim=True)
                    normalized = (active_x - mean) / torch.sqrt(variance + 1e-3)
                    x = x.clone()
                    x[:, active] = torch.tanh(normalized)
                else:
                    x = torch.zeros_like(x)
            else:
                x = torch.nn.functional.leaky_relu(x, negative_slope=0.2)
        return tuple(outputs)

    def mean_pre_activation(self, physical: torch.Tensor, task_index: int) -> torch.Tensor:
        features = self._features_for_task(physical, task_index)
        head_slice = slice(
            task_index * self.action_dim, (task_index + 1) * self.action_dim
        )
        return torch.nn.functional.linear(
            features, self.mean_head.weight[head_slice], None
        )

    def distribution_parameters(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        physical, task_one_hot = self._split_observation(observations)
        indices = task_one_hot.argmax(dim=1)
        if not torch.all(task_one_hot.sum(dim=1) == 1):
            raise ValueError("SSDE actor requires valid one-hot task IDs.")
        means = torch.empty((observations.shape[0], self.action_dim), device=observations.device)
        log_stds = torch.empty_like(means)
        for task_index in indices.unique(sorted=True).tolist():
            selected = indices == task_index
            features = self._features_for_task(physical[selected], int(task_index))
            head_slice = slice(task_index * self.action_dim, (task_index + 1) * self.action_dim)
            overlap = self._overlap_masks.get(int(task_index))
            beta = self._beta_by_task.get(int(task_index))
            mean_weight = self.mean_head.weight[head_slice]
            if overlap is not None and beta is not None:
                mean_overlap = overlap[-1].to(device=mean_weight.device)[head_slice]
                mean_weight = mean_weight * torch.where(
                    mean_overlap, beta[-1], 1.0
                ).to(mean_weight.dtype)
            raw_mean = torch.nn.functional.linear(features, mean_weight, None)
            raw_log_std = torch.nn.functional.linear(
                features,
                self.log_std_head.weight[head_slice],
                self.log_std_head.bias[head_slice],
            )
            means[selected] = torch.tanh(raw_mean)
            bounded = -20.0 + 11.0 * (torch.tanh(raw_log_std) + 1.0)
            log_stds[selected] = torch.log(torch.nn.functional.softplus(bounded) + 1e-8)
        return means, log_stds

    def sample(self, observations: torch.Tensor) -> ActorOutput:
        mean, log_std = self.distribution_parameters(observations)
        distribution = Normal(mean, log_std.exp())
        pre_tanh = distribution.rsample()
        squashed = torch.tanh(pre_tanh)
        sampled = squashed * self.action_scale + self.action_bias
        mean_action = torch.tanh(mean) * self.action_scale + self.action_bias
        correction = 2.0 * (
            math.log(2.0) - pre_tanh - torch.nn.functional.softplus(-2.0 * pre_tanh)
        )
        log_probability = (
            distribution.log_prob(pre_tanh) - correction - torch.log(self.action_scale)
        ).sum(dim=-1, keepdim=True)
        return ActorOutput(mean_action, sampled, log_probability, log_std)

    def deterministic_action(self, observations: torch.Tensor) -> torch.Tensor:
        mean, _ = self.distribution_parameters(observations)
        return torch.tanh(mean) * self.action_scale + self.action_bias

    @torch.no_grad()
    def act(self, observation: np.ndarray, deterministic: bool, device: torch.device | str) -> np.ndarray:
        tensor = torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        action = self.deterministic_action(tensor) if deterministic else self.sample(tensor).sampled_action
        return action.squeeze(0).cpu().numpy()

    def get_extra_state(self) -> dict[str, Any]:
        return {
            "overlap_masks": self._overlap_masks,
            "beta_by_task": self._beta_by_task,
        }

    def set_extra_state(self, state: dict[str, Any]) -> None:
        self._overlap_masks = {
            int(key): tuple(mask.detach().cpu().bool() for mask in masks)
            for key, masks in state.get("overlap_masks", {}).items()
        }
        self._beta_by_task = {
            int(key): tuple(float(value) for value in values)
            for key, values in state.get("beta_by_task", {}).items()
        }


class SSDEFullSACAgent(SACAgent):
    """Algorithmically faithful SSDE mechanisms under the local SAC runner."""

    def __init__(
        self,
        *args,
        ssde_seed: int = 0,
        hidden_size: int = 1024,
        descriptor_dim: int = 384,
        fixed_lasso_alpha: float = 0.01,
        random_lasso_alpha_start: float = 0.01,
        random_lasso_alpha_end: float = 0.0001,
        default_beta: float = 0.3,
        use_adaptive_beta: bool = False,
        beta_lambda: float = 0.5,
        sensitivity_interval: int = 80_000,
        sensitivity_batch_size: int = 1_000,
        sensitivity_threshold: float = 0.6,
        stop_reset_after_steps: int = 650_000,
        ssde_target_entropy: float = -2.0,
        random_distillation: bool = True,
        distillation_steps: int = 20_000,
        distillation_batch_size: int = 256,
        descriptor_encoder: Callable[[str], np.ndarray] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if hidden_size <= 0 or descriptor_dim <= 0:
            raise ValueError("SSDE hidden and descriptor dimensions must be positive.")
        if not 0.0 <= default_beta <= 1.0:
            raise ValueError("SSDE beta must lie in [0, 1].")
        if sensitivity_interval <= 0 or sensitivity_batch_size <= 0:
            raise ValueError("SSDE sensitivity settings must be positive.")
        self.ssde_seed = int(ssde_seed)
        self.descriptor_dim = int(descriptor_dim)
        self.default_beta = float(default_beta)
        self.use_adaptive_beta = bool(use_adaptive_beta)
        self.beta_lambda = float(beta_lambda)
        self.sensitivity_interval = int(sensitivity_interval)
        self.sensitivity_batch_size = int(sensitivity_batch_size)
        self.sensitivity_threshold = float(sensitivity_threshold)
        self.stop_reset_after_steps = int(stop_reset_after_steps)
        if distillation_steps < 0 or distillation_batch_size <= 0:
            raise ValueError("SSDE distillation settings are invalid.")
        self.random_distillation = bool(random_distillation)
        self.distillation_steps = int(distillation_steps)
        self.distillation_batch_size = int(distillation_batch_size)
        if not math.isfinite(ssde_target_entropy):
            raise ValueError("SSDE target entropy must be finite.")
        self.target_entropy = float(ssde_target_entropy)
        self._descriptor_encoder = descriptor_encoder

        low = self.actor.action_bias.detach().cpu().numpy() - self.actor.action_scale.detach().cpu().numpy()
        high = self.actor.action_bias.detach().cpu().numpy() + self.actor.action_scale.detach().cpu().numpy()
        self.actor = SSDEGaussianActor(
            observation_dim=self.observation_dim,
            action_low=low,
            action_high=high,
            task_id_dim=self.task_id_dim,
            num_heads=self.num_tasks,
            hide_task_id=self.hide_task_id,
            hidden_sizes=(hidden_size,) * 4,
        ).to(self.device)

        self._fixed_dictionaries = tuple(
            FixedSparseDictionary(
                descriptor_dim=descriptor_dim,
                gate_dim=hidden_size,
                seed=self.ssde_seed + layer + 1,
                atom_norm=1.0,
                lasso_alpha=fixed_lasso_alpha,
            )
            for layer in range(4)
        )
        random_alphas = np.linspace(
            random_lasso_alpha_start,
            random_lasso_alpha_end,
            len(CW10_TASK_DESCRIPTIONS),
        )
        self._random_lasso_alphas = tuple(float(value) for value in random_alphas[: self.num_tasks])
        self._cumulative_masks = tuple(torch.zeros(hidden_size, dtype=torch.bool) for _ in range(4))
        self._parameter_masks = self._gradient_masks_from_neurons(self._cumulative_masks)
        self._task_descriptors: dict[int, np.ndarray] = {}
        self._task_steps = 0
        self._active_task_index = 0
        self._pre_task_actor_state: dict[str, torch.Tensor] | None = None
        self._sensitivity_observations: deque[np.ndarray] = deque(
            maxlen=self.sensitivity_batch_size
        )
        self._last_sensitivity_step = -1
        self._last_reactivated = 0
        self._task_reactivated_total = 0
        self._task_distillation_updates = 0
        self.rebuild_optimizer()

    def _encode_description(self, description: str) -> np.ndarray:
        if self._descriptor_encoder is not None:
            encoded = self._descriptor_encoder(description)
        else:
            global _SENTENCE_ENCODER_CACHE
            from sentence_transformers import SentenceTransformer

            if _SENTENCE_ENCODER_CACHE is None:
                try:
                    _SENTENCE_ENCODER_CACHE = SentenceTransformer(
                        "all-MiniLM-L12-v2", local_files_only=True
                    )
                except Exception as error:
                    raise RuntimeError(
                        "SSDE requires the cached all-MiniLM-L12-v2 descriptor "
                        "model. Install dependencies and cache the model once "
                        "before launching a long run."
                    ) from error
            encoded = _SENTENCE_ENCODER_CACHE.encode(
                description,
                convert_to_numpy=True,
            )
        result = np.asarray(encoded, dtype=np.float32).reshape(-1)
        if result.shape != (self.descriptor_dim,) or not np.isfinite(result).all():
            raise ValueError("SSDE descriptor encoder returned an invalid vector.")
        return result

    def on_task_start(self, *, task_index: int, replay_buffer: object) -> None:
        if task_index >= len(CW10_TASK_DESCRIPTIONS):
            raise ValueError("No SSDE description is configured for this task index.")
        self._active_task_index = int(task_index)
        self._task_steps = 0
        self._last_sensitivity_step = -1
        self._task_reactivated_total = 0
        self._task_distillation_updates = 0
        self._sensitivity_observations.clear()
        descriptor = self._encode_description(CW10_TASK_DESCRIPTIONS[task_index])
        self._task_descriptors[task_index] = descriptor.copy()
        with torch.no_grad():
            for layer, dictionary in enumerate(self._fixed_dictionaries):
                self.actor.task_embeddings[layer][task_index].copy_(
                    torch.as_tensor(dictionary.encode(descriptor), device=self.device)
                )
                random_dictionary = FixedSparseDictionary(
                    descriptor_dim=self.descriptor_dim,
                    gate_dim=self.actor.hidden_sizes[layer],
                    seed=task_index * 10_000 + layer + 10_000 + self.ssde_seed * 512,
                    atom_norm=1.0,
                    lasso_alpha=self._random_lasso_alphas[task_index],
                )
                self.actor.random_task_embeddings[layer][task_index].copy_(
                    torch.as_tensor(random_dictionary.encode(descriptor), device=self.device)
                )
        current_masks = tuple(mask.detach().cpu().bool() for mask in self.actor.task_masks(task_index))
        current_grad = self._gradient_masks_from_neurons(current_masks)
        overlap_keys = tuple(
            key
            for index in range(4)
            for key in (
                f"backbone_layers.{index}.weight",
                f"backbone_layers.{index}.bias",
            )
        ) + ("mean_head.weight",)
        overlap = tuple(
            (~current_grad[key]) & (~self._parameter_masks[key])
            for key in overlap_keys
        )
        beta = []
        beta_keys = tuple(
            f"backbone_layers.{index}.weight" for index in range(4)
        ) + ("mean_head.weight",)
        overlap_by_key = dict(zip(overlap_keys, overlap, strict=True))
        for key in beta_keys:
            overlap_mask = overlap_by_key[key]
            if self.use_adaptive_beta:
                forward_count = int((~current_grad[key]).sum())
                overlap_count = int(overlap_mask.sum())
                value = 0.0 if forward_count == 0 else self.beta_lambda * min(
                    max(forward_count - overlap_count, 0) / max(overlap_count, 1), 1.0
                )
            else:
                value = self.default_beta
            beta.append(value)
        self.actor.set_task_overlap(task_index, overlap, tuple(beta))
        self.rebuild_optimizer()
        self._distill_random_policy(
            task_index=task_index,
            replay_buffer=replay_buffer,
        )
        # SSDE discards distillation Adam moments before online SAC starts.
        self.rebuild_optimizer()
        # Dormant units are restored to the post-distillation task start, which
        # matches the ordering in the official training loop.
        self._pre_task_actor_state = {
            name: parameter.detach().clone()
            for name, parameter in self.actor.named_parameters()
        }

    def _distill_random_policy(
        self,
        *,
        task_index: int,
        replay_buffer: object,
    ) -> None:
        """Match the official optional N(0, I) task-start distillation."""
        if (
            task_index == 0
            or not self.random_distillation
            or self.distillation_steps == 0
        ):
            return
        if len(replay_buffer) == 0 or not hasattr(replay_buffer, "sample"):
            raise RuntimeError(
                "SSDE random distillation requires the previous task replay buffer."
            )

        actor_parameters = tuple(self.actor.parameters())
        self._task_distillation_updates = self.distillation_steps
        report_every = max(self.distillation_steps // 10, 1)
        for update_index in range(self.distillation_steps):
            batch = replay_buffer.sample(
                self.distillation_batch_size,
                self.device,
            )
            observations = batch.observations.to(
                self.device,
                dtype=torch.float32,
            ).clone()
            observations[:, -self.task_id_dim :] = 0.0
            observations[:, -self.task_id_dim + task_index] = 1.0
            means, log_stds = self.actor.distribution_parameters(observations)
            variances = torch.exp(2.0 * log_stds)
            # KL[N(mean, std) || N(0, I)], averaged as in the official code.
            distillation_loss = 0.5 * (
                variances + means.square() - 1.0 - 2.0 * log_stds
            ).mean()
            gradients = torch.autograd.grad(
                distillation_loss,
                actor_parameters,
            )
            gradients = self.adjust_actor_gradients(
                gradients=gradients,
                parameters=actor_parameters,
                task_index=task_index,
            )
            self._apply_gradients(
                parameters=actor_parameters,
                gradients=gradients,
                group_name="ssde_distillation",
            )
            if update_index == 0 or (update_index + 1) % report_every == 0:
                print(
                    f"[ssde-distill] task={task_index} "
                    f"update={update_index + 1}/{self.distillation_steps} "
                    f"loss={float(distillation_loss.detach()):.6f}",
                    flush=True,
                )

    def _gradient_masks_from_neurons(
        self, neuron_masks: tuple[torch.Tensor, ...]
    ) -> dict[str, torch.Tensor]:
        result: dict[str, torch.Tensor] = {}
        previous: torch.Tensor | None = None
        for index, current in enumerate(neuron_masks):
            current = current.bool().cpu()
            if previous is None:
                weight_frozen = current[:, None].expand(current.shape[0], self.actor.backbone_layers[index].in_features)
            else:
                weight_frozen = current[:, None] & previous[None, :]
            result[f"backbone_layers.{index}.weight"] = ~weight_frozen
            result[f"backbone_layers.{index}.bias"] = ~current
            previous = current
        assert previous is not None
        # Under the shared CW10 protocol the output is multi-head. The current
        # task selects one slice, so old heads already receive zero gradients.
        result["mean_head.weight"] = torch.ones_like(
            self.actor.mean_head.weight, dtype=torch.bool, device="cpu"
        )
        return result

    def adjust_actor_gradients(
        self,
        *,
        gradients: tuple[torch.Tensor, ...],
        parameters: tuple[nn.Parameter, ...],
        task_index: int,
    ) -> tuple[torch.Tensor, ...]:
        del task_index
        names = {id(parameter): name for name, parameter in self.actor.named_parameters()}
        adjusted = []
        for gradient, parameter in zip(gradients, parameters, strict=True):
            name = names[id(parameter)]
            if name in self._parameter_masks:
                mask = self._parameter_masks[name].to(device=gradient.device, dtype=gradient.dtype)
                adjusted.append(gradient * mask)
            elif name.startswith("task_embeddings") or name.startswith("random_task_embeddings"):
                adjusted.append(torch.zeros_like(gradient))
            else:
                adjusted.append(gradient)
        return tuple(adjusted)

    def update_batch(self, *args, **kwargs):
        result = super().update_batch(*args, **kwargs)
        if (
            self._task_steps > 0
            and self._task_steps % self.sensitivity_interval == 0
            and self._task_steps < self.stop_reset_after_steps
            and self._last_sensitivity_step != self._task_steps
        ):
            self._last_reactivated = self._reactivate_dormant_neurons()
            self._task_reactivated_total += self._last_reactivated
            self._last_sensitivity_step = self._task_steps
            self.rebuild_optimizer()
            if result is not None:
                result["ssde_reactivated"] = float(self._last_reactivated)
        return result

    def on_environment_transition(
        self,
        *,
        observation: np.ndarray,
        next_observation: np.ndarray,
    ) -> None:
        del next_observation
        observation = np.asarray(observation, dtype=np.float32)
        if observation.shape != (self.observation_dim,):
            raise ValueError("SSDE sensitivity observation has the wrong shape.")
        self._sensitivity_observations.append(observation.copy())
        self._task_steps += 1

    def _reactivate_dormant_neurons(self) -> int:
        if self._pre_task_actor_state is None or not self._sensitivity_observations:
            return 0
        observations = torch.as_tensor(
            np.stack(tuple(self._sensitivity_observations), axis=0),
            dtype=torch.float32,
            device=self.device,
        )
        physical, _ = self.actor._split_observation(observations)
        # Match the official implementation's deterministic 1% mean-input
        # perturbation instead of injecting RNG-dependent Gaussian noise.
        perturbation = physical.mean(dim=0, keepdim=True) * 0.01
        current_masks = tuple(mask.detach().bool() for mask in self.actor.task_masks(self._active_task_index))
        total = 0
        with torch.no_grad():
            clean_layers = self.actor.intermediate_features(
                physical, self._active_task_index
            )
            perturbed_layers = self.actor.intermediate_features(
                physical + perturbation, self._active_task_index
            )
            for index, (clean, perturbed) in enumerate(
                zip(clean_layers, perturbed_layers, strict=True)
            ):
                layer = self.actor.backbone_layers[index]
                delta = (clean - perturbed).abs().mean(dim=0)
                active = current_masks[index]
                relative = delta / (delta[active].mean() + 1e-8) if active.any() else delta
                dormant = active & (relative < self.sensitivity_threshold) & self._cumulative_masks[index].logical_not().to(self.device)
                if dormant.any():
                    name = f"backbone_layers.{index}"
                    layer.weight[dormant] = self._pre_task_actor_state[f"{name}.weight"][dormant]
                    layer.bias[dormant] = self._pre_task_actor_state[f"{name}.bias"][dormant]
                    if index + 1 < len(self.actor.backbone_layers):
                        next_name = f"backbone_layers.{index + 1}.weight"
                        self.actor.backbone_layers[index + 1].weight[:, dormant] = (
                            self._pre_task_actor_state[next_name][:, dormant]
                        )
                    else:
                        head_slice = slice(
                            self._active_task_index * self.action_dim,
                            (self._active_task_index + 1) * self.action_dim,
                        )
                        self.actor.mean_head.weight[head_slice, dormant] = (
                            self._pre_task_actor_state["mean_head.weight"][head_slice, dormant]
                        )
                        self.actor.log_std_head.weight[head_slice, dormant] = (
                            self._pre_task_actor_state["log_std_head.weight"][head_slice, dormant]
                        )
                    total += int(dormant.sum().item())
            clean_mean = self.actor.mean_pre_activation(
                physical, self._active_task_index
            )
            perturbed_mean = self.actor.mean_pre_activation(
                physical + perturbation, self._active_task_index
            )
            mean_delta = (clean_mean - perturbed_mean).abs().mean(dim=0)
            relative_mean = mean_delta / (mean_delta.mean() + 1e-8)
            dormant_mean = relative_mean < self.sensitivity_threshold
            if dormant_mean.any():
                head_start = self._active_task_index * self.action_dim
                rows = torch.arange(
                    head_start,
                    head_start + self.action_dim,
                    device=self.device,
                )[dormant_mean]
                self.actor.mean_head.weight[rows] = self._pre_task_actor_state[
                    "mean_head.weight"
                ][rows]
                total += int(dormant_mean.sum().item())
        self._sensitivity_observations.clear()
        return total

    @torch.no_grad()
    def task_diagnostics(self) -> dict[str, float | int | None]:
        masks = self.actor.task_masks(self._active_task_index)
        active = sum(int(mask.detach().count_nonzero()) for mask in masks)
        total = sum(mask.numel() for mask in masks)
        overlap_masks = self.actor._overlap_masks.get(self._active_task_index, ())
        overlap_active = sum(int(mask.count_nonzero()) for mask in overlap_masks)
        overlap_total = sum(mask.numel() for mask in overlap_masks)
        beta = self.actor._beta_by_task.get(self._active_task_index, ())
        return {
            "ssde_gate_density": active / total if total else None,
            "ssde_overlap_ratio": (
                overlap_active / overlap_total if overlap_total else None
            ),
            "ssde_beta_mean": float(np.mean(beta)) if beta else None,
            "ssde_distillation_updates": self._task_distillation_updates,
            "ssde_reactivated_neurons": self._task_reactivated_total,
        }

    def on_task_end(self, *, task_index: int, replay_buffer: object, batch_size: int) -> None:
        del replay_buffer, batch_size
        current = tuple(mask.detach().cpu().bool() for mask in self.actor.task_masks(task_index))
        self._cumulative_masks = tuple(
            old | new for old, new in zip(self._cumulative_masks, current, strict=True)
        )
        current_gradient_masks = self._gradient_masks_from_neurons(current)
        self._parameter_masks = {
            key: self._parameter_masks[key] & current_gradient_masks[key]
            for key in self._parameter_masks
        }
        self._pre_task_actor_state = None
        self._sensitivity_observations.clear()
        self.rebuild_optimizer()

    def get_extra_state(self) -> dict[str, Any]:
        return {
            "cumulative_masks": self._cumulative_masks,
            "parameter_masks": self._parameter_masks,
            "task_descriptors": self._task_descriptors,
            "task_steps": self._task_steps,
            "active_task_index": self._active_task_index,
            "last_reactivated": self._last_reactivated,
            "last_sensitivity_step": self._last_sensitivity_step,
            "task_reactivated_total": self._task_reactivated_total,
            "task_distillation_updates": self._task_distillation_updates,
            "fixed_dictionaries": [item.state_dict() for item in self._fixed_dictionaries],
            "random_lasso_alphas": self._random_lasso_alphas,
            "random_distillation": self.random_distillation,
            "distillation_steps": self.distillation_steps,
            "distillation_batch_size": self.distillation_batch_size,
        }

    def set_extra_state(self, state: dict[str, Any]) -> None:
        self._cumulative_masks = tuple(mask.detach().cpu().bool() for mask in state["cumulative_masks"])
        self._parameter_masks = {
            key: value.detach().cpu().bool() for key, value in state["parameter_masks"].items()
        }
        self._task_descriptors = {
            int(key): np.asarray(value, dtype=np.float32).copy()
            for key, value in state.get("task_descriptors", {}).items()
        }
        self._task_steps = int(state.get("task_steps", 0))
        self._active_task_index = int(state.get("active_task_index", 0))
        self._last_reactivated = int(state.get("last_reactivated", 0))
        self._last_sensitivity_step = int(state.get("last_sensitivity_step", -1))
        self._task_reactivated_total = int(state.get("task_reactivated_total", 0))
        self._task_distillation_updates = int(
            state.get("task_distillation_updates", 0)
        )
        for learner, learner_state in zip(
            self._fixed_dictionaries, state["fixed_dictionaries"], strict=True
        ):
            learner.load_state_dict(learner_state)
        stored_alphas = tuple(float(value) for value in state["random_lasso_alphas"])
        if len(stored_alphas) != self.num_tasks:
            raise ValueError("Stored SSDE random-alpha schedule has the wrong length.")
        self._random_lasso_alphas = stored_alphas
        self.random_distillation = bool(
            state.get("random_distillation", self.random_distillation)
        )
        self.distillation_steps = int(
            state.get("distillation_steps", self.distillation_steps)
        )
        self.distillation_batch_size = int(
            state.get("distillation_batch_size", self.distillation_batch_size)
        )
