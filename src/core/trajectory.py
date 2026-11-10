"""Conditioned state rollout and shared SOH readout."""

import torch
from torch import nn

from .embeddings import ACTIVATIONS
from .state_estimator import StateEstimator


def _three_layer_mlp(input_dim, output_dim, *, hidden_dims, activation):
    if len(hidden_dims) != 2 or any(
        type(width) is not int or width < 1 for width in hidden_dims
    ):
        raise ValueError("hidden_dims must contain two positive integer widths")
    if activation not in ACTIVATIONS:
        raise ValueError(f"activation must be one of {', '.join(ACTIVATIONS)}")
    first, second = hidden_dims
    return nn.Sequential(
        nn.Linear(input_dim, first),
        ACTIVATIONS[activation](),
        nn.Linear(first, second),
        ACTIVATIONS[activation](),
        nn.Linear(second, output_dim),
    )


class DynamicsModel(nn.Module):
    """Predict a residual state update for one cycle."""

    def __init__(self, state_dim, condition_dim, static_dim, **mlp_config):
        super().__init__()
        self.network = _three_layer_mlp(
            state_dim + condition_dim + static_dim, state_dim, **mlp_config
        )

    def forward(self, state, condition, static):
        inputs = torch.cat((state, condition, static), dim=-1)
        return state + self.network(inputs)


class SOHHead(nn.Module):
    """Read a scalar SOH from the state and static cell attributes."""

    def __init__(self, state_dim, static_dim, **mlp_config):
        super().__init__()
        self.network = _three_layer_mlp(state_dim + static_dim, 1, **mlp_config)

    def forward(self, state, static):
        return self.network(torch.cat((state, static), dim=-1)).squeeze(-1)


class BatteryTrajectoryModel(nn.Module):
    """Estimate the current state and predict one SOH per future cycle."""

    def __init__(self, config):
        super().__init__()
        self.encoder = StateEstimator.from_config(config)
        input_encoder = self.encoder.input_encoder
        model_config = config["model"]
        estimator_config = model_config["state_estimator"]
        width = estimator_config["d_model"]
        state_dim = estimator_config["state_dim"]
        static_dim = width * (
            len(input_encoder.static_categorical)
            + len(input_encoder.static_numeric.fields)
        )
        condition_dim = width * len(input_encoder.conditions.fields)
        self.dynamics = DynamicsModel(
            state_dim, condition_dim, static_dim, **model_config["dynamics"]
        )
        self.head = SOHHead(state_dim, static_dim, **model_config["soh_head"])

    def forward(self, batch):
        inputs = self.encoder.encode_for_dynamics(batch)
        state = inputs["state"]
        static = inputs["static"]
        predictions = []
        for condition in inputs["future_conditions"].unbind(dim=1):
            state = self.dynamics(state, condition, static)
            predictions.append(self.head(state, static))
        return torch.stack(predictions, dim=1)
