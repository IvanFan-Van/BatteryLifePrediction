"""Estimate the current Tongji state with field, time, and cycle attention."""

from collections.abc import Mapping

import torch
from torch import nn

from src.utils.files import read_yaml

from .embeddings import TongjiInputEncoder


class _AttentionPool(nn.Module):
    """Prepend a learned summary token and read its contextualized output."""

    def __init__(self, width, heads, feedforward_dim, dropout, layers):
        super().__init__()
        self.summary = nn.Parameter(torch.empty(1, 1, width))
        nn.init.normal_(self.summary, std=0.02)
        self.layers = nn.ModuleList(
            nn.TransformerEncoderLayer(
                width,
                heads,
                dim_feedforward=feedforward_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(layers)
        )

    def forward(self, tokens, valid=None):
        summary = self.summary.to(tokens.dtype).expand(tokens.shape[0], -1, -1)
        sequence = torch.cat((summary, tokens), dim=1)
        padding = None
        if valid is not None:
            summary_valid = valid.new_ones(valid.shape[0], 1)
            padding = ~torch.cat((summary_valid, valid), dim=1)
        for layer in self.layers:
            sequence = layer(sequence, src_key_padding_mask=padding)
        return sequence[:, 0]


class StateEstimator(nn.Module):
    """Read one current state per sample from observations and history only."""

    @classmethod
    def from_config(cls, config):
        if not isinstance(config, Mapping):
            config = read_yaml(config)
        return cls(
            TongjiInputEncoder.from_config(config), **config["model"]["state_estimator"]
        )

    def __init__(
        self,
        input_encoder,
        *,
        d_model,
        state_dim,
        num_heads,
        feedforward_dim,
        field_layers,
        time_layers,
        cycle_layers,
        dropout,
    ):
        super().__init__()
        if min(d_model, num_heads) < 1 or d_model % num_heads:
            raise ValueError(
                "d_model must be positive and divisible by positive num_heads"
            )
        if min(state_dim, feedforward_dim, field_layers, time_layers, cycle_layers) < 1:
            raise ValueError(
                "State width, feedforward width, and layer counts must be positive"
            )

        self.input_encoder = input_encoder
        self.numeric_projection = nn.Linear(input_encoder.numeric_dim, d_model)
        self.categorical_projection = nn.Linear(input_encoder.categorical_dim, d_model)

        self.field_types = nn.ParameterDict()
        field_counts = {
            "observation": len(input_encoder.observations.fields),
            "time": 1,
            "static": len(input_encoder.static_categorical)
            + len(input_encoder.static_numeric.fields),
            "condition": len(input_encoder.conditions.fields),
            "distance": 1,
        }
        for name, count in field_counts.items():
            embedding = nn.Parameter(torch.empty(count, d_model))
            nn.init.normal_(embedding, std=0.02)
            self.field_types[name] = embedding

        self.field_attention = _AttentionPool(
            d_model, num_heads, feedforward_dim, dropout, field_layers
        )
        self.time_attention = _AttentionPool(
            d_model, num_heads, feedforward_dim, dropout, time_layers
        )
        self.cycle_attention = _AttentionPool(
            d_model, num_heads, feedforward_dim, dropout, cycle_layers
        )
        self.state_projection = nn.Linear(d_model, state_dim)

    def _static_tokens(self, batch):
        categorical, numeric = self.input_encoder.encode_static_tokens(
            batch["static_categorical"], batch["static_numeric"], batch["static_valid"]
        )
        return torch.cat(
            (
                self.categorical_projection(categorical),
                self.numeric_projection(numeric),
            ),
            dim=1,
        )

    def _timeline_context(self, batch):
        first_id = batch["cycle_id"][:, :1]
        last_distance = batch["last_cycle_id"][:, None] - first_id
        cycle_slots = batch["cycle_id"] - first_id
        history = batch["history_u"]
        steps = torch.arange(history.shape[1] + 1, device=history.device)
        valid = steps <= last_distance
        distance = self.numeric_projection(
            self.input_encoder.cycle_distance(last_distance - steps, valid)
        )

        # The loader provides every step before the forecast origin in order.
        # Select real rows by length so a genuine cycle numbered zero is retained.
        rows, columns = (steps[:-1] < last_distance).nonzero(as_tuple=True)
        history_slots = batch["history_cycle_id"][rows, columns] - first_id[rows, 0]
        values = history.new_zeros(history.shape[0], steps.numel(), history.shape[-1])
        condition_valid = batch["history_u_valid"].new_zeros(values.shape)
        values[rows, history_slots] = history[rows, columns]
        condition_valid[rows, history_slots] = batch["history_u_valid"][rows, columns]
        conditions = self.numeric_projection(
            self.input_encoder.conditions.encode_tokens(values, condition_valid)
        )
        return cycle_slots, valid, conditions, distance

    def _observed_cycles(self, batch, cycle_slots, static, context):
        point_valid = batch["point_valid"]
        rows, cycles, points = point_valid.nonzero(as_tuple=True)
        observed = self.input_encoder.observations.encode_tokens(
            batch["values"][rows, cycles, points],
            batch["value_visible"][rows, cycles, points],
        )
        observed = self.numeric_projection(observed)
        seconds = batch["time_in_cycle_s"][rows, cycles, points]
        time = self.numeric_projection(
            self.input_encoder.time(seconds, torch.ones_like(seconds, dtype=torch.bool))
        )
        fields = torch.cat(
            (
                observed + self.field_types["observation"].to(observed.dtype),
                time.unsqueeze(1) + self.field_types["time"].to(time.dtype),
                static[rows],
                context[rows, cycle_slots[rows, cycles]],
            ),
            dim=1,
        )
        points = self.field_attention(fields) + time
        point_sequence = points.new_zeros(*point_valid.shape, points.shape[-1])
        point_sequence[point_valid] = points

        cycle_valid = batch["cycle_valid"]
        return self.time_attention(
            point_sequence[cycle_valid], point_valid[cycle_valid]
        )

    def _encode_history(self, batch):
        static = self._static_tokens(batch)
        typed_static = static + self.field_types["static"].to(static.dtype)
        cycle_slots, valid, conditions, distance = self._timeline_context(batch)
        context = torch.cat(
            (
                conditions + self.field_types["condition"].to(conditions.dtype),
                distance.unsqueeze(-2)
                + self.field_types["distance"].to(distance.dtype),
            ),
            dim=-2,
        )
        observed = self._observed_cycles(batch, cycle_slots, typed_static, context)

        observed_rows, observed_cycles = batch["cycle_valid"].nonzero(as_tuple=True)
        observed_slots = cycle_slots[observed_rows, observed_cycles]
        timeline = distance.new_zeros(distance.shape)
        timeline[observed_rows, observed_slots] = observed

        history_only = valid.clone()
        history_only[observed_rows, observed_slots] = False
        history_rows, history_slots = history_only.nonzero(as_tuple=True)
        if history_rows.numel():
            fields = torch.cat(
                (typed_static[history_rows], context[history_rows, history_slots]),
                dim=1,
            )
            timeline[history_rows, history_slots] = self.field_attention(fields)

        summary = self.cycle_attention(timeline + distance, valid)
        return self.state_projection(summary), static

    def forward(self, batch):
        """Return the current state of shape (B, state_dim)."""
        state, _ = self._encode_history(batch)
        return state

    def encode_for_dynamics(self, batch):
        """Return the state, static attributes, and per-step future conditions."""
        state, static = self._encode_history(batch)
        future = self.input_encoder.conditions.encode_tokens(
            batch["future_u"], batch["future_u_valid"]
        )
        return {
            "state": state,
            "static": static.flatten(-2),
            "future_conditions": self.numeric_projection(future).flatten(-2),
        }
