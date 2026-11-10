"""Field-wise embeddings for Tongji observations, metadata, and positions."""

import math
from collections.abc import Mapping

import torch
from torch import nn

from src.data import load_cell
from src.data.loading import fit_normalization
from src.utils.files import read_json, read_yaml
from src.utils.paths import processed_data_path

ACTIVATIONS = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU}


def _check_dimension(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


class ScalarEncoder(nn.Module):
    """Encode a scalar at each input position, with a learned missing vector."""

    mean: torch.Tensor
    std: torch.Tensor

    def __init__(self, *, output_dim, hidden_dim, activation, mean=0.0, std=1.0):
        super().__init__()
        _check_dimension(output_dim, "output_dim")
        _check_dimension(hidden_dim, "hidden_dim")
        if activation not in ACTIVATIONS:
            raise ValueError(f"activation must be one of {', '.join(ACTIVATIONS)}")
        if not math.isfinite(mean) or not math.isfinite(std) or std <= 0:
            raise ValueError("mean must be finite and std must be finite and positive")

        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32))
        self.network = nn.Sequential(
            nn.Linear(1, hidden_dim),
            ACTIVATIONS[activation](),
            nn.Linear(hidden_dim, output_dim),
        )
        self.missing = nn.Parameter(torch.empty(output_dim))
        nn.init.normal_(self.missing, std=0.02)

    def forward(self, values, valid):
        safe = torch.where(valid, values.to(self.mean.dtype), self.mean)
        normalized = (safe - self.mean) / self.std
        encoded = self.network(normalized.unsqueeze(-1))
        return torch.where(valid.unsqueeze(-1), encoded, self.missing.to(encoded.dtype))


class NumericFeatureEncoder(nn.Module):
    """Encode each column independently in column order."""

    def __init__(self, mean, std, *, output_dim, hidden_dim, activation):
        super().__init__()
        self.fields = nn.ModuleList(
            ScalarEncoder(
                output_dim=output_dim,
                hidden_dim=hidden_dim,
                activation=activation,
                mean=center,
                std=scale,
            )
            for center, scale in zip(mean, std, strict=True)
        )

    def encode_tokens(self, values, valid):
        if values.shape != valid.shape or values.shape[-1] != len(self.fields):
            raise ValueError(
                "values and valid must match the configured numeric fields"
            )
        return torch.stack(
            [
                encoder(values[..., index], valid[..., index])
                for index, encoder in enumerate(self.fields)
            ],
            dim=-2,
        )

    def forward(self, values, valid):
        return self.encode_tokens(values, valid).flatten(-2)


class CategoricalEncoder(nn.Module):
    """Encode a categorical field using index zero for unknown values."""

    lookup: torch.Tensor

    def __init__(self, known_ids, *, output_dim):
        super().__init__()
        _check_dimension(output_dim, "output_dim")
        known_ids = sorted(set(known_ids) - {0})
        if any(index < 0 for index in known_ids):
            raise ValueError("Known category IDs must be nonnegative")
        lookup = torch.zeros(max(known_ids, default=0) + 1, dtype=torch.long)
        if known_ids:
            lookup[known_ids] = torch.arange(1, len(known_ids) + 1)
        self.register_buffer("lookup", lookup)
        self.embedding = nn.Embedding(len(known_ids) + 1, output_dim)

    def forward(self, ids, valid):
        in_range = valid & (ids > 0) & (ids < self.lookup.numel())
        safe_ids = torch.where(in_range, ids, 0)
        return self.embedding(self.lookup[safe_ids])


class TongjiInputEncoder(nn.Module):
    """Convert a collated Tongji batch into vectors for downstream modules."""

    @classmethod
    def from_config(cls, config):
        if not isinstance(config, Mapping):
            config = read_yaml(config)
        data = config["data"]
        root = processed_data_path(data["root"])
        training_cells = data["cell_ids"]["train"]
        normalization = fit_normalization(root, training_cells)

        schema = read_json(root / "schema.json")
        known_ids = [set() for _ in schema["static_categorical"]]
        for cell_id in training_cells:
            cell = load_cell(root / "cells" / cell_id)
            for index, field_ids in enumerate(known_ids):
                if cell["static_valid"][index]:
                    field_ids.add(int(cell["static_categorical"][index]))

        return cls(config["model"]["embedding"], normalization, known_ids)

    def __init__(self, config, normalization, known_ids):
        super().__init__()
        scalar = config["scalar"]
        self.numeric_dim = scalar["output_dim"]
        self.categorical_dim = config["categorical"]["output_dim"]
        self.observations = NumericFeatureEncoder(
            normalization["feature_mean"], normalization["feature_std"], **scalar
        )
        self.static_numeric = NumericFeatureEncoder(
            normalization["static_mean"], normalization["static_std"], **scalar
        )
        self.conditions = NumericFeatureEncoder(
            normalization["condition_mean"], normalization["condition_std"], **scalar
        )
        self.static_categorical = nn.ModuleList(
            CategoricalEncoder(field_ids, **config["categorical"])
            for field_ids in known_ids
        )
        self.time = ScalarEncoder(std=config["time_scale_s"], **scalar)
        self.cycle_distance = ScalarEncoder(
            std=config["cycle_distance_scale"], **scalar
        )

    def encode_observations(self, values, visible, point_valid):
        real_points = point_valid.unsqueeze(-1)
        encoded = self.observations(values, visible & real_points)
        return encoded.masked_fill(~real_points, 0)

    def encode_static_tokens(self, categories, numeric, valid):
        categorical_count = len(self.static_categorical)
        fields = [
            encoder(categories[..., index], valid[..., index])
            for index, encoder in enumerate(self.static_categorical)
        ]
        return torch.stack(fields, dim=-2), self.static_numeric.encode_tokens(
            numeric, valid[..., categorical_count:]
        )

    def encode_static(self, categories, numeric, valid):
        categorical, numeric = self.encode_static_tokens(categories, numeric, valid)
        return torch.cat((categorical.flatten(-2), numeric.flatten(-2)), dim=-1)

    def forward(self, batch):
        point_valid = batch["point_valid"]
        cycle_valid = batch["cycle_valid"]
        time = self.time(batch["time_in_cycle_s"], point_valid)
        distance = batch["last_cycle_id"][:, None] - batch["cycle_id"]
        cycle_distance = self.cycle_distance(distance, cycle_valid)
        return {
            "observations": self.encode_observations(
                batch["values"], batch["value_visible"], point_valid
            ),
            "static": self.encode_static(
                batch["static_categorical"],
                batch["static_numeric"],
                batch["static_valid"],
            ),
            "time": time.masked_fill(~point_valid.unsqueeze(-1), 0),
            "cycle_distance": cycle_distance.masked_fill(~cycle_valid.unsqueeze(-1), 0),
            "history_conditions": self.conditions(
                batch["history_u"], batch["history_u_valid"]
            ),
            "future_conditions": self.conditions(
                batch["future_u"], batch["future_u_valid"]
            ),
        }
