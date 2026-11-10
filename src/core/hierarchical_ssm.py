"""Small hierarchical-attention state-space model.

Static tokens always fuse at the cycle level. The 'early' ablation also reuses
the same tokens in within-cycle attention, without adding parameters.
"""
import torch
from torch import nn


class ScalarEncoder(nn.Module):
    """A field-specific 1 -> 8 -> output nonlinear map and learned UNK."""
    def __init__(self, output_dim, hidden_dim=8):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(1, hidden_dim), nn.SiLU(),
                                     nn.Linear(hidden_dim, output_dim))
        self.unk = nn.Parameter(torch.randn(output_dim) * 0.02)

    def forward(self, values, valid, mean=0.0, std=1.0):
        safe = torch.where(valid, values, torch.as_tensor(mean, device=values.device))
        encoded = self.network(((safe - mean) / std).unsqueeze(-1))
        return torch.where(valid.unsqueeze(-1), encoded, self.unk)


class HierarchicalStateSpaceModel(nn.Module):
    feature_mean: torch.Tensor
    feature_std: torch.Tensor
    static_mean: torch.Tensor
    static_std: torch.Tensor
    condition_mean: torch.Tensor
    condition_std: torch.Tensor

    def __init__(self, vocabulary_sizes, config, normalization):
        super().__init__()
        if config['static_fusion'] not in ('cycle', 'early'):
            raise ValueError('static_fusion must be cycle or early')
        self.config = config
        d, z = config['d_model'], config['state_dim']
        feature_dim = config['feature_dim']
        scalar_hidden = config['scalar_hidden']
        self.d_model = d
        for name, values in normalization.items():
            self.register_buffer(name, torch.tensor(values, dtype=torch.float32))
        self.observation_encoders = nn.ModuleList(
            [ScalarEncoder(feature_dim, scalar_hidden) for _ in range(5)])
        self.time_encoder = nn.Sequential(nn.Linear(1, scalar_hidden), nn.SiLU(),
                                          nn.Linear(scalar_hidden, config['time_dim']))
        self.point_projection = nn.Sequential(
            nn.Linear(5 * feature_dim + config['time_dim'], d), nn.SiLU(), nn.LayerNorm(d))
        self.cycle_token = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.state_token = nn.Parameter(torch.randn(1, 1, d) * 0.02)

        def block():
            return nn.TransformerEncoderLayer(
                d_model=d, nhead=config['heads'], dim_feedforward=config['feedforward_dim'],
                dropout=config['dropout'], activation='gelu', batch_first=True, norm_first=True)
        self.point_blocks = nn.ModuleList([block() for _ in range(config['point_layers'])])
        self.cycle_blocks = nn.ModuleList([block() for _ in range(config['cycle_layers'])])
        # Every field has its own table/network, including its own UNK parameters.
        self.categorical_encoders = nn.ModuleList([nn.Embedding(size, d) for size in vocabulary_sizes])
        self.numeric_encoders = nn.ModuleList([ScalarEncoder(d, scalar_hidden) for _ in range(3)])
        self.static_fields = nn.Parameter(torch.randn(len(vocabulary_sizes) + 3, d) * 0.02)
        self.token_types = nn.Parameter(torch.randn(2, d) * 0.02)
        self.static_context = nn.Sequential(nn.Linear(d, config['context_dim']), nn.SiLU())
        self.static_query = nn.Linear(config['context_dim'], d)
        self.condition_encoders = nn.ModuleList([ScalarEncoder(feature_dim, scalar_hidden) for _ in range(4)])
        self.condition_projection = nn.Sequential(
            nn.Linear(4 * feature_dim, config['context_dim']), nn.SiLU())
        self.history_projection = nn.Linear(config['context_dim'], d)
        self.cycle_position = nn.Sequential(nn.Linear(2, 16), nn.SiLU(), nn.Linear(16, d))
        self.state_projection = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, z))
        self.dynamics = nn.Sequential(
            nn.Linear(z + 2 * config['context_dim'], config['dynamics_hidden']), nn.SiLU(),
            nn.Linear(config['dynamics_hidden'], z))
        dynamics_output = self.dynamics[-1]
        assert isinstance(dynamics_output, nn.Linear) and dynamics_output.bias is not None
        nn.init.normal_(dynamics_output.weight, std=0.001)
        nn.init.zeros_(dynamics_output.bias)
        self.head = nn.Sequential(nn.Linear(z + config['context_dim'], config['head_hidden']),
                                  nn.SiLU(), nn.Linear(config['head_hidden'], 1))
        head_output = self.head[-1]
        assert isinstance(head_output, nn.Linear) and head_output.bias is not None
        nn.init.constant_(head_output.bias, config['initial_soh_bias'])

    def encode_static(self, batch):
        tokens = []
        categories, valid = batch['static_categorical'], batch['static_valid']
        ncat = len(self.categorical_encoders)
        for field, encoder in enumerate(self.categorical_encoders):
            ids = categories[:, field]
            known = valid[:, field] & (ids >= 0) & (ids < encoder.num_embeddings)
            ids = torch.where(known, ids, 0)
            tokens.append(encoder(ids))
        for field, encoder in enumerate(self.numeric_encoders):
            tokens.append(encoder(batch['static_numeric'][:, field], valid[:, ncat + field],
                                  self.static_mean[field], self.static_std[field]))
        tokens = torch.stack(tokens, dim=1) + self.static_fields + self.token_types[0]
        return tokens, self.static_context(tokens.mean(dim=1))

    def encode_conditions(self, values, valid):
        encoded = [encoder(values[..., f], valid[..., f], self.condition_mean[f], self.condition_std[f])
                   for f, encoder in enumerate(self.condition_encoders)]
        return self.condition_projection(torch.cat(encoded, dim=-1))

    def encode(self, batch):
        values = batch['values']
        point_valid = batch['point_valid']
        use = batch['value_valid'] & batch['value_visible'] & point_valid.unsqueeze(-1)
        features = [encoder(values[..., f], use[..., f], self.feature_mean[f], self.feature_std[f])
                    for f, encoder in enumerate(self.observation_encoders)]
        time = torch.where(point_valid, batch['time_in_cycle_s'], 0).to(values.dtype)
        time = self.time_encoder((time / self.config['time_scale_s']).unsqueeze(-1))
        points = self.point_projection(torch.cat([*features, time], dim=-1))
        points = torch.where(point_valid.unsqueeze(-1), points, 0)
        cycle_valid = point_valid.any(dim=-1)
        if not cycle_valid.any(dim=1).all():
            raise ValueError('Every sample needs at least one real cycle')
        static_tokens, context = self.encode_static(batch)
        packed = points[cycle_valid]
        packed_valid = point_valid[cycle_valid]
        cls = self.cycle_token.expand(len(packed), -1, -1)
        if self.config['static_fusion'] == 'early':
            owners = torch.arange(len(values), device=values.device)[:, None].expand_as(cycle_valid)[cycle_valid]
            packed_static = static_tokens[owners]
            sequence = torch.cat([cls, packed_static, packed], dim=1)
            prefix = torch.zeros(len(packed), 1 + packed_static.shape[1], dtype=torch.bool, device=values.device)
        else:
            sequence = torch.cat([cls, packed], dim=1)
            prefix = torch.zeros_like(packed_valid[:, :1])
        padding = torch.cat([prefix, ~packed_valid], dim=1)
        for block in self.point_blocks:
            sequence = block(sequence, src_key_padding_mask=padding)
        cycles = values.new_zeros(*cycle_valid.shape, self.d_model)
        cycles[cycle_valid] = sequence[:, 0]
        ids = torch.where(cycle_valid, batch['cycle_id'], 0).to(values.dtype)
        ages = torch.where(cycle_valid, batch['last_cycle_id'][:, None] - batch['cycle_id'], 0).to(values.dtype)
        position = torch.stack([ids, ages], dim=-1) / self.config['cycle_scale']
        cycles = cycles + self.cycle_position(position) + self.token_types[1]

        history = self.encode_conditions(batch['history_u'], batch['history_u_valid'])
        history_valid = batch['history_u_valid'].any(dim=-1)
        history = (history * history_valid.unsqueeze(-1)).sum(dim=1)
        history = history / history_valid.sum(dim=1, keepdim=True).clamp_min(1)
        state = self.state_token.expand(len(values), -1, -1)
        state = state + self.static_query(context).unsqueeze(1) + self.history_projection(history).unsqueeze(1)
        sequence = torch.cat([state, static_tokens, cycles], dim=1)
        prefix_valid = torch.ones(len(values), 1 + static_tokens.shape[1], dtype=torch.bool, device=values.device)
        padding = ~torch.cat([prefix_valid, cycle_valid], dim=1)
        for block in self.cycle_blocks:
            sequence = block(sequence, src_key_padding_mask=padding)
        return self.state_projection(sequence[:, 0]), context

    def forward(self, batch, return_states=False):
        state, context = self.encode(batch)
        conditions = self.encode_conditions(batch['future_u'], batch['future_u_valid'])
        predictions, states = [], []
        for step in range(conditions.shape[1]):
            delta = self.dynamics(torch.cat([state, conditions[:, step], context], dim=-1))
            state = state + self.config['transition_scale'] * delta
            predictions.append(self.head(torch.cat([state, context], dim=-1)).squeeze(-1))
            if return_states:
                states.append(state)
        prediction = torch.stack(predictions, dim=1)
        if return_states:
            return prediction, torch.stack(states, dim=1)
        return prediction
