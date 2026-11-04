"""Random partial-charge observations and future SOH targets for PyTorch.

Disk arrays remain in physical units. Fit any normalization on training cells
only. ``deterministic=True`` fixes the view for each index (validation/test).
Otherwise repeated accesses draw new views; seed + epoch controls reproducibility.
"""
from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info


ARRAY_NAMES = ('values', 'value_valid', 'time_in_cycle', 'raw_sample_index',
               'cycle_offsets', 'cycle_id', 'static_categorical', 'static_numeric',
               'static_valid', 'conditions', 'conditions_valid', 'soh', 'soh_valid')


def load_cell(directory):
    """Memory-map one preprocessed cell; never alter these arrays in place."""
    directory = Path(directory)
    return {name: np.load(directory / f'{name}.npy', mmap_mode='r', allow_pickle=False)
            for name in ARRAY_NAMES}


def tensor(array):
    # Copy prevents tensors from writing into read-only memory-mapped data.
    return torch.from_numpy(np.array(array, copy=True))


class TongjiDataset(Dataset):
    """One index identifies a cell and a view slot, not an independent cell.

    Cell selection must be split BEFORE constructing train/validation/test
    datasets. An anchor k is always observed. Other observations are sampled
    from recorded cycles <= k. Targets are k+1..k+horizon by TRUE cycle ID,
    not the next stored rows. Missing future labels remain invalid.

    Future conditions are the known constant experimental protocol, never
    extracted from future sensor measurements. History conditions include the
    unobserved cycles between the earliest selected cycle and the anchor.
    With persistent DataLoader workers, recreate the loader after set_epoch().
    """
    def __init__(self, root='data/tongji_v1', cell_ids=None, *, samples_per_cell=128,
                 history_cycles=(1, 8), segment_points=(16, 128), horizon=32,
                 max_anchor_cycle=100, channel_mask_probability=0.0,
                 seed=0, deterministic=False, cache_size=2):
        self.root = Path(root)
        manifest = json.loads((self.root / 'manifest.json').read_text(encoding='utf-8'))
        if not manifest.get('complete'):
            raise ValueError('Preprocessing is incomplete')
        known = {cell['cell_id'] for cell in manifest['cells']}
        self.cell_ids = sorted(known if cell_ids is None else cell_ids)
        if not self.cell_ids or len(set(self.cell_ids)) != len(self.cell_ids):
            raise ValueError('cell_ids must be nonempty and unique')
        if not set(self.cell_ids) <= known:
            raise ValueError(f'Unknown cells: {set(self.cell_ids) - known}')
        for bounds in [history_cycles, segment_points]:
            if len(bounds) != 2 or not 1 <= bounds[0] <= bounds[1]:
                raise ValueError('Sampling bounds must satisfy 1 <= minimum <= maximum')
        if samples_per_cell < 1 or horizon < 1 or cache_size < 1 or seed < 0:
            raise ValueError('Positive sizes and a nonnegative seed are required')
        if not 0 <= channel_mask_probability <= 1:
            raise ValueError('channel_mask_probability must be in [0, 1]')
        self.samples_per_cell = samples_per_cell
        self.history_cycles = tuple(history_cycles)
        self.segment_points = tuple(segment_points)
        self.horizon = horizon
        self.max_anchor_cycle = max_anchor_cycle
        self.channel_mask_probability = channel_mask_probability
        self.seed = seed
        self.deterministic = deterministic
        self.cache_size = cache_size
        self.epoch = 0
        self._cache = OrderedDict()
        self._rng = None
        self._rng_identity = None
        self._anchors = {}
        self._eligible_history = {}
        for cell_id in self.cell_ids:
            cell = self._load(cell_id)
            ids = cell['cycle_id']
            eligible = np.flatnonzero(np.diff(cell['cycle_offsets']) >= segment_points[0])
            valid_targets = ids[cell['soh_valid']]
            anchors = []
            for row in eligible:
                k = int(ids[row])
                if max_anchor_cycle is not None and k > max_anchor_cycle:
                    continue
                if np.count_nonzero(eligible <= row) < history_cycles[0]:
                    continue
                if np.any((valid_targets > k) & (valid_targets <= k + horizon)):
                    anchors.append(row)
            if not anchors:
                raise ValueError(f'{cell_id}: no eligible anchors under the configured limits')
            self._anchors[cell_id] = np.array(anchors, dtype=np.int64)
            self._eligible_history[cell_id] = eligible

    def _load(self, cell_id):
        if cell_id not in self._cache:
            self._cache[cell_id] = load_cell(self.root / 'cells' / cell_id)
            if len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        self._cache.move_to_end(cell_id)
        return self._cache[cell_id]

    def close(self):
        """Release cached memory maps (important before file cleanup on Windows)."""
        self._cache.clear()

    def __getstate__(self):
        state = self.__dict__.copy()
        state.update(_cache=OrderedDict(), _rng=None, _rng_identity=None)
        return state

    def set_epoch(self, epoch):
        if epoch < 0:
            raise ValueError('epoch must be nonnegative')
        self.epoch = epoch
        self._rng = None

    def _generator(self, index):
        if self.deterministic:
            return np.random.default_rng(np.random.SeedSequence([self.seed, index]))
        worker = get_worker_info()
        worker_seed = worker.seed if worker else 0
        identity = (self.epoch, worker_seed)
        if self._rng is None or self._rng_identity != identity:
            self._rng = np.random.default_rng(np.random.SeedSequence(
                [self.seed, self.epoch, worker_seed]))
            self._rng_identity = identity
        return self._rng

    def __len__(self):
        return len(self.cell_ids) * self.samples_per_cell

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(index)
        rng = self._generator(index)
        cell_id = self.cell_ids[index // self.samples_per_cell]
        cell = self._load(cell_id)
        anchor_row = int(rng.choice(self._anchors[cell_id]))
        anchor = int(cell['cycle_id'][anchor_row])
        previous = self._eligible_history[cell_id]
        previous = previous[previous < anchor_row]
        count = int(rng.integers(self.history_cycles[0],
                                min(self.history_cycles[1], len(previous) + 1) + 1))
        rows = np.sort(np.r_[rng.choice(previous, count - 1, replace=False), anchor_row]).astype(int)
        values, validity, visibility, times, raw_indices, segment_bounds = [], [], [], [], [], []
        for row in rows:
            start, end = cell['cycle_offsets'][row:row + 2]
            length = int(rng.integers(self.segment_points[0],
                                     min(self.segment_points[1], end - start) + 1))
            local_start = int(rng.integers(0, end - start - length + 1))
            selection = slice(start + local_start, start + local_start + length)
            v = tensor(cell['values'][selection])
            valid = tensor(cell['value_valid'][selection])
            # Whole-channel masking per selected segment. Natural validity stays intact.
            keep = rng.random(v.shape[-1]) >= self.channel_mask_probability
            visible = tensor(np.broadcast_to(keep, v.shape))
            # Hidden measurements are scrubbed; raw numerical values cannot leak.
            v[~(valid & visible)] = 0
            values.append(v)
            validity.append(valid)
            visibility.append(visible)
            times.append(tensor(cell['time_in_cycle'][selection]))
            raw_indices.append(tensor(cell['raw_sample_index'][selection]))
            segment_bounds.append([local_start, local_start + length])

        target_ids = anchor + np.arange(1, self.horizon + 1, dtype=np.int64)
        positions = np.searchsorted(cell['cycle_id'], target_ids)
        safe_positions = np.minimum(positions, len(cell['cycle_id']) - 1)
        recorded = (positions < len(cell['cycle_id'])) & (cell['cycle_id'][safe_positions] == target_ids)
        target_valid = recorded & cell['soh_valid'][safe_positions]
        target = np.where(target_valid, cell['soh'][safe_positions], 0).astype(np.float32)
        # The preprocessing contract declares protocol setpoints constant per cell.
        protocol = cell['conditions'][anchor_row]
        protocol_valid = cell['conditions_valid'][anchor_row]
        history_ids = np.arange(int(cell['cycle_id'][rows[0]]), anchor, dtype=np.int64)
        return dict(cell_id=cell_id, anchor_cycle=anchor,
                    values=values, value_valid=validity, value_visible=visibility,
                    time_in_cycle=times, raw_sample_index=raw_indices,
                    cycle_id=tensor(cell['cycle_id'][rows]),
                    segment_bounds=torch.tensor(segment_bounds, dtype=torch.int64),
                    static_categorical=tensor(cell['static_categorical']),
                    static_numeric=tensor(cell['static_numeric']),
                    static_valid=tensor(cell['static_valid']),
                    history_cycle_id=tensor(history_ids),
                    history_u=tensor(np.tile(protocol, (len(history_ids), 1))),
                    history_u_valid=tensor(np.tile(protocol_valid, (len(history_ids), 1))),
                    future_u=tensor(np.tile(protocol, (self.horizon, 1))),
                    future_u_valid=tensor(np.tile(protocol_valid, (self.horizon, 1))),
                    target_cycle_id=tensor(target_ids), target_soh=tensor(target),
                    target_valid=tensor(target_valid))


def collate_tongji(samples):
    """Batch-local zero padding. True always means valid/visible.

    cycle_valid is derived from point_valid; it is not an independent mask.
    A genuine point with no valid channels remains a genuine point.
    """
    if not samples:
        raise ValueError('Cannot collate an empty batch')
    batch_size = len(samples)
    n = max(len(s['values']) for s in samples)
    t = max(len(v) for s in samples for v in s['values'])
    f = samples[0]['values'][0].shape[-1]
    batch = dict(cell_id=[s['cell_id'] for s in samples],
                 values=torch.zeros(batch_size, n, t, f),
                 value_valid=torch.zeros(batch_size, n, t, f, dtype=torch.bool),
                 value_visible=torch.zeros(batch_size, n, t, f, dtype=torch.bool),
                 point_valid=torch.zeros(batch_size, n, t, dtype=torch.bool),
                 time_in_cycle=torch.zeros(batch_size, n, t, dtype=torch.float64),
                 raw_sample_index=torch.full((batch_size, n, t), -1, dtype=torch.int64),
                 cycle_id=torch.zeros(batch_size, n, dtype=torch.int64),
                 segment_bounds=torch.zeros(batch_size, n, 2, dtype=torch.int64))
    for b, sample in enumerate(samples):
        count = len(sample['values'])
        batch['cycle_id'][b, :count] = sample['cycle_id']
        batch['segment_bounds'][b, :count] = sample['segment_bounds']
        for j, value in enumerate(sample['values']):
            length = len(value)
            batch['point_valid'][b, j, :length] = True
            for key in ['values', 'value_valid', 'value_visible', 'time_in_cycle', 'raw_sample_index']:
                batch[key][b, j, :length] = sample[key][j]
    batch['cycle_valid'] = batch['point_valid'].any(dim=-1)
    history_length = max(len(s['history_cycle_id']) for s in samples)
    du = samples[0]['future_u'].shape[-1]
    batch['history_cycle_id'] = torch.zeros(batch_size, history_length, dtype=torch.int64)
    batch['history_u'] = torch.zeros(batch_size, history_length, du)
    batch['history_u_valid'] = torch.zeros(batch_size, history_length, du, dtype=torch.bool)
    for b, sample in enumerate(samples):
        length = len(sample['history_cycle_id'])
        for key in ['history_cycle_id', 'history_u', 'history_u_valid']:
            batch[key][b, :length] = sample[key]
    for key in ['static_categorical', 'static_numeric', 'static_valid', 'future_u',
                'future_u_valid', 'target_cycle_id', 'target_soh', 'target_valid']:
        batch[key] = torch.stack([s[key] for s in samples])
    batch['anchor_cycle'] = torch.tensor([s['anchor_cycle'] for s in samples])
    return batch
