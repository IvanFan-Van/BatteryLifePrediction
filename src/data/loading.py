"""Cell-level selection, explicit anchor splits, and training-only scaling."""
import numpy as np
from torch.utils.data import DataLoader

from src.data import (
    ObservationSchedule,
    TongjiDataset,
    TongjiTaskDataset,
    TongjiTaskSampler,
    collate_tongji,
    load_cell,
)
from src.utils.files import read_json
from src.utils.paths import processed_data_path


def fit_normalization(root, training_cell_ids):
    """Fit on training cells only, never on SOH or the held-out test cell."""
    cells_dir = processed_data_path(root) / 'cells'
    groups = [('feature', 'values', 'value_valid'),
              ('static', 'static_numeric', 'static_valid'),
              ('condition', 'conditions', 'conditions_valid')]
    result = {}
    for prefix, key, valid_key in groups:
        sums = squares = counts = None
        for cell_id in training_cell_ids:
            cell = load_cell(cells_dir / cell_id)
            x = np.asarray(cell[key], dtype=np.float64)
            valid = np.asarray(cell[valid_key])
            if prefix == 'static':
                x = x[None]
                valid = valid[-x.shape[-1]:][None]
            safe = np.where(valid, x, 0)
            if sums is None:
                sums = np.zeros(x.shape[-1]); squares = sums.copy(); counts = sums.copy()
            sums += safe.sum(axis=0)
            squares += (safe * safe).sum(axis=0)
            counts += valid.sum(axis=0)
        mean = sums / np.maximum(counts, 1)
        variance = np.maximum(squares / np.maximum(counts, 1) - mean**2, 0)
        std = np.sqrt(variance)
        std[std < 1e-6] = 1.0
        result[f'{prefix}_mean'] = mean.tolist()
        result[f'{prefix}_std'] = std.tolist()
    return result


def build_datasets(config, saved_splits=None):
    data = config['data']
    shared = dict(root=data['root'], horizon=data['horizon'])
    if 'observation_schedule' in config:
        schema = read_json(processed_data_path(data['root']) / 'schema.json')
        schedule = ObservationSchedule.from_config(config, feature_count=len(schema['features']))
        dataset_type = TongjiTaskDataset
        shared.update(schedule=schedule, max_last_observed_cycle=data['max_anchor_cycle'])
        train_options, eval_options = {}, {}
    else:
        dataset_type = TongjiDataset
        shared.update(history_cycles=tuple(data['history_cycles']),
                      segment_points=tuple(data['segment_points']),
                      max_anchor_cycle=data['max_anchor_cycle'], enumerate_anchors=True)
        train_options = dict(channel_mask_probability=data['channel_mask_probability'])
        eval_options = dict(deterministic=True)
    train_cell, test_cell = data['train_cell'], data['test_cell']
    if train_cell == test_cell:
        raise ValueError('Test cell must be independent of the training cell')
    if saved_splits is None:
        candidate = dataset_type(**shared, cell_ids=[train_cell], **eval_options)
        anchors = candidate.anchor_cycles(train_cell)
        candidate.close()
        rng = np.random.default_rng(data['split_seed'])
        shuffled = rng.permutation(anchors)
        nval = max(1, int(round(len(anchors) * data['val_ratio'])))
        if nval >= len(anchors):
            raise ValueError('Not enough anchors for train/validation split')
        saved_splits = dict(train_cell=train_cell, test_cell=test_cell,
                            train_anchors=sorted(shuffled[nval:].tolist()),
                            val_anchors=sorted(shuffled[:nval].tolist()),
                            note='Same-cell held-out anchors for validation. Historical observations and future labels may overlap; this is smoke validation, not independent-cell validation.')
    if set(saved_splits['train_anchors']) & set(saved_splits['val_anchors']):
        raise ValueError('Train and validation anchor IDs overlap')
    train_anchors = {train_cell: saved_splits['train_anchors']}
    val_anchors = {train_cell: saved_splits['val_anchors']}
    train = dataset_type(**shared, cell_ids=[train_cell], anchor_cycle_ids=train_anchors,
                         seed=config['seed'], **train_options)
    train_eval = dataset_type(**shared, cell_ids=[train_cell], anchor_cycle_ids=train_anchors,
                              seed=data['eval_seed'], **eval_options)
    val = dataset_type(**shared, cell_ids=[train_cell], anchor_cycle_ids=val_anchors,
                       seed=data['eval_seed'], **eval_options)
    test = dataset_type(**shared, cell_ids=[test_cell], seed=data['eval_seed'], **eval_options)
    return dict(train=train, train_eval=train_eval, val=val, test=test), saved_splits


def make_loader(dataset, config, shuffle=False):
    sampler = None
    if isinstance(dataset, TongjiTaskDataset):
        # Use the final observation policy for every fixed-view evaluation.
        fixed_epoch = None if shuffle else dataset.schedule.max_epochs - 1
        sampler = TongjiTaskSampler(dataset, seed=config['seed'], fixed_epoch=fixed_epoch)
        shuffle = False
    return DataLoader(dataset, batch_size=config['data']['batch_size'], shuffle=shuffle,
                      sampler=sampler, num_workers=config['data']['num_workers'],
                      collate_fn=collate_tongji, persistent_workers=False)


def vocabulary_sizes(root):
    root = processed_data_path(root)
    mapping = read_json(root / 'category_mapping.json')
    schema = read_json(root / 'schema.json')
    return [len(mapping[field]) for field in schema['static_categorical']]
