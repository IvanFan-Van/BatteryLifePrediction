"""Convert trusted BatteryLife Tongji pickles into an immutable NumPy dataset.

Run from the repository root:
    python scripts/preprocess_tongji.py --output data/tongji_v1
No interpolation, normalization, or raw-file modification is performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import re
from pathlib import Path

import numpy as np

FEATURES = ['voltage_in_V', 'current_in_A', 'charge_capacity_in_Ah',
            'discharge_capacity_in_Ah', 'temperature_in_C']
CATEGORICAL = ['form_factor', 'cathode_material', 'anode_material',
               'electrolyte_material', 'separator_material']
NUMERIC = ['nominal_capacity_in_Ah', 'max_voltage_limit_in_V', 'min_voltage_limit_in_V']
CONDITIONS = ['ambient_temperature_in_C', 'charge_rate_in_C',
              'discharge_rate_in_C', 'post_charge_rest_in_s']
REFERENCE = 'https://www.nature.com/articles/s41467-022-29837-w'


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def rate(token):
    """Decode the published filename notation: 025=0.25, 05=0.5."""
    if token.startswith('0') and '.' not in token:
        return float('0.' + token[1:])
    return float(token)


def protocol(cell_id):
    match = re.fullmatch(r'Tongji([123])_CY(\d+)-([\d.]+)_([\d.]+)--(\d+)', cell_id)
    if match is None:
        raise ValueError(f'Unrecognized Tongji filename: {cell_id}')
    group, temperature, charge, discharge, _ = match.groups()
    return np.array([float(temperature), rate(charge), rate(discharge),
                     3600.0 if group == '3' else 1800.0], dtype=np.float32)


def array_field(cycle, key, size):
    value = cycle.get(key)
    if value is None:
        return np.full(size, np.nan, dtype=np.float64)
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,):
        raise ValueError(f'{key}: expected {(size,)}, got {result.shape}')
    return result


def longest_run(mask):
    """Half-open bounds of the longest contiguous True run; first run wins ties."""
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    if not len(starts):
        return 0, 0
    best = np.argmax(ends - starts)
    return int(starts[best]), int(ends[best])


def extract_cycle(cycle, nominal_capacity, cutoff, current_threshold_c=0.005,
                  cutoff_tolerance_v=0.05):
    """Keep the longest positive-current charge run; audit label reliability.

    A small C-rate threshold rejects idle-current noise. SOH is the positive
    maximum cumulative discharge capacity / nominal capacity, only if discharge
    starts near zero capacity and reaches the specified cutoff voltage.
    """
    current = np.asarray(cycle['current_in_A'], dtype=np.float64)
    if current.ndim != 1:
        raise ValueError('current_in_A must be one-dimensional')
    size = len(current)
    raw = np.column_stack([array_field(cycle, key, size) for key in FEATURES])
    time = array_field(cycle, 'time_in_s', size)
    finite_time = np.isfinite(time)
    if not finite_time.all() or np.any(np.diff(time) < 0):
        raise ValueError('Time must be finite and nondecreasing')
    threshold = nominal_capacity * current_threshold_c
    start, end = longest_run(np.isfinite(current) & (current > threshold))
    selected = raw[start:end]
    valid = np.isfinite(selected)
    values = np.where(valid, selected, 0).astype(np.float32)
    # Reference the full cycle start, not the randomly cropped segment start.
    relative_time = time[start:end] - (time[0] if size else 0)

    discharging = np.isfinite(current) & (current < -threshold)
    reasons = []
    capacity = 0.0
    if np.count_nonzero(discharging) < 2:
        reasons.append('no_discharge_segment')
    else:
        q = raw[discharging, 3]
        voltage = raw[discharging, 0]
        if not np.isfinite(q).all() or not np.isfinite(voltage).all():
            reasons.append('nonfinite_discharge')
        else:
            capacity = float(np.max(q))
            if capacity <= 0:
                reasons.append('nonpositive_discharge_capacity')
            if abs(float(q[0])) > 0.05 * nominal_capacity:
                reasons.append('discharge_counter_not_near_zero')
            if np.any(np.diff(q) < -0.001 * nominal_capacity):
                reasons.append('discharge_counter_decreases')
            if not np.isfinite(cutoff) or np.min(voltage) > cutoff + cutoff_tolerance_v:
                reasons.append('discharge_cutoff_not_reached')
    soh_valid = not reasons
    return dict(values=values, value_valid=valid,
                time_in_cycle=relative_time.astype(np.float64),
                raw_sample_index=np.arange(start, end, dtype=np.int64),
                soh=capacity / nominal_capacity if soh_valid else 0.0,
                soh_valid=soh_valid, label_issues=reasons)


def process_cell(path, destination, mappings, args):
    payload = path.read_bytes()
    source_sha256 = hashlib.sha256(payload).hexdigest()
    battery = pickle.loads(payload)
    del payload
    nominal = float(battery['nominal_capacity_in_Ah'])
    if not np.isfinite(nominal) or nominal <= 0:
        raise ValueError(f'{path.name}: invalid nominal capacity')
    cycles = sorted(battery['cycle_data'], key=lambda c: c['cycle_number'])
    ids = np.array([c['cycle_number'] for c in cycles], dtype=np.int64)
    if not len(ids) or np.any(np.diff(ids) <= 0):
        raise ValueError(f'{path.name}: empty or duplicate cycle IDs')
    # Always use the source filename as the directory identity.
    cell_id = path.stem
    destination.mkdir()
    outputs = [extract_cycle(c, nominal, float(battery['min_voltage_limit_in_V']),
                             args.current_threshold_c, args.cutoff_tolerance_v)
               for c in cycles]
    offsets = np.r_[0, np.cumsum([len(c['values']) for c in outputs])].astype(np.int64)
    for key in ['values', 'value_valid', 'time_in_cycle', 'raw_sample_index']:
        np.save(destination / f'{key}.npy', np.concatenate([c[key] for c in outputs]))
    for key, array in dict(cycle_id=ids, cycle_offsets=offsets,
                           soh=np.array([c['soh'] for c in outputs], np.float32),
                           soh_valid=np.array([c['soh_valid'] for c in outputs], bool)).items():
        np.save(destination / f'{key}.npy', array)
    static_cat, static_num, static_valid = [], [], []
    for key in CATEGORICAL:
        value = battery.get(key)
        known = isinstance(value, str) and bool(value.strip())
        vocabulary = mappings[key]
        if known and value not in vocabulary:
            vocabulary[value] = len(vocabulary)
        static_cat.append(vocabulary[value] if known else 0)
        static_valid.append(known)
    for key in NUMERIC:
        value = battery.get(key)
        known = value is not None and np.isfinite(value)
        static_num.append(float(value) if known else 0.0)
        static_valid.append(bool(known))
    arrays = dict(static_categorical=np.array(static_cat, np.int64),
                  static_numeric=np.array(static_num, np.float32),
                  static_valid=np.array(static_valid, bool),
                  conditions=np.tile(protocol(cell_id), (len(ids), 1)),
                  conditions_valid=np.ones((len(ids), len(CONDITIONS)), bool))
    for key, array in arrays.items():
        np.save(destination / f'{key}.npy', array)
    issues = {str(ids[i]): c['label_issues'] for i, c in enumerate(outputs) if c['label_issues']}
    metadata = dict(cell_id=cell_id, source_file=path.name, source_sha256=source_sha256,
                    condition_source=REFERENCE, condition_note='Protocol setpoints, not measured cell temperature; assumed constant per cell.',
                    time_note='BatteryLife normalized time; may compress original gaps. Not a raw calendar clock.',
                    static_raw={key: battery.get(key) for key in CATEGORICAL + NUMERIC},
                    cycle_count=len(ids), charge_points=int(offsets[-1]),
                    valid_soh_count=sum(c['soh_valid'] for c in outputs),
                    empty_charge_count=sum(len(c['values']) == 0 for c in outputs),
                    label_issues=issues)
    write_json(destination / 'metadata.json', metadata)
    return {key: metadata[key] for key in ['cell_id', 'cycle_count', 'charge_points',
                                         'valid_soh_count', 'empty_charge_count']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-dir', type=Path, default=Path('raw_data/Tongji'))
    parser.add_argument('--output', type=Path, default=Path('data/tongji_v1'))
    parser.add_argument('--current-threshold-c', type=float, default=0.005)
    parser.add_argument('--cutoff-tolerance-v', type=float, default=0.05)
    args = parser.parse_args()
    paths = sorted(args.raw_dir.glob('*.pkl'))
    if not paths:
        parser.error(f'No pickles in {args.raw_dir}')
    if args.current_threshold_c <= 0 or args.cutoff_tolerance_v < 0:
        parser.error('Current threshold must be positive; voltage tolerance nonnegative')
    if args.output.resolve().is_relative_to(args.raw_dir.resolve()):
        parser.error('Output must not be inside raw data')
    # Refuse to overwrite even an incomplete prior run; choose a new version path.
    args.output.mkdir(parents=True, exist_ok=False)
    cells_dir = args.output / 'cells'
    cells_dir.mkdir()
    mappings = {key: {'<UNK>': 0} for key in CATEGORICAL}
    manifest = []
    for index, path in enumerate(paths):
        result = process_cell(path, cells_dir / path.stem, mappings, args)
        manifest.append(result)
        print(f'[{index + 1}/{len(paths)}] {path.stem}: '
              f'{result["cycle_count"]} cycles, {result["valid_soh_count"]} SOH labels', flush=True)
    write_json(args.output / 'category_mapping.json', mappings)
    schema = dict(version=1, features=FEATURES, feature_units=['V', 'A', 'Ah', 'Ah', 'degC'],
                  static_categorical=CATEGORICAL, static_numeric=NUMERIC,
                  static_valid_order=CATEGORICAL + NUMERIC, conditions=CONDITIONS,
                  condition_units=['degC', 'C', 'C', 's'],
                  charge_extraction='Longest contiguous run with I > nominal_capacity * current_threshold_c',
                  current_threshold_c=args.current_threshold_c,
                  cutoff_tolerance_v=args.cutoff_tolerance_v,
                  soh_definition='max cumulative discharge capacity in qualifying discharge points / nominal capacity; no clipping or smoothing',
                  vocabulary_note='Dataset metadata vocabulary, not fitted numerical statistics; no scaling is applied.',
                  padding_value=0, raw_directory=str(args.raw_dir.resolve()))
    write_json(args.output / 'schema.json', schema)
    write_json(args.output / 'manifest.json', dict(complete=True, cells=manifest))
    print(f'Complete: {args.output.resolve()}', flush=True)


if __name__ == '__main__':
    main()
