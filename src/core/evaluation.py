"""Fixed-view evaluation and shareable local figures; no test-time fitting."""
import csv
from pathlib import Path

import numpy as np
import torch

from .module import trajectory_errors


@torch.no_grad()
def evaluate(module, loader, limit_batches=None):
    module.eval()
    device = next(module.parameters()).device
    rows = {key: [] for key in [
        "last_cycle_position", "last_cycle_id", "prediction", "target_soh",
        "target_soh_valid", "target_cycle_id", "mse", "mae",
    ]}
    for index, batch in enumerate(loader):
        if limit_batches is not None and index >= limit_batches:
            break
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        prediction = module(batch)
        mse, mae = trajectory_errors(prediction, batch['target_soh'], batch['target_soh_valid'])
        if not torch.isfinite(prediction).all():
            raise FloatingPointError('Nonfinite evaluation output')
        tensors = {
            "last_cycle_position": batch["last_cycle_position"],
            "last_cycle_id": batch["last_cycle_id"],
            "prediction": prediction,
            "target_soh": batch["target_soh"],
            "target_soh_valid": batch["target_soh_valid"],
            "target_cycle_id": batch["target_cycle_id"],
            "mse": mse,
            "mae": mae,
        }
        for key, value in tensors.items():
            rows[key].append(value.cpu().numpy())
    arrays = {key: np.concatenate(value) for key, value in rows.items()}
    mse = float(arrays['mse'].mean())
    metrics = {
        "mse": mse,
        "rmse": mse**0.5,
        "mae": float(arrays["mae"].mean()),
        "samples": len(arrays["last_cycle_position"]),
        "valid_target_points": int(arrays["target_soh_valid"].sum()),
    }
    return metrics, arrays


def plot_results(run_dir, predictions):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    run_dir = Path(run_dir)
    csv_path = run_dir / 'csv' / 'main' / 'metrics.csv'
    history = {}
    with csv_path.open(newline='', encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            if not row.get('epoch'):
                continue
            epoch = int(float(row['epoch']))
            for key in ['train/mse', 'val/mse']:
                if row.get(key):
                    history.setdefault(key, {})[epoch] = float(row[key])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for name, points in history.items():
        axes[0].plot(list(points), list(points.values()), '.-', label=name)
    axes[0].set(xlabel='Epoch (zero-based)', ylabel='Trajectory MSE', title='Training and fixed-view validation')
    axes[0].set_yscale('log')
    axes[0].legend()
    for stage, arrays in predictions.items():
        valid = arrays['target_soh_valid']
        axes[1].scatter(arrays['target_soh'][valid][::4], arrays['prediction'][valid][::4],
                        s=5, alpha=.35, label=stage)
    axes[1].plot([.65, 1.0], [.65, 1.0], 'k--', linewidth=1)
    axes[1].set(xlabel='Measured SOH', ylabel='Predicted SOH', title='Best checkpoint, fixed evaluation views')
    axes[1].legend()
    fig.savefig(run_dir / 'learning_curves.png', dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
    for row, stage in enumerate(['val', 'test']):
        arrays = predictions[stage]
        order = np.argsort(arrays['last_cycle_position'])
        indices = order[np.linspace(0, len(order)-1, 3).astype(int)]
        for ax, index in zip(axes[row], indices):
            valid = arrays['target_soh_valid'][index]
            x = arrays['target_cycle_id'][index]
            ax.plot(x, arrays['prediction'][index], label='Prediction')
            ax.plot(x[valid], arrays['target_soh'][index, valid], '.-', label='Measured valid SOH')
            ax.set(title=f'{stage}, last cycle {arrays["last_cycle_id"][index]}', xlabel='True cycle ID', ylabel='SOH')
            ax.legend(fontsize=8)
    fig.savefig(run_dir / 'trajectories.png', dpi=140)
    plt.close(fig)
