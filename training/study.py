"""Shared protocol for the full-corpus Tongji fusion ablation."""
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import lightning as L
import numpy as np
import torch

from battery_data import TongjiDataset
from .module import BatteryTrajectoryModule, trajectory_errors


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def clean_environment():
    """Do not copy API credentials into Pueue's persisted task environment."""
    allowed = {'PATH','PATHEXT','SYSTEMROOT','WINDIR','COMSPEC','USERPROFILE','APPDATA',
               'LOCALAPPDATA','TEMP','TMP','USERNAME','USERDOMAIN','PROCESSOR_ARCHITECTURE',
               'NUMBER_OF_PROCESSORS','PROGRAMDATA','PROGRAMFILES','PROGRAMFILES(X86)',
               'COMMONPROGRAMFILES','VIRTUAL_ENV','UV_CACHE_DIR','UV_INTERNAL__PYTHONHOME',
               'PYTHONUTF8','PYTHONIOENCODING'}
    environment = {k:v for k,v in os.environ.items() if k.upper() in allowed}
    environment.update(WANDB_MODE='offline', WANDB_SILENT='true',
                       CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES', '0' if os.name=='nt' else '2'),
                       CUBLAS_WORKSPACE_CONFIG=':4096:8',
                       PYTHONUTF8='1', PYTHONIOENCODING='utf-8')
    return environment


def stratified_cell_split(root, seed=2026, val_ratio=.2, test_ratio=.2):
    manifest = json.loads((Path(root)/'manifest.json').read_text(encoding='utf-8'))
    groups = defaultdict(list)
    for cell in manifest['cells']:
        # Family + ambient temperature + charging/discharging protocol.
        groups[cell['cell_id'].split('--')[0]].append(cell['cell_id'])
    rng = np.random.default_rng(seed)
    split = dict(train=[], val=[], test=[], groups={})
    for group, cells in sorted(groups.items()):
        if len(cells) < 3:
            raise ValueError(f'{group} needs >=3 cells for a three-way stratified split')
        cells = rng.permutation(sorted(cells)).tolist()
        nval = max(1, int(round(len(cells)*val_ratio)))
        ntest = max(1, int(round(len(cells)*test_ratio)))
        if nval + ntest >= len(cells):
            nval = ntest = 1
        parts = dict(val=cells[:nval], test=cells[nval:nval+ntest], train=cells[nval+ntest:])
        split['groups'][group] = parts
        for name in ['train','val','test']:
            split[name].extend(parts[name])
    for name in ['train','val','test']:
        split[name].sort()
    if set(split['train']) & set(split['val']) or set(split['test']) & (set(split['train'])|set(split['val'])):
        raise AssertionError('Cell split overlap')
    split['seed'] = seed
    split['protocol'] = 'Disjoint cells, stratified within material-family and experimental-protocol groups; all recorded cells used.'
    return split


def study_datasets(config, splits, include_test=False):
    d = config['data']
    shared = dict(root=d['root'], history_cycles=tuple(d['history_cycles']),
                  segment_points=tuple(d['segment_points']), horizon=d['horizon'],
                  max_anchor_cycle=None, enumerate_anchors=True, cache_size=d.get('cache_size',128))
    datasets = dict(train=TongjiDataset(**shared, cell_ids=splits['train'], seed=config['seed'],
                                       channel_mask_probability=d['channel_mask_probability']),
                    val=TongjiDataset(**shared, cell_ids=splits['val'], seed=d['eval_seed'], deterministic=True))
    if include_test:
        datasets['train_eval'] = TongjiDataset(**shared, cell_ids=splits['train'], seed=d['eval_seed'], deterministic=True)
        datasets['test'] = TongjiDataset(**shared, cell_ids=splits['test'], seed=d['eval_seed'], deterministic=True)
    return datasets


class StudyModule(BatteryTrajectoryModule):
    """Tune using equal weight per validation cell, rather than cell lifespan."""
    def on_validation_epoch_start(self):
        self.cell_errors = defaultdict(lambda:[0.,0])

    def validation_step(self, batch, batch_idx):
        prediction = self(batch)
        if not torch.isfinite(prediction).all():
            raise FloatingPointError('Nonfinite validation trajectory')
        mse, mae = trajectory_errors(prediction, batch['target_soh'], batch['target_valid'])
        self.log('val/mse',mse.mean(),on_epoch=True,batch_size=len(prediction))
        self.log('val/mae',mae.mean(),on_epoch=True,batch_size=len(prediction))
        for cell, error in zip(batch['cell_id'],mse.detach().cpu().tolist()):
            self.cell_errors[cell][0] += error
            self.cell_errors[cell][1] += 1

    def on_validation_epoch_end(self):
        macro = np.mean([total/count for total,count in self.cell_errors.values()])
        self.log('val/cell_mse',float(macro),on_epoch=True,prog_bar=True)


class StudyProgress(L.Callback):
    def __init__(self, run_dir, deadline=None):
        self.run_dir = Path(run_dir)
        self.deadline = datetime.fromisoformat(deadline) if deadline else None
        self.started = None
        self.durations = []

    def on_train_epoch_start(self, trainer, pl_module):
        import time
        self.started = time.perf_counter()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.deadline and datetime.now(timezone.utc) >= self.deadline:
            trainer.fit_loop.min_epochs = 0
            trainer.should_stop = True

    def on_train_epoch_end(self, trainer, pl_module):
        import time
        self.durations.append(time.perf_counter()-self.started)
        row = dict(epochs_completed=trainer.current_epoch+1,epoch_seconds=self.durations,
                   val_cell_mse=float(trainer.callback_metrics['val/cell_mse']),
                   updated_at=datetime.now(timezone.utc).isoformat())
        atomic_json(self.run_dir/'progress.json',row)
        print('PROGRESS',json.dumps({k:v for k,v in row.items() if k!='epoch_seconds'}),flush=True)


@torch.no_grad()
def evaluate_cells(module, loader, limit_batches=None):
    module.eval()
    device = next(module.parameters()).device
    arrays = defaultdict(list)
    for index,batch in enumerate(loader):
        if limit_batches is not None and index >= limit_batches:
            break
        cells = batch['cell_id']
        batch = {k:v.to(device) if torch.is_tensor(v) else v for k,v in batch.items()}
        prediction = module(batch)
        if not torch.isfinite(prediction).all():
            raise FloatingPointError('Nonfinite evaluation prediction')
        mse,mae = trajectory_errors(prediction,batch['target_soh'],batch['target_valid'])
        tensors = dict(anchor=batch['anchor_cycle'],prediction=prediction,target=batch['target_soh'],
                       valid=batch['target_valid'],target_cycle_id=batch['target_cycle_id'],mse=mse,mae=mae)
        for key,value in tensors.items():
            arrays[key].append(value.cpu().numpy())
        arrays['cell_id'].append(np.array(cells))
    arrays = {k:np.concatenate(v) for k,v in arrays.items()}
    per_cell = {}
    for cell in sorted(set(arrays['cell_id'])):
        pick = arrays['cell_id']==cell
        mse = float(arrays['mse'][pick].mean())
        per_cell[cell] = dict(mse=mse,rmse=mse**.5,mae=float(arrays['mae'][pick].mean()),samples=int(pick.sum()))
    metrics = dict(mse=float(arrays['mse'].mean()),mae=float(arrays['mae'].mean()),
                   cell_macro_mse=float(np.mean([x['mse'] for x in per_cell.values()])),
                   cell_macro_rmse=float(np.mean([x['rmse'] for x in per_cell.values()])),
                   cell_macro_mae=float(np.mean([x['mae'] for x in per_cell.values()])),
                   cells=len(per_cell),samples=len(arrays['anchor']),per_cell=per_cell)
    metrics['rmse'] = metrics['mse']**.5
    return metrics,arrays
