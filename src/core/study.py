"""Cell-weighted validation and fixed-view evaluation."""
from collections import defaultdict

import numpy as np
import torch

from .module import BatteryTrajectoryModule, trajectory_errors


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
