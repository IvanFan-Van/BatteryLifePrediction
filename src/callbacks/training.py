"""Lightning callbacks for gradient checks and study progress."""
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import lightning as L
import torch
from lightning.pytorch.callbacks import Callback

from src.utils.files import atomic_json


class RunDiagnostics(Callback):
    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)
        self.recorded = False

    def on_after_backward(self, trainer, pl_module):
        norms = dict(encoder=0.0, dynamics=0.0, head=0.0)
        for name, parameter in pl_module.model.named_parameters():
            if parameter.grad is None:
                continue
            if not torch.isfinite(parameter.grad).all():
                raise FloatingPointError(f"Nonfinite gradient: {name}")
            group = "encoder"
            if name.startswith("dynamics."):
                group = "dynamics"
            elif name.startswith("head."):
                group = "head"
            norms[group] += float(parameter.grad.detach().square().sum())
        if not self.recorded:
            norms = {key: value**0.5 for key, value in norms.items()}
            if not all(value > 0 for value in norms.values()):
                raise RuntimeError(f"A module has no gradient: {norms}")
            (self.run_dir / "gradient_check.json").write_text(
                json.dumps(norms, indent=2)
            )
            print("Finite nonzero gradient norms:", norms, flush=True)
            self.recorded = True

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        values = {
            key: float(metrics[key])
            for key in ["train/mse", "val/mse"]
            if key in metrics
        }
        print(f"Epoch {trainer.current_epoch + 1}: {values}", flush=True)


class StudyProgress(L.Callback):
    def __init__(self, run_dir, deadline=None):
        self.run_dir = Path(run_dir)
        self.deadline = datetime.fromisoformat(deadline) if deadline else None
        self.started = None
        self.durations = []

    def on_train_epoch_start(self, trainer, pl_module):
        self.started = time.perf_counter()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.deadline and datetime.now(timezone.utc) >= self.deadline:
            trainer.fit_loop.min_epochs = 0
            trainer.should_stop = True

    def on_train_epoch_end(self, trainer, pl_module):
        self.durations.append(time.perf_counter()-self.started)
        row = dict(epochs_completed=trainer.current_epoch+1,epoch_seconds=self.durations,
                   val_cell_mse=float(trainer.callback_metrics['val/cell_mse']),
                   updated_at=datetime.now(timezone.utc).isoformat())
        atomic_json(self.run_dir/'progress.json',row)
        print('PROGRESS',json.dumps({k:v for k,v in row.items() if k!='epoch_seconds'}),flush=True)
