"""Lightning callbacks for curriculum training diagnostics."""

import json
from pathlib import Path

import torch
from lightning.pytorch.callbacks import Callback


class RunDiagnostics(Callback):
    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)
        self.recorded = False

    def on_after_backward(self, trainer, pl_module):
        norms = {"encoder": 0.0, "dynamics": 0.0, "head": 0.0}
        for name, parameter in pl_module.model.named_parameters():  # type: ignore
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
