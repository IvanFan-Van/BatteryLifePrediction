"""Lightning training and masked errors for Tongji SOH trajectories."""

import lightning as L
import torch

from .trajectory import BatteryTrajectoryModel


def trajectory_errors(prediction, target, valid):
    """Compute per-sample MSE and MAE over available future labels."""
    count = valid.sum(dim=-1)
    if not (count > 0).all():
        raise ValueError("Each trajectory needs at least one valid target")
    error = torch.where(valid, prediction - target, 0)
    return error.square().sum(dim=-1) / count, error.abs().sum(dim=-1) / count


class BatteryTrajectoryModule(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.model = BatteryTrajectoryModel(config)

    def forward(self, batch):
        return self.model(batch)

    def shared_step(self, batch, stage):
        prediction = self(batch)
        if not torch.isfinite(prediction).all():
            raise FloatingPointError("Nonfinite trajectory prediction")
        mse, mae = trajectory_errors(
            prediction, batch["target_soh"], batch["target_soh_valid"]
        )
        loss = mse.mean()
        self.log(
            f"{stage}/mse",
            loss,
            batch_size=len(prediction),
            on_epoch=True,
            on_step=False,
        )
        self.log(
            f"{stage}/mae",
            mae.mean(),
            batch_size=len(prediction),
            on_epoch=True,
            on_step=False,
        )
        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")

    def configure_optimizers(self):
        config = self.hparams["config"]["optimizer"]
        return torch.optim.AdamW(
            self.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
        )
