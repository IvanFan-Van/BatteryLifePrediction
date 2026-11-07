"""Lightning wrapper; ordinary PyTorch model lives in hierarchical_ssm.py."""
import lightning as L
import torch

from .hierarchical_ssm import HierarchicalStateSpaceModel


def trajectory_errors(prediction, target, valid):
    """Per-sample normalization prevents longer recorded tails dominating loss."""
    count = valid.sum(dim=-1)
    if not (count > 0).all():
        raise ValueError('Each trajectory needs at least one valid target')
    safe_target = torch.where(valid, target, 0)
    error = prediction - safe_target
    square = torch.where(valid, error.square(), 0).sum(dim=-1) / count
    absolute = torch.where(valid, error.abs(), 0).sum(dim=-1) / count
    return square, absolute


class BatteryTrajectoryModule(L.LightningModule):
    def __init__(self, config, vocabulary_sizes, normalization):
        super().__init__()
        self.save_hyperparameters()
        self.model = HierarchicalStateSpaceModel(vocabulary_sizes, config['model'], normalization)

    def forward(self, batch):
        return self.model(batch)

    def shared_step(self, batch, stage):
        prediction = self(batch)
        if not torch.isfinite(prediction).all():
            raise FloatingPointError('Nonfinite trajectory prediction')
        mse, mae = trajectory_errors(prediction, batch['target_soh'], batch['target_valid'])
        self.log(f'{stage}/mse', mse.mean(), on_step=False, on_epoch=True,
                 batch_size=len(prediction), prog_bar=True)
        self.log(f'{stage}/mae', mae.mean(), on_step=False, on_epoch=True,
                 batch_size=len(prediction))
        return mse.mean()

    def on_train_epoch_start(self):
        self.trainer.train_dataloader.dataset.set_epoch(self.current_epoch)

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        self.shared_step(batch, 'val')

    def test_step(self, batch, batch_idx):
        self.shared_step(batch, 'test')

    def configure_optimizers(self):
        cfg = self.hparams.config['optimizer']
        return torch.optim.AdamW(self.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
