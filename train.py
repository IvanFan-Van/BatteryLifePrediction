from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated
from uuid import uuid4

import lightning as L
import torch
import torch.nn.functional as F
import tyro
import wandb
import yaml
from lightning.pytorch.callbacks import (
    DeviceStatsMonitor,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import resnet18


@dataclass
class Args:
    config: str = "config/baseline.yaml"
    resume: Annotated[str | None, tyro.conf.arg(metavar="RUN_ID")] = None


class LitResNet(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)

        self.net = resnet18(weights=None)

        # 将 ImageNet 风格的 stem 适配到 32x32 CIFAR-10
        self.net.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.net.maxpool = nn.Identity()  # type: ignore
        self.net.fc = nn.Linear(self.net.fc.in_features, cfg["model"]["num_classes"])

    def forward(self, x):
        return self.net(x)

    def shared_step(self, batch, stage):
        x, y = batch
        logits = self(x)

        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()

        self.log(
            f"{stage}/loss",
            loss,
            on_step=(stage == "train"),
            on_epoch=True,
            batch_size=len(x),
        )
        self.log(
            f"{stage}/acc",
            acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(x),
        )
        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self.shared_step(batch, "val")

    def configure_optimizers(self):  # type: ignore
        cfg = self.hparams
        opt = cfg["optimizer"]
        optimizer = optim.SGD(
            self.parameters(),
            lr=opt["lr"],
            momentum=opt["momentum"],
            weight_decay=opt["weight_decay"],
        )

        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg["trainer"]["max_epochs"]
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }


def create_dataloaders(cfg):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    train_tf = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    root = cfg["data"]["root"]

    train_full = datasets.CIFAR10(root, train=True, download=True, transform=train_tf)
    val_full = datasets.CIFAR10(root, train=True, download=False, transform=val_tf)

    # 用固定种子生成一次数据划分
    generator = torch.Generator().manual_seed(cfg["data"]["split_seed"])

    indices = torch.randperm(len(train_full), generator=generator).tolist()

    n_val = int(len(train_full) * cfg["data"]["val_ratio"])

    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    # 两套 Dataset 分别使用不重叠的索引
    train_ds = Subset(train_full, train_indices)
    val_ds = Subset(val_full, val_indices)

    loader_args = {
        "batch_size": cfg["data"]["batch_size"],
        "num_workers": cfg["data"]["num_workers"],
    }

    return (
        DataLoader(train_ds, shuffle=True, **loader_args),
        DataLoader(val_ds, shuffle=False, **loader_args),
    )


def main():
    args = tyro.cli(Args)

    if args.resume is None:
        # 新实验：自动生成 ID
        run_id = datetime.now(tz=UTC).strftime("%Y%m%d-%H%M%S-") + uuid4().hex[:6]
        run_dir = Path("runs") / run_id
        run_dir.mkdir(parents=True, exist_ok=False)

        with open(args.config) as f:
            cfg = yaml.safe_load(f)

        with open(run_dir / "config.yaml", "w") as f:
            yaml.safe_dump(cfg, f)

        ckpt_path = None
    else:
        # 恢复实验：使用原来的 ID
        run_id = args.resume
        run_dir = Path("runs") / run_id

        if not run_dir.is_dir():
            raise FileNotFoundError(run_dir)

        with open(run_dir / "config.yaml") as f:
            cfg = yaml.safe_load(f)

        ckpt_path = run_dir / "checkpoints" / "last.ckpt"

        if not ckpt_path.is_file():
            raise FileNotFoundError(ckpt_path)

    L.seed_everything(cfg["seed"], workers=True)
    tb = TensorBoardLogger(str(run_dir), name="tensorboard", version="main")
    wb = WandbLogger(
        project=cfg["experiment"]["project"],
        group=cfg["experiment"]["group"],
        name=cfg["experiment"]["name"],
        save_dir=str(run_dir),
        id=run_id,
        resume="must" if args.resume else "never",
        log_model=False,
        offline=True,  # TODO: offline
    )
    wb.experiment.config.update(cfg)
    checkpoint = ModelCheckpoint(
        dirpath=run_dir / "checkpoints",
        monitor="val/acc",
        mode="max",
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
        save_weights_only=False,
    )

    train_loader, val_loader = create_dataloaders(cfg)
    model = LitResNet(cfg)
    wb.watch(
        model,
        log="gradients",
        log_freq=100,
    )

    tcfg = cfg["trainer"]
    trainer = L.Trainer(
        max_epochs=tcfg["max_epochs"],
        limit_train_batches=tcfg["limit_train_batches"],
        limit_val_batches=tcfg["limit_val_batches"],
        log_every_n_steps=tcfg["log_every_n_steps"],
        num_sanity_val_steps=0,
        accelerator="auto",
        devices=1,
        logger=[tb, wb],
        callbacks=[
            checkpoint,
            # DeviceStatsMonitor(),
            LearningRateMonitor(logging_interval="epoch"),
        ],
        deterministic="warn",
    )

    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=(str(ckpt_path) if ckpt_path is not None else None),
    )

    print("Run:", run_id)
    print("Best Validation Accuracy:", checkpoint.best_model_score.item())  # type: ignore
    print("Best Checkpoint:", checkpoint.best_model_path)
    print("Best checkpoint:", checkpoint.best_model_path)

    wandb.finish()


if __name__ == "__main__":
    main()
