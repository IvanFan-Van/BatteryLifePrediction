"""Train the Tongji state-space model with offline experiment logs.

uv run --locked python -m src.core.train --config configs/basic.yaml
"""

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import lightning as L
import numpy as np
import torch
import tyro
import wandb
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger, WandbLogger

from src.callbacks import RunDiagnostics
from src.data import TongjiTaskSampler
from src.data.loading import build_datasets, make_loader
from src.utils.files import read_json, read_yaml, write_json
from src.utils.paths import CONFIGS_DIR, OUTPUTS_DIR

from .evaluation import evaluate, plot_results
from .module import BatteryTrajectoryModule


@dataclass
class Args:
    config: Path = CONFIGS_DIR / "basic.yaml"
    """配置文件路径"""
    resume: Path | None = None
    """恢复训练的目录路径"""
    resume_config: Path | None = None
    """恢复训练时使用的配置快照路径"""
    run_dir: Path | None = None
    """排队任务的输出目录"""


def prepare_run(args: Args):
    if args.resume and args.run_dir:
        raise ValueError("Use either --resume or --run-dir")
    if args.resume_config and not args.resume:
        raise ValueError("--resume-config requires --resume")
    if args.resume:
        run_dir = Path(args.resume).expanduser().resolve()
        config = read_yaml(args.resume_config or run_dir / "config.yaml")
        ckpt_path = str(run_dir / "checkpoints/last.ckpt")
        if not Path(ckpt_path).is_file():
            raise FileNotFoundError(ckpt_path)
    else:
        config = read_yaml(args.config)
        run_id = (
            datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f-")
            + config["experiment"]["name"]
        )
        run_dir = args.run_dir.resolve() if args.run_dir else OUTPUTS_DIR / run_id
        if run_dir.exists():
            if (
                not (run_dir / "queued.json").is_file()
                or (run_dir / "checkpoints").exists()
            ):
                raise FileExistsError("Use --resume for an existing training run")
        else:
            run_dir.mkdir(parents=True)
        # 复制 config 配置到实验目录
        (run_dir / "config.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )
        ckpt_path = None
    return config, run_dir, ckpt_path


def main():
    args = tyro.cli(Args)
    config, run_dir, ckpt_path = prepare_run(args)

    torch.set_num_threads(config["trainer"]["cpu_threads"])
    L.seed_everything(config["seed"], workers=True)

    # 获取数据集, datasets 包含 train, train_eval, val, test 四个数据集. splits 包含如何划分的信息
    saved_splits = None
    if args.resume:
        saved_splits = read_json(run_dir / "splits.json")
    datasets, splits = build_datasets(config, saved_splits)
    write_json(run_dir / "splits.json", splits)
    data = config["data"]
    max_epochs = datasets["train"].max_epochs

    model = BatteryTrajectoryModule(config)
    parameter_count = sum(p.numel() for p in model.parameters())
    print("Run directory:", run_dir, flush=True)
    print("Parameters:", parameter_count, flush=True)
    print(
        "Forecast tasks:",
        {name: len(ds.tasks) for name, ds in datasets.items()},
        flush=True,
    )
    print(
        "Samples per epoch:",
        {name: len(ds) for name, ds in datasets.items()},
        flush=True,
    )
    checkpoint = ModelCheckpoint(
        dirpath=run_dir / "checkpoints",
        monitor="val/mse",
        mode="min",
        filename="best-{epoch:03d}",
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=True,
    )
    wb = WandbLogger(
        project=config["experiment"]["project"],
        group=config["experiment"]["group"],
        name=config["experiment"]["name"],
        save_dir=str(run_dir),
        id=run_dir.name,
        offline=True,
        log_model=False,
    )
    wb.experiment.config.update(config)
    tb = TensorBoardLogger(str(run_dir), name="tensorboard", version="main")
    csv = CSVLogger(str(run_dir), name="csv", version="main")
    trainer = L.Trainer(
        accelerator=config["trainer"]["accelerator"],
        devices=1,
        max_epochs=max_epochs,
        limit_train_batches=config["trainer"]["limit_train_batches"],
        limit_val_batches=config["trainer"]["limit_val_batches"],
        limit_test_batches=config["trainer"]["limit_test_batches"],
        gradient_clip_val=config["trainer"]["gradient_clip_val"],
        log_every_n_steps=1,
        num_sanity_val_steps=1,
        deterministic=True,
        enable_progress_bar=False,
        logger=[
            csv,
            tb,
            wb,
        ],
        callbacks=[checkpoint, RunDiagnostics(run_dir)],
    )
    loaders = {
        k: make_loader(v, config, shuffle=(k == "train")) for k, v in datasets.items()
    }
    try:
        if ckpt_path is not None:
            train_sampler = loaders["train"].sampler
            assert isinstance(train_sampler, TongjiTaskSampler)
            previous = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            # Lightning can prefetch before calling set_epoch on the resumed sampler.
            next_epoch = int(previous["epoch"]) + 1
            train_sampler.set_epoch(min(next_epoch, max_epochs - 1))
            del previous
        trainer.fit(
            model,
            train_dataloaders=loaders["train"],
            val_dataloaders=loaders["val"],
            ckpt_path=ckpt_path,
        )
        best_name = Path(checkpoint.best_model_path).name
        best_score = checkpoint.best_model_score
        if ckpt_path is not None and best_score is None:
            # Lightning skips restoring the best score when the checkpoint directory moved.
            previous = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            best = previous["callbacks"][checkpoint.state_key]
            best_name = Path(best["best_model_path"]).name
            best_score = best["best_model_score"]
        if best_score is None:
            raise RuntimeError("No validation checkpoint score was recorded")
        best_path = run_dir / "checkpoints" / best_name
        test_metrics = trainer.test(
            model, dataloaders=loaders["test"], ckpt_path=str(best_path)
        )[0]
        # trainer.test loads the best validation checkpoint into this model.
        metrics, predictions = {}, {}
        limit = config["trainer"].get("evaluation_batches")
        for stage, key in [("train", "train_eval"), ("val", "val"), ("test", "test")]:
            metrics[stage], predictions[stage] = evaluate(model, loaders[key], limit)
            npz_path = run_dir / f"{stage}_predictions.npz"
            np.savez_compressed(npz_path, **predictions[stage])
        results = {
            "metrics": metrics,
            "lightning_test": test_metrics,
            "best_checkpoint": str(best_path),
            "best_val_mse": float(best_score),
            "parameter_count": parameter_count,
            "evaluation_view_seed": data["eval_seed"],
            "validation_note": splits["note"],
            "device": config["trainer"]["accelerator"],
        }
        write_json(run_dir / "metrics.json", results)
        plot_results(run_dir, predictions)
        for stage, value in metrics.items():
            for metric in ["mse", "rmse", "mae"]:
                wb.experiment.summary[f"best/{stage}/{metric}"] = value[metric]
        wb.experiment.summary["parameter_count"] = parameter_count
        print(json.dumps(results, indent=2), flush=True)
    finally:
        for dataset in datasets.values():
            dataset.close()
        wandb.finish()


if __name__ == "__main__":
    main()
