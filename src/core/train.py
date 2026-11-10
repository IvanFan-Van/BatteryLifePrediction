"""Train the Tongji state-space model with offline experiment logs.

uv run --locked python -m src.core.train --config configs/basic.yaml
"""

import argparse
import json
import os
import socket
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import lightning as L
import numpy as np
import torch
import wandb
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger, WandbLogger

from src.callbacks import RunDiagnostics
from src.data import TongjiTaskSampler
from src.data.loading import build_datasets, make_loader
from src.utils.files import read_json, read_yaml, write_json
from src.utils.paths import (
    CONFIGS_DIR,
    OUTPUTS_DIR,
    PROJECT_ROOT,
    processed_data_path,
    run_path,
)
from src.utils.snapshot import copy_source, source_manifest

from .evaluation import evaluate, plot_results
from .module import BatteryTrajectoryModule


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=CONFIGS_DIR / "basic.yaml"
    )
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument(
        "--resume", type=Path, help="Existing run directory with checkpoints/last.ckpt"
    )
    args = parser.parse_args()

    # 根据是否 resume 配置 config, run_dir, ckpt_path
    if args.resume:
        run_dir = run_path(args.resume)
        config = read_yaml(run_dir / "config.yaml")
        ckpt_path = str(run_dir / "checkpoints/last.ckpt")
        if not Path(ckpt_path).is_file():
            raise FileNotFoundError(ckpt_path)
    else:
        run_id = datetime.now(UTC).strftime("%Y%m%d-%H%M%S-") + uuid4().hex[:6]
        run_dir = args.run_dir.resolve() if args.run_dir else (OUTPUTS_DIR / run_id)
        config = read_yaml(args.config)
        config["data"]["root"] = str(processed_data_path(config["data"]["root"]))
        if run_dir.exists():
            if not (run_dir / "queued.json").is_file():
                raise FileExistsError(run_dir)
            if (run_dir / "checkpoints").exists():
                raise FileExistsError("Use --resume for an existing training run")
        else:
            run_dir.mkdir(parents=True)
        (run_dir / "config.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )
        ckpt_path = None
    config["data"]["root"] = str(processed_data_path(config["data"]["root"]))
    os.environ["WANDB_MODE"] = "offline"
    os.environ["WANDB_SILENT"] = "true"
    if config["trainer"]["accelerator"] == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    elif (
        os.name != "nt"
        and "farady" in socket.gethostname().lower()
        and os.environ.get("CUDA_VISIBLE_DEVICES") not in ["2", "3", "2,3", "3,2"]
    ):
        raise ValueError(
            "farady GPU runs require explicit CUDA_VISIBLE_DEVICES selecting physical GPU 2/3"
        )
    torch.set_num_threads(config["trainer"]["cpu_threads"])
    L.seed_everything(config["seed"], workers=True)
    saved_splits = None
    if args.resume:
        saved_splits = read_json(run_dir / "splits.json")
    datasets, splits = build_datasets(config, saved_splits)
    write_json(run_dir / "splits.json", splits)
    max_epochs = datasets["train"].max_epochs
    module = BatteryTrajectoryModule(config)
    if not args.resume:
        try:
            git_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=PROJECT_ROOT,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            git_commit = "source-snapshot"
        sources = source_manifest(PROJECT_ROOT)
        write_json(
            run_dir / "provenance.json",
            {
                "git_commit": git_commit,
                "source_sha256": sources,
                "torch_version": torch.__version__,
                "wandb_mode": "offline",
            },
        )
        if not (run_dir / "source").exists():
            copy_source(PROJECT_ROOT, run_dir / "source", sources)
    parameter_count = sum(p.numel() for p in module.parameters())
    print("Run directory:", run_dir, flush=True)
    print("Parameters:", parameter_count, flush=True)
    print("Forecast tasks:", {name: len(ds.tasks) for name, ds in datasets.items()}, flush=True)
    print("Samples per epoch:", {name: len(ds) for name, ds in datasets.items()}, flush=True)
    checkpoint = ModelCheckpoint(
        dirpath=run_dir / "checkpoints",
        monitor="val/mse",
        mode="min",
        filename="best-{epoch:03d}",
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
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
            CSVLogger(str(run_dir), name="csv", version="main"),
            TensorBoardLogger(str(run_dir), name="tensorboard", version="main"),
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
            module,
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
            module, dataloaders=loaders["test"], ckpt_path=str(best_path)
        )[0]
        # trainer.test loads the best validation checkpoint into this module.
        metrics, predictions = {}, {}
        limit = config["trainer"].get("evaluation_batches")
        for stage, key in [("train", "train_eval"), ("val", "val"), ("test", "test")]:
            metrics[stage], predictions[stage] = evaluate(module, loaders[key], limit)
            npz_path = run_dir / f"{stage}_predictions.npz"
            np.savez_compressed(npz_path, **predictions[stage])
        results = {
            "metrics": metrics,
            "lightning_test": test_metrics,
            "best_checkpoint": str(best_path),
            "best_val_mse": float(best_score),
            "parameter_count": parameter_count,
            "evaluation_view_seed": config["data"]["eval_seed"],
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
