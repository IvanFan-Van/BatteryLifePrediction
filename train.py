"""Train the Tongji hierarchical state-space demo with offline experiment logs.

uv run python train.py --config configs/tongji_pilot.yaml
uv run python train.py --config configs/tongji_smoke.yaml
"""

import argparse
import hashlib
import json
import os
import shutil
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import lightning as L
import numpy as np
import torch
import wandb
import yaml
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger, WandbLogger

from training.data import (
    build_datasets,
    fit_normalization,
    make_loader,
    vocabulary_sizes,
)
from training.evaluation import evaluate, plot_results
from training.module import BatteryTrajectoryModule


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


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def source_manifest(root):
    files = [Path("train.py")]
    for folder in ["models", "training", "battery_data"]:
        files.extend(p.relative_to(root) for p in (root / folder).glob("*.py"))
    return {
        str(path): hashlib.sha256((root / path).read_bytes()).hexdigest()
        for path in files
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/tongji_smoke.yaml")
    )
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument(
        "--resume", type=Path, help="Existing run directory with checkpoints/last.ckpt"
    )
    args = parser.parse_args()
    project = Path(__file__).resolve().parent
    if args.resume:
        run_dir = args.resume.resolve()
        config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
        ckpt_path = str(run_dir / "checkpoints/last.ckpt")
        if not Path(ckpt_path).is_file():
            raise FileNotFoundError(ckpt_path)
    else:
        config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-") + uuid4().hex[:6]
        run_dir = (
            args.run_dir.resolve() if args.run_dir else (project / "runs" / run_id)
        )
        if run_dir.exists():
            if not (run_dir / "queued.json").is_file():
                raise FileExistsError(run_dir)
            if (run_dir / "checkpoints").exists():
                raise FileExistsError("Use --resume for an existing training run")
        else:
            run_dir.mkdir(parents=True)
        config["data"]["root"] = str(Path(config["data"]["root"]).resolve())
        (run_dir / "config.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )
        ckpt_path = None
    os.environ["WANDB_MODE"] = "offline"
    os.environ["WANDB_SILENT"] = "true"
    if config["trainer"]["accelerator"] == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    elif os.name != "nt" and "farady" in socket.gethostname().lower() and os.environ.get("CUDA_VISIBLE_DEVICES") not in ["2", "3", "2,3", "3,2"]:
        raise ValueError(
            "farady GPU runs require explicit CUDA_VISIBLE_DEVICES selecting physical GPU 2/3"
        )
    torch.set_num_threads(config["trainer"]["cpu_threads"])
    L.seed_everything(config["seed"], workers=True)
    saved_splits = None
    if args.resume:
        saved_splits = json.loads((run_dir / "splits.json").read_text())
    datasets, splits = build_datasets(config, saved_splits)
    write_json(run_dir / "splits.json", splits)
    if args.resume:
        normalization = json.loads((run_dir / "normalization.json").read_text())
    else:
        normalization = fit_normalization(
            config["data"]["root"], [config["data"]["train_cell"]]
        )
        write_json(run_dir / "normalization.json", normalization)
    sizes = vocabulary_sizes(config["data"]["root"])
    module = BatteryTrajectoryModule(config, sizes, normalization)
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=project, text=True
        ).strip()
    except subprocess.CalledProcessError:
        git_commit = "source-snapshot"
    sources = source_manifest(project)
    write_json(
        run_dir / "provenance.json",
        dict(
            git_commit=git_commit,
            source_sha256=sources,
            torch_version=torch.__version__,
            wandb_mode="offline",
        ),
    )
    if not args.resume and not (run_dir / "source").exists():
        for name in sources:
            destination = run_dir / "source" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(project / name, destination)
    parameter_count = sum(p.numel() for p in module.parameters())
    print("Run directory:", run_dir, flush=True)
    print("Parameters:", parameter_count, flush=True)
    print("Anchor samples:", {k: len(v) for k, v in datasets.items()}, flush=True)
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
        max_epochs=config["trainer"]["max_epochs"],
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
        trainer.fit(
            module,
            train_dataloaders=loaders["train"],
            val_dataloaders=loaders["val"],
            ckpt_path=ckpt_path,
        )
        test_metrics = trainer.test(
            module, dataloaders=loaders["test"], ckpt_path="best"
        )[0]
        # trainer.test loads the best validation checkpoint into this module.
        metrics, predictions = {}, {}
        limit = config["trainer"].get("evaluation_batches")
        for stage, key in [("train", "train_eval"), ("val", "val"), ("test", "test")]:
            metrics[stage], predictions[stage] = evaluate(module, loaders[key], limit)
            npz_path = run_dir / f"{stage}_predictions.npz"
            np.savez_compressed(npz_path, **predictions[stage])
        results = dict(
            metrics=metrics,
            lightning_test=test_metrics,
            best_checkpoint=checkpoint.best_model_path,
            best_val_mse=float(checkpoint.best_model_score),
            parameter_count=parameter_count,
            evaluation_view_seed=config["data"]["eval_seed"],
            validation_note=splits["note"],
            device=config["trainer"]["accelerator"],
        )
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
