"""One immutable trial; tuning evaluates validation only, finals unlock test."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

os.environ.update(WANDB_MODE='offline',WANDB_SILENT='true',CUBLAS_WORKSPACE_CONFIG=':4096:8')
import lightning as L
import numpy as np
import torch
import wandb
import yaml
from lightning.pytorch.callbacks import EarlyStopping,ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger,TensorBoardLogger,WandbLogger

from src.callbacks import RunDiagnostics, StudyProgress
from src.data.loading import make_loader,vocabulary_sizes
from .evaluation import plot_results
from src.data.study import study_datasets
from src.utils.files import atomic_json
from .study import StudyModule, evaluate_cells


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--run-dir',type=Path,required=True)
    args=parser.parse_args()
    run=args.run_dir.resolve();run.mkdir(parents=True,exist_ok=False)
    config=yaml.safe_load(args.config.read_text(encoding='utf-8'))
    (run/'config.yaml').write_text(yaml.safe_dump(config,sort_keys=False),encoding='utf-8')
    started=time.perf_counter()
    datasets={}
    wb=None
    try:
        import socket
        accelerator=config['trainer']['accelerator']
        visible=config['trainer'].get('gpu_ids','0' if os.name=='nt' else '2')
        if accelerator=='gpu':
            if os.name!='nt' and 'farady' in socket.gethostname().lower() and any(x in ['0','1'] for x in visible.split(',')):
                raise ValueError('farady physical GPU 0/1 remain forbidden')
            os.environ['CUDA_VISIBLE_DEVICES']=visible
        torch.set_num_threads(config['trainer']['cpu_threads'])
        L.seed_everything(config['seed'],workers=True)
        study=Path(config['study_dir'])
        splits=json.loads((study/'splits.json').read_text(encoding='utf-8'))
        normalization=json.loads((study/'normalization.json').read_text(encoding='utf-8'))
        final=config['stage']=='final'
        datasets=study_datasets(config,splits,include_test=final)
        loaders={k:make_loader(v,config,shuffle=(k=='train')) for k,v in datasets.items()}
        module=StudyModule(config,vocabulary_sizes(config['data']['root']),normalization)
        params=sum(p.numel() for p in module.parameters())
        print('TRIAL',json.dumps(dict(stage=config['stage'],variant=config['model']['static_fusion'],seed=config['seed'],parameters=params,
                                     anchors={k:len(v) for k,v in datasets.items()},optimizer=config['optimizer'])),flush=True)
        checkpoint=ModelCheckpoint(dirpath=run/'checkpoints',monitor='val/cell_mse',mode='min',
                                   filename='best-{epoch:03d}',save_top_k=1,save_last=True,auto_insert_metric_name=False)
        callbacks=[checkpoint,RunDiagnostics(run),StudyProgress(run,config.get('deadline_utc'))]
        patience=config['trainer'].get('patience')
        if patience is not None:
            callbacks.append(EarlyStopping(monitor='val/cell_mse',mode='min',patience=patience,check_on_train_epoch_end=False))
        wb=WandbLogger(project='battery-life-prediction',group=study.name,name=run.name,
                       save_dir=str(run),id=run.name,offline=True,log_model=False)
        wb.experiment.config.update(config)
        t=config['trainer']
        trainer=L.Trainer(accelerator=accelerator,devices=1,max_epochs=t['max_epochs'],min_epochs=t.get('min_epochs',1),
                          limit_train_batches=t.get('limit_train_batches',1.0),
                          limit_val_batches=t.get('limit_val_batches',1.0),num_sanity_val_steps=0,
                          gradient_clip_val=t['gradient_clip_val'],deterministic=True,enable_progress_bar=False,
                          log_every_n_steps=50,enable_model_summary=False,
                          logger=[CSVLogger(str(run),name='csv',version='main'),
                                  TensorBoardLogger(str(run),name='tensorboard',version='main'),wb],callbacks=callbacks)
        trainer.fit(module,train_dataloaders=loaders['train'],val_dataloaders=loaders['val'])
        restored=StudyModule.load_from_checkpoint(checkpoint.best_model_path,map_location='cpu',weights_only=False).to(module.device)
        metrics,predictions={},{}
        stages=[('val','val')]
        if final:
            stages=[('train','train_eval'),('val','val'),('test','test')]
        for name,key in stages:
            metrics[name],predictions[name]=evaluate_cells(restored,loaders[key],t.get('evaluation_batches'))
            if final:
                np.savez_compressed(run/f'{name}_predictions.npz',**predictions[name])
        progress=json.loads((run/'progress.json').read_text())
        result=dict(status='completed',stage=config['stage'],variant=config['model']['static_fusion'],
                    seed=config['seed'],optimizer=config['optimizer'],metrics=metrics,
                    objective=float(checkpoint.best_model_score),best_checkpoint=checkpoint.best_model_path,
                    parameters=params,epochs_completed=progress['epochs_completed'],
                    accelerator=accelerator,software=dict(torch=torch.__version__,lightning=L.__version__,cuda=torch.version.cuda),
                    gpu_name=torch.cuda.get_device_name() if accelerator=='gpu' else None,peak_cuda_memory_bytes=torch.cuda.max_memory_allocated() if accelerator=='gpu' else 0,
                    epoch_seconds=progress['epoch_seconds'],elapsed_seconds=time.perf_counter()-started,
                    completed_at=datetime.now(timezone.utc).isoformat(),test_evaluated=final,
                    source_snapshot=str(study/'source'))
        atomic_json(run/'result.json',result)
        if final:
            plot_results(run,predictions)
        for name,values in metrics.items():
            for metric in ['cell_macro_mse','cell_macro_rmse','cell_macro_mae']:
                wb.experiment.summary[f'{name}/{metric}']=values[metric]
        print('RESULT',json.dumps({k:v for k,v in result.items() if k not in ['metrics','epoch_seconds']}),flush=True)
    except BaseException as error:
        atomic_json(run/'failure.json',dict(status='failed',type=type(error).__name__,message=str(error),
                                          elapsed_seconds=time.perf_counter()-started))
        raise
    finally:
        for dataset in datasets.values():dataset.close()
        if wb is not None:wandb.finish()


if __name__=='__main__':main()
