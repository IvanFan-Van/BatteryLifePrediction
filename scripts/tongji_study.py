"""Prepare, queue, supervise, and summarize a frozen overnight Tongji study."""
import argparse
from collections import Counter
from datetime import datetime,timedelta,timezone
import hashlib
import itertools
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from training.data import fit_normalization
from training.study import atomic_json,clean_environment,stratified_cell_split


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def client(study,*arguments):
    return subprocess.check_output(['pueue','--config',str(study/'pueue.yml'),*map(str,arguments)],
                                   text=True,env=clean_environment(),stderr=subprocess.STDOUT)


def write_report(study,state):
    lines=['# Tongji static-fusion comparison','',f"Status: **{state['status']}**; updated {datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')}.",'',
           'GPU experiments; two parallel experiment slots; W&B offline. All trials use the same disjoint-cell split and training-only normalization.',
           'Current model: static tokens in cycle-level attention. Early model: the SAME static tokens also enter within-cycle attention; cycle-level fusion is retained. Matched configurations have equal parameter counts.','']
    splits=read_json(study/'splits.json')
    lines += [f"Cells: train {len(splits['train'])}, validation {len(splits['val'])}, test {len(splits['test'])}; {len(splits['groups'])} stratification groups. All 130 cells are assigned exactly once.",
              'Forecast horizon: 32 future cycles. All eligible anchors across complete cell histories are used. This is within-material/protocol cell generalization, not a held-out-chemistry or full-EOL forecast.',
              'Selection objective: mean per-cell validation MSE. No tuning trial evaluates test. Only final selected configurations evaluate test.','',
              '| Trial | Stage | Model | Seed | LR | Weight decay | Status | Epochs | Best val cell MSE |',
              '| --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: |']
    finals=[]
    for job in state['jobs']:
        result_path=study/'trials'/job['name']/'result.json'
        result=read_json(result_path) if result_path.exists() else None
        progress_path=study/'trials'/job['name']/'progress.json'
        progress=read_json(progress_path) if progress_path.exists() else {}
        epoch=result['epochs_completed'] if result else progress.get('epochs_completed','')
        objective=f"{result['objective']:.7g}" if result else ''
        lines.append(f"| [{job['name']}](trials/{job['name']}/config.yaml) | {job['stage']} | {job['variant']} | {job['seed']} | {job['optimizer']['lr']} | {job['optimizer']['weight_decay']} | {job.get('status','queued')} | {epoch} | {objective} |")
        if result and result['stage']=='final':finals.append(result)
    lines += ['','## Final evaluation','', 'Errors below are SOH percentage points. Cell macro RMSE gives equal weight to each cell; anchor RMSE gives equal weight to each anchor.','']
    if finals:
        lines += ['| Model | Seed | Train cell RMSE | Val cell RMSE | Test cell RMSE | Test anchor RMSE |',
                  '| --- | ---: | ---: | ---: | ---: | ---: |']
        for result in finals:
            m=result['metrics'];fmt=lambda x:f'{100*x:.4f}'
            lines.append(f"| {result['variant']} | {result['seed']} | {fmt(m['train']['cell_macro_rmse'])} | {fmt(m['val']['cell_macro_rmse'])} | {fmt(m['test']['cell_macro_rmse'])} | {fmt(m['test']['rmse'])} |")
        for variant in ['cycle','early']:
            results=[x for x in finals if x['variant']==variant]
            if results:
                scores=np.array([x['metrics']['test']['cell_macro_rmse']*100 for x in results])
                std=float(scores.std(ddof=1)) if len(scores)>1 else 0.
                lines += ['',f"**{variant}** test cell-macro RMSE: {scores.mean():.4f} ± {std:.4f} percentage points ({len(scores)} seeds; sample standard deviation)."]
        if (study/'comparison.png').exists():
            lines += ['','[Comparison figure](comparison.png) · [Machine-readable summary](summary.json)']
    else:lines.append('Final runs are not complete yet. This report updates automatically.')
    if (study/'selected.json').exists():
        lines += ['','## Selected hyperparameters','', '```json',json.dumps(read_json(study/'selected.json'),indent=2),'```']
    lines += ['','## Protocol and limitations','',
              'Identical search spaces and budgets per model: coarse grid, top-candidate refinement, then paired final seeds. A finite search identifies the best observed validation configuration, not a global optimum.',
              'Data source and code hashes are in provenance.json. Every running task uses source/ and an immutable per-trial YAML. Predictions, checkpoints, per-cell/family metrics and offline W&B logs are under trials/.',
              'Final training can stop early by validation patience or the morning time budget; completed epochs are reported. No results are silently substituted for failed tasks.',
              'Primary statistics are descriptive across three paired seeds and one fixed cell split. Generalization across experimental protocols and materials requires separate studies.']
    if state.get('error'):lines += ['',f"Controller error: {state['error']}"]
    temp=study/'report.md.tmp';temp.write_text('\n'.join(lines)+'\n',encoding='utf-8');temp.replace(study/'report.md')


def aggregate(study,state):
    results=[]
    for job in state['jobs']:
        p=study/'trials'/job['name']/'result.json'
        if p.exists():results.append(read_json(p))
    summary=dict(status=state['status'],final_results=[x for x in results if x['stage']=='final'],
                 tuning_results=[{k:v for k,v in x.items() if k not in ['metrics','epoch_seconds']} for x in results if x['stage']!='final'])
    atomic_json(study/'summary.json',summary)
    finals=summary['final_results']
    if not finals:return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    for index,variant in enumerate(['cycle','early']):
        chosen=[x for x in finals if x['variant']==variant]
        scores=[x['metrics']['test']['cell_macro_rmse']*100 for x in chosen]
        if not scores:continue
        axes[0].bar(index,np.mean(scores),alpha=.5)
        axes[0].scatter(np.full(len(scores),index),scores,color='black',s=18)
        family={}
        for name in ['Tongji1','Tongji2','Tongji3']:
            errors=[cell['rmse']*100 for result in chosen for cid,cell in result['metrics']['test']['per_cell'].items() if cid.startswith(name+'_')]
            family[name]=np.mean(errors)
        axes[1].plot(list(family),list(family.values()),'o-',label=variant)
    axes[0].set_xticks([0,1],['Current (cycle)','Early + cycle'])
    axes[0].set(ylabel='Test cell-macro RMSE (percentage points)',title='Final seeds (dots), mean (bar)')
    axes[1].set(ylabel='Test cell-macro RMSE (percentage points)',title='By material family; average over seeds')
    axes[1].legend();fig.savefig(study/'comparison.png',dpi=150);plt.close(fig)


def queue_trial(study,state,config,name):
    existing=next((j for j in state['jobs'] if j['name']==name),None)
    if existing:return existing
    config_path=study/'configs'/f'{name}.yaml'
    with config_path.open('x',encoding='utf-8') as stream:yaml.safe_dump(config,stream,sort_keys=False)
    command=subprocess.list2cmdline([state['python'],str(study/'source/scripts/tongji_trial.py'),
                                    '--config',str(config_path),'--run-dir',str(study/'trials'/name)])
    task_id=client(study,'add','--print-task-id','--group','experiments','--working-directory',study/'source',
                   '--label',name,'--',command).strip()
    job=dict(name=name,task_id=task_id,stage=config['stage'],variant=config['model']['static_fusion'],
             optimizer=config['optimizer'],seed=config['seed'],status='queued')
    state['jobs'].append(job);atomic_json(study/'state.json',state)
    return job


def await_jobs(study,state,jobs):
    while True:
        # Never print raw status: Pueue status contains task environment variables.
        tasks=json.loads(client(study,'status','--json'))['tasks']
        for job in state['jobs']:
            task=tasks.get(job['task_id'],{})
            status=task.get('status',{})
            job['status']=next(iter(status)) if isinstance(status,dict) else str(status)
            result=study/'trials'/job['name']/'result.json'
            if result.exists():job['status']='completed'
            elif (study/'trials'/job['name']/'failure.json').exists():job['status']='failed'
            elif isinstance(status,dict) and 'Done' in status:job['status']='failed'
        atomic_json(study/'state.json',state);write_report(study,state)
        if all(job['status'] in ['completed','failed'] for job in jobs):break
        time.sleep(state['settings'].get('poll_seconds',15))


def make_config(study,state,variant,optimizer,stage,seed,epochs):
    config=json.loads(json.dumps(state['base_config']))
    config.update(stage=stage,seed=seed,study_dir=str(study),deadline_utc=state['deadline_utc'])
    config['model']['static_fusion']=variant
    config['optimizer']=optimizer.copy()
    config['trainer']['max_epochs']=epochs
    config['trainer']['min_epochs']=min(8,epochs) if stage=='final' else 1
    config['trainer']['patience']=state['settings']['patience'] if stage=='final' else None
    return config


def control(study):
    state=read_json(study/'state.json');settings=state['settings']
    awake=False
    if os.name=='nt':
        import ctypes
        awake=bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001))
    try:
        state['status']='hyperparameter coarse search'
        jobs=[]
        choices=list(itertools.product(settings['learning_rates'],settings['weight_decays']))
        for index,(lr,wd) in enumerate(choices):
            for variant in ['cycle','early']:
                cfg=make_config(study,state,variant,dict(lr=lr,weight_decay=wd),'coarse',settings['search_seed'],settings['coarse_epochs'])
                jobs.append(queue_trial(study,state,cfg,f'coarse-{variant}-{index:02d}'))
        await_jobs(study,state,jobs)
        candidates={}
        for variant in ['cycle','early']:
            successful=[read_json(study/'trials'/j['name']/'result.json') for j in jobs if j['variant']==variant and j['status']=='completed']
            if not successful:raise RuntimeError(f'All coarse trials failed for {variant}')
            candidates[variant]=sorted(successful,key=lambda r:r['objective'])
        state['status']='hyperparameter refinement'
        refined=[]
        for index in range(settings['refine_top_k']):
            for variant in ['cycle','early']:
                if index>=len(candidates[variant]):continue
                cfg=make_config(study,state,variant,candidates[variant][index]['optimizer'],'refine',settings['search_seed'],settings['refine_epochs'])
                refined.append(queue_trial(study,state,cfg,f'refine-{variant}-{index:02d}'))
        if refined:await_jobs(study,state,refined)
        selected={}
        for variant in ['cycle','early']:
            pool=[read_json(study/'trials'/j['name']/'result.json') for j in refined if j['variant']==variant and j['status']=='completed']
            best=min(pool or candidates[variant],key=lambda r:r['objective'])
            selected[variant]=dict(optimizer=best['optimizer'],validation_cell_mse=best['objective'],
                                   selection_stage=best['stage'],parameters=best['parameters'])
        atomic_json(study/'selected.json',selected)
        durations=[d for j in state['jobs'] if j['status']=='completed' for d in read_json(study/'trials'/j['name']/'result.json')['epoch_seconds']]
        seconds=max(float(np.percentile(durations,75)),1.)
        remaining=(datetime.fromisoformat(state['deadline_utc'])-datetime.now(timezone.utc)).total_seconds()
        # Reserve 30% for loading/evaluation and use one equal final budget for both variants.
        capacity=int(max(remaining,0)*settings['parallel']*.7/(2*len(settings['final_seeds'])*seconds))
        final_epochs=max(1,min(settings['final_epochs'],capacity))
        state['final_epoch_budget']=final_epochs
        state['status']='final paired-seed training and held-out test'
        finals=[]
        for seed in settings['final_seeds']:
            for variant in ['cycle','early']:
                cfg=make_config(study,state,variant,selected[variant]['optimizer'],'final',seed,final_epochs)
                finals.append(queue_trial(study,state,cfg,f'final-{variant}-seed{seed}'))
        await_jobs(study,state,finals)
        state['status']='complete' if all(j['status']=='completed' for j in finals) else 'complete with failed final runs'
    except BaseException as error:
        state.update(status='controller failed',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        atomic_json(study/'state.json',state);aggregate(study,state);write_report(study,state)
        if awake:
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


def prepare(config_path):
    settings=yaml.safe_load(Path(config_path).read_text(encoding='utf-8'))
    base=yaml.safe_load((ROOT/settings['base_config']).read_text(encoding='utf-8'))
    base['data'].update(settings.get('data_overrides',{}))
    base['data']['root']=str((ROOT/base['data']['root']).resolve())
    base['trainer'].update(settings.get('trainer_overrides',{}))
    base['trainer']['cpu_threads']=settings['cpu_threads']
    base['trainer']['accelerator']=settings.get('accelerator','gpu')
    base['trainer']['gpu_ids']=settings.get('gpu_ids','0')
    deadline=datetime.fromisoformat(settings['deadline_local']).astimezone(timezone.utc)
    if deadline<=datetime.now(timezone.utc):raise ValueError('Morning deadline has already passed')
    stamp=datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d-%H%M%S')
    study=ROOT/'runs'/f"{settings.get('name','tongji-ablation')}-{stamp}"
    study.mkdir(parents=True,exist_ok=False)
    for folder in ['configs','trials','source']:(study/folder).mkdir()
    splits=stratified_cell_split(base['data']['root'],settings['split_seed'])
    atomic_json(study/'splits.json',splits)
    atomic_json(study/'normalization.json',fit_normalization(base['data']['root'],splits['train']))
    hashes={}
    files=[ROOT/'train.py',ROOT/'pyproject.toml',ROOT/'uv.lock']
    for folder in ['models','training','battery_data','scripts']:
        files.extend((ROOT/folder).glob('*.py'))
    for path in files:
        relative=path.relative_to(ROOT);destination=study/'source'/relative
        destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,destination)
        hashes[str(relative)]=hashlib.sha256(path.read_bytes()).hexdigest()
    source_metadata={cid:read_json(Path(base['data']['root'])/'cells'/cid/'metadata.json')['source_sha256'] for name in ['train','val','test'] for cid in splits[name]}
    atomic_json(study/'provenance.json',dict(source_sha256=hashes,data_source_sha256=source_metadata,
                                           config_source=str(Path(config_path).resolve())))
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    storage=study/'pueue-state'
    pueue=dict(shared=dict(pueue_directory=str(storage),runtime_directory=str(storage),host='127.0.0.1',port=str(port),
                          use_unix_socket=False,daemon_cert=str(storage/'certs/daemon.cert'),
                          daemon_key=str(storage/'certs/daemon.key'),shared_secret_path=str(storage/'shared_secret')),
               daemon=dict(shell_command=[os.environ.get('COMSPEC',r'C:\Windows\System32\cmd.exe'),'/C','{{ pueue_command_string }}']))
    (study/'pueue.yml').write_text(yaml.safe_dump(pueue),encoding='utf-8')
    state=dict(status='prepared',jobs=[],settings=settings,base_config=base,python=sys.executable,
               deadline_utc=deadline.isoformat())
    atomic_json(study/'state.json',state);write_report(study,state)
    return study


def launch(config_path):
    study=prepare(config_path)
    flags=subprocess.CREATE_NO_WINDOW|subprocess.DETACHED_PROCESS if os.name=='nt' else 0
    with (study/'daemon.log').open('ab') as output:
        process=subprocess.Popen(['pueued','--config',str(study/'pueue.yml')],stdout=output,stderr=output,
                                 env=clean_environment(),creationflags=flags,start_new_session=os.name!='nt')
    for _ in range(30):
        try:client(study,'group');break
        except subprocess.CalledProcessError:time.sleep(1)
    else:raise RuntimeError('Pueue daemon did not start')
    client(study,'group','add','experiments')
    settings=read_json(study/'state.json')['settings']
    client(study,'parallel','--group','experiments',settings['parallel'])
    command=subprocess.list2cmdline([sys.executable,str(study/'source/scripts/tongji_study.py'),'control','--study',str(study)])
    task_id=client(study,'add','--print-task-id','--working-directory',study/'source','--label','tongji-study-controller','--',command).strip()
    atomic_json(study/'launcher.json',dict(controller_task_id=task_id,daemon_pid=process.pid))
    print(json.dumps(dict(study_dir=str(study),controller_task_id=task_id,parallel=settings['parallel'],
                          cells={k:len(read_json(study/'splits.json')[k]) for k in ['train','val','test']}),indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['launch','control','report'])
    parser.add_argument('--config',type=Path)
    parser.add_argument('--study',type=Path)
    args=parser.parse_args()
    if args.action=='launch':launch(args.config)
    elif args.action=='control':control(args.study.resolve())
    else:
        study=args.study.resolve();state=read_json(study/'state.json');aggregate(study,state);write_report(study,state)


if __name__=='__main__':main()
