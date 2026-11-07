"""Queue a frozen Tongji experiment with a saved config and source snapshot.

    uv run python scripts/queue_tongji.py --config configs/tongji_smoke.yaml
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
from uuid import uuid4

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from train import source_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--pueue-config', type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    config['data']['root'] = str(Path(config['data']['root']).resolve())
    run_id = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-') + uuid4().hex[:6]
    run_dir = ROOT/'runs'/run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir/'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    files = source_manifest(ROOT)
    for name in [*files, 'pyproject.toml', 'uv.lock']:
        destination = run_dir/'source'/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, destination)
    command = subprocess.list2cmdline([sys.executable, str(run_dir/'source/train.py'),
                                      '--config', str(run_dir/'config.yaml'), '--run-dir', str(run_dir)])
    queued = dict(command=command, source_sha256=files, config_source=str(args.config.resolve()))
    (run_dir/'queued.json').write_text(json.dumps(queued, indent=2), encoding='utf-8')
    client = ['pueue']
    if args.pueue_config:
        client += ['--config', str(args.pueue_config.resolve())]
    output = subprocess.check_output([*client, 'add', '--print-task-id', '--working-directory',
                                      str(run_dir/'source'), '--label', 'tongji-single-cell-smoke', '--', command],text=True)
    queued['task_id'] = output.strip()
    (run_dir/'queued.json').write_text(json.dumps(queued, indent=2), encoding='utf-8')
    print(json.dumps(dict(run_dir=str(run_dir),task_id=output.strip()),indent=2))


if __name__ == '__main__':
    main()
