"""Queue a frozen Tongji experiment with a saved config and source snapshot.

    uv run python scripts/queue_tongji.py --config configs/tongji_curriculum_pilot.yaml
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from uuid import uuid4

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils.files import read_yaml, write_json
from src.utils.paths import OUTPUTS_DIR, processed_data_path
from src.utils.runtime import clean_environment
from src.utils.snapshot import copy_source, source_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--pueue-config', type=Path)
    args = parser.parse_args()
    config = read_yaml(args.config)
    config['data']['root'] = str(processed_data_path(config['data']['root']))
    run_id = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-') + uuid4().hex[:6]
    run_dir = OUTPUTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir/'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    files = source_manifest(ROOT)
    copy_source(ROOT, run_dir / 'source', [*files, 'pyproject.toml', 'uv.lock', 'README.md'])
    arguments = [
        sys.executable, '-m', 'src.core.train', '--config', str(run_dir / 'config.yaml'),
        '--run-dir', str(run_dir),
    ]
    command = subprocess.list2cmdline(arguments) if os.name == 'nt' else shlex.join(arguments)
    queued = dict(command=command, source_sha256=files, config_source=str(args.config.resolve()))
    write_json(run_dir / 'queued.json', queued)
    client = ['pueue']
    if args.pueue_config:
        client += ['--config', str(args.pueue_config.resolve())]
    output = subprocess.check_output([*client, 'add', '--print-task-id', '--working-directory',
                                      str(run_dir/'source'), '--label', 'tongji-curriculum', '--', command],
                                     text=True, env=clean_environment())
    queued['task_id'] = output.strip()
    write_json(run_dir / 'queued.json', queued)
    print(json.dumps(dict(run_dir=str(run_dir),task_id=output.strip()),indent=2))


if __name__ == '__main__':
    main()
