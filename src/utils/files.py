"""Read configuration and write experiment metadata."""
import json
from pathlib import Path

import yaml


def read_yaml(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
