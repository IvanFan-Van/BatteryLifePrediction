"""Paths relative to the project or a frozen source snapshot."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
TONGJI_DATA_DIR = PROCESSED_DATA_DIR / "tongji_v1"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def processed_data_path(value):
    """Resolve the old root from saved Tongji experiment configs."""
    path = Path(value).expanduser().resolve()
    if path == DATA_DIR / "tongji_v1" and TONGJI_DATA_DIR.is_dir():
        return TONGJI_DATA_DIR
    return path


def run_path(value):
    path = Path(value).expanduser().resolve()
    legacy = PROJECT_ROOT / "runs"
    if path.is_relative_to(legacy):
        migrated = OUTPUTS_DIR / path.relative_to(legacy)
        if migrated.exists():
            return migrated
    return path
