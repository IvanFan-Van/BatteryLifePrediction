"""Entry point for a frozen study trial."""
import os
from pathlib import Path
import sys
os.environ.update(WANDB_MODE="offline",WANDB_SILENT="true",CUBLAS_WORKSPACE_CONFIG=":4096:8")
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from training.study_worker import main

if __name__=="__main__":
    main()
