"""Prepare Tongji data using the shared preprocessing implementation."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.preprocess_tongji import main


if __name__ == "__main__":
    main()
