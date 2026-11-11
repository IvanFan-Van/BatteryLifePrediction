"""Prepare Tongji data using the shared preprocessing implementation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.preprocess_tongji import main

if __name__ == "__main__":
    main()
