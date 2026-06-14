#!/usr/bin/env python
"""Root entry point for the ECG Heartbeat Classifier.

Run from the project root:
    python main.py train --epochs 30
    python main.py evaluate
    python main.py predict --sample-index 100
"""
from src.cli import main

if __name__ == "__main__":
    main()
