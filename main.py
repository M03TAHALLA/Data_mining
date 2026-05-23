"""Main pipeline runner for software defect prediction."""

from __future__ import annotations

from pathlib import Path

from src.pipeline import run_pipeline


if __name__ == "__main__":
    run_pipeline(Path(__file__).resolve().parent)
