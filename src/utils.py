"""Utility helpers for the software defect prediction project."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import json
import joblib
import pandas as pd

RANDOM_STATE: int = 42


def ensure_dir(path: Path) -> None:
    """Create a directory if it does not exist."""
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"[ERROR] Failed to create directory: {path}")
        raise exc


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    """Save a DataFrame to CSV."""
    try:
        df.to_csv(path, index=False)
    except OSError as exc:
        print(f"[ERROR] Failed to save DataFrame to {path}")
        raise exc


def save_scaler(scaler: Any, path: Path) -> None:
    """Persist a scaler using joblib."""
    try:
        joblib.dump(scaler, path)
    except OSError as exc:
        print(f"[ERROR] Failed to save scaler to {path}")
        raise exc


def load_scaler(path: Path) -> Any:
    """Load a scaler from disk."""
    try:
        return joblib.load(path)
    except (OSError, FileNotFoundError) as exc:
        print(f"[ERROR] Failed to load scaler from {path}")
        raise exc


def save_model(model: Any, path: Path) -> None:
    """Save a trained model using joblib."""
    try:
        joblib.dump(model, path)
    except OSError as exc:
        print(f"[ERROR] Failed to save model to {path}")
        raise exc


def load_model(path: Path) -> Any:
    """Load a trained model from disk."""
    try:
        return joblib.load(path)
    except (OSError, FileNotFoundError) as exc:
        print(f"[ERROR] Failed to load model from {path}")
        raise exc


def save_json(data: Dict[str, Any], path: Path) -> None:
    """Save a dictionary to JSON."""
    try:
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except OSError as exc:
        print(f"[ERROR] Failed to save JSON to {path}")
        raise exc


def load_json(path: Path) -> Dict[str, Any]:
    """Load a dictionary from JSON."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[ERROR] Failed to load JSON from {path}")
        raise exc


def print_step(step: int, description: str) -> None:
    """Print a standardized progress header."""
    print("=" * 60)
    print(f"[STEP {step}] {description}")
    print("=" * 60)
