"""Data loading utilities for NASA PROMISE datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
from scipy.io import arff
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from .utils import ensure_dir


DEFAULT_KC1_URL: str = "https://promise.softwareengineering.ca/Downloads/defect_data/KC1.arff"


def get_dataset_path(data_dir: Path, filename: str) -> Path:
    """Build the dataset path under the raw data directory."""
    return data_dir / "raw" / filename


def download_dataset(url: str, dest_path: Path) -> Path:
    """Download a dataset file to the raw data folder."""
    try:
        ensure_dir(dest_path.parent)
        with urlopen(url) as response:
            dest_path.write_bytes(response.read())
        print(f"[INFO] Downloaded dataset to {dest_path}")
        return dest_path
    except (HTTPError, URLError, OSError) as exc:
        print(f"[ERROR] Failed to download dataset from {url}")
        raise exc


def _decode_object_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Decode byte columns to UTF-8 strings."""
    try:
        object_cols = df.select_dtypes(include=["object"]).columns.tolist()
        for col in object_cols:
            df[col] = df[col].apply(
                lambda value: value.decode("utf-8") if isinstance(value, (bytes, bytearray)) else value
            )
        return df
    except (AttributeError, UnicodeDecodeError) as exc:
        print("[ERROR] Failed to decode object columns")
        raise exc


def load_raw_data(path: Path) -> pd.DataFrame:
    """Load raw dataset from CSV or ARFF format."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path}")
    try:
        if path.suffix.lower() == ".arff":
            raw_data, _meta = arff.loadarff(path)
            df = pd.DataFrame(raw_data)
            df = _decode_object_columns(df)
            return df
        df = pd.read_csv(path)
        return df
    except (ValueError, OSError, pd.errors.ParserError, arff.ParseError) as exc:
        print(f"[ERROR] Failed to load dataset from {path}")
        raise exc


def load_or_download(data_dir: Path, filename: str = "KC1.arff", url: Optional[str] = None) -> pd.DataFrame:
    """Load dataset, downloading it if missing."""
    try:
        dataset_path = get_dataset_path(data_dir, filename)
        if not dataset_path.exists():
            dataset_url = url or DEFAULT_KC1_URL
            download_dataset(dataset_url, dataset_path)
        df = load_raw_data(dataset_path)
        print(f"[INFO] Loaded dataset with shape: {df.shape}")
        return df
    except (FileNotFoundError, OSError, ValueError, pd.errors.ParserError) as exc:
        print("[ERROR] Unable to load or download dataset")
        raise exc
