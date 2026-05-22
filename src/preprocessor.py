"""Data preprocessing pipeline for defect prediction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .utils import RANDOM_STATE, ensure_dir, save_dataframe, save_scaler


@dataclass
class PreprocessResult:
    """Container for preprocessing outputs."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    scaler: StandardScaler
    feature_names: List[str]
    full_features: pd.DataFrame
    full_target: pd.Series
    clean_df: pd.DataFrame


def _handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing numerical values with the median."""
    try:
        missing_before = df.isna().sum().sum()
        print(f"[INFO] Missing values before: {missing_before}")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            median_value = df[col].median()
            df[col] = df[col].fillna(median_value)
        missing_after = df.isna().sum().sum()
        print(f"[INFO] Missing values after: {missing_after}")
        return df
    except (KeyError, ValueError, TypeError) as exc:
        print("[ERROR] Failed handling missing values")
        raise exc


def _remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows."""
    try:
        duplicates = df.duplicated().sum()
        df = df.drop_duplicates()
        print(f"[INFO] Duplicates removed: {duplicates}")
        return df
    except (ValueError, KeyError) as exc:
        print("[ERROR] Failed removing duplicates")
        raise exc


def _encode_target(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Encode target column to binary values."""
    try:
        if target_col not in df.columns:
            raise KeyError(f"Target column '{target_col}' not found.")
        mapping = {
            "true": 1,
            "false": 0,
            "yes": 1,
            "no": 0,
            "y": 1,
            "n": 0,
            "defective": 1,
            "clean": 0,
            "1": 1,
            "0": 0,
        }
        if df[target_col].dtype == bool:
            df[target_col] = df[target_col].map({True: 1, False: 0})
        else:
            df[target_col] = (
                df[target_col]
                .astype(str)
                .str.strip()
                .str.lower()
                .map(mapping)
            )
        if df[target_col].isna().any():
            raise ValueError("Target encoding resulted in NaN values.")
        print("[INFO] Target distribution:")
        print(df[target_col].value_counts())
        return df
    except (KeyError, ValueError, TypeError) as exc:
        print("[ERROR] Failed encoding target variable")
        raise exc


def _remove_zero_variance_features(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, List[str]]:
    """Drop zero-variance features."""
    try:
        feature_cols = [col for col in df.columns if col != target_col]
        zero_var_cols = [col for col in feature_cols if df[col].nunique() <= 1]
        df = df.drop(columns=zero_var_cols)
        print(f"[INFO] Zero-variance columns dropped: {zero_var_cols}")
        return df, zero_var_cols
    except (KeyError, ValueError) as exc:
        print("[ERROR] Failed removing zero-variance features")
        raise exc


def _normalize_features(df: pd.DataFrame, target_col: str, output_dir: Path) -> Tuple[pd.DataFrame, StandardScaler]:
    """Scale numerical features using StandardScaler."""
    try:
        scaler = StandardScaler()
        feature_cols = [col for col in df.columns if col != target_col]
        scaled = scaler.fit_transform(df[feature_cols])
        scaled_df = pd.DataFrame(scaled, columns=feature_cols)
        scaled_df[target_col] = df[target_col].values
        scaler_path = output_dir / "scaler.joblib"
        save_scaler(scaler, scaler_path)
        print(f"[INFO] Saved scaler to {scaler_path}")
        return scaled_df, scaler
    except (ValueError, TypeError, OSError) as exc:
        print("[ERROR] Failed normalizing features")
        raise exc


def _apply_smote(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    apply_smote: bool,
) -> Tuple[pd.DataFrame, pd.Series, bool]:
    """Apply SMOTE if class imbalance is significant."""
    try:
        print(f"[INFO] Class distribution before SMOTE:\n{y_train.value_counts()}")
        if apply_smote:
            smote = SMOTE(random_state=RANDOM_STATE)
            X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
            print(f"[INFO] Class distribution after SMOTE:\n{y_resampled.value_counts()}")
            return X_resampled, y_resampled, True
        print("[INFO] SMOTE not applied (class ratio >= 0.30)")
        return X_train, y_train, False
    except (ValueError, TypeError) as exc:
        print("[ERROR] Failed applying SMOTE")
        raise exc


def preprocess_data(
    df: pd.DataFrame,
    output_dir: Path,
    target_col: str = "defects",
) -> PreprocessResult:
    """Run the complete preprocessing pipeline and save artifacts."""
    try:
        ensure_dir(output_dir)
        df = df.copy()
        df = _handle_missing_values(df)
        df = _remove_duplicates(df)
        df = _encode_target(df, target_col=target_col)
        df, _dropped_cols = _remove_zero_variance_features(df, target_col=target_col)
        clean_df = df.copy()
        scaled_df, scaler = _normalize_features(df, target_col=target_col, output_dir=output_dir)

        X = scaled_df.drop(columns=[target_col])
        y = scaled_df[target_col]
        feature_names = X.columns.tolist()

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            stratify=y,
            random_state=RANDOM_STATE,
        )
        class_counts = y.value_counts()
        minority_ratio = class_counts.min() / class_counts.sum()
        apply_smote = minority_ratio < 0.3
        X_train, y_train, smote_applied = _apply_smote(X_train, y_train, apply_smote)
        print(f"[INFO] SMOTE applied: {smote_applied}")

        save_dataframe(X_train, output_dir / "X_train.csv")
        save_dataframe(X_test, output_dir / "X_test.csv")
        save_dataframe(y_train.to_frame(name=target_col), output_dir / "y_train.csv")
        save_dataframe(y_test.to_frame(name=target_col), output_dir / "y_test.csv")
        save_dataframe(scaled_df, output_dir / "processed_full.csv")
        print(f"[INFO] Processed data saved to {output_dir}")

        return PreprocessResult(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            scaler=scaler,
            feature_names=feature_names,
            full_features=X,
            full_target=y,
            clean_df=clean_df,
        )
    except (KeyError, ValueError, TypeError, OSError) as exc:
        print("[ERROR] Preprocessing failed")
        raise exc
