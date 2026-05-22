"""Exploratory data analysis utilities."""

from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_classif

from .utils import ensure_dir


def plot_class_distribution(df: pd.DataFrame, target_col: str, output_dir: Path) -> None:
    """Plot and save class distribution."""
    try:
        ensure_dir(output_dir)
        counts = df[target_col].value_counts().sort_index()
        percentages = (counts / counts.sum()) * 100

        plt.figure(figsize=(6, 4))
        ax = sns.barplot(x=counts.index.astype(str), y=counts.values, palette="viridis")
        for idx, value in enumerate(counts.values):
            ax.text(idx, value, f"{percentages.iloc[idx]:.1f}%", ha="center", va="bottom")
        plt.title("Class Distribution")
        plt.xlabel("Defects")
        plt.ylabel("Count")
        plt.tight_layout()
        path = output_dir / "class_distribution.png"
        plt.savefig(path)
        plt.show()
        print(f"[INFO] Saved {path}")
    except (KeyError, ValueError, OSError) as exc:
        print("[ERROR] Failed to plot class distribution")
        raise exc


def plot_correlation_heatmap(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot and save correlation heatmap."""
    try:
        ensure_dir(output_dir)
        corr = df.corr(numeric_only=True)
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        path = output_dir / "correlation_heatmap.png"
        plt.savefig(path)
        plt.show()
        print(f"[INFO] Saved {path}")
    except (ValueError, OSError) as exc:
        print("[ERROR] Failed to plot correlation heatmap")
        raise exc


def plot_feature_histograms(df: pd.DataFrame, target_col: str, output_dir: Path) -> None:
    """Plot histograms for each numerical feature."""
    try:
        ensure_dir(output_dir)
        feature_cols = [col for col in df.columns if col != target_col]
        n_cols = 4
        n_rows = int(np.ceil(len(feature_cols) / n_cols))
        plt.figure(figsize=(16, max(4, n_rows * 3)))
        for idx, col in enumerate(feature_cols, start=1):
            plt.subplot(n_rows, n_cols, idx)
            sns.histplot(data=df, x=col, hue=target_col, bins=30, element="step", common_norm=False)
            plt.title(col)
        plt.tight_layout()
        path = output_dir / "feature_histograms.png"
        plt.savefig(path)
        plt.show()
        print(f"[INFO] Saved {path}")
    except (KeyError, ValueError, OSError) as exc:
        print("[ERROR] Failed to plot feature histograms")
        raise exc


def plot_boxplots_by_class(df: pd.DataFrame, target_col: str, output_dir: Path) -> None:
    """Plot boxplots for top correlated features by class."""
    try:
        ensure_dir(output_dir)
        corr = df.corr(numeric_only=True)[target_col].drop(target_col).abs()
        top_features = corr.sort_values(ascending=False).head(8).index.tolist()
        n_cols = 4
        n_rows = int(np.ceil(len(top_features) / n_cols))
        plt.figure(figsize=(16, max(4, n_rows * 3)))
        for idx, col in enumerate(top_features, start=1):
            plt.subplot(n_rows, n_cols, idx)
            sns.boxplot(data=df, x=target_col, y=col, palette="Set3")
            plt.title(col)
        plt.tight_layout()
        path = output_dir / "boxplots_by_class.png"
        plt.savefig(path)
        plt.show()
        print(f"[INFO] Saved {path}")
    except (KeyError, ValueError, OSError) as exc:
        print("[ERROR] Failed to plot boxplots")
        raise exc


def plot_feature_importance_statistical(df: pd.DataFrame, target_col: str, output_dir: Path) -> None:
    """Plot feature importance using SelectKBest."""
    try:
        ensure_dir(output_dir)
        X = df.drop(columns=[target_col])
        y = df[target_col]
        selector = SelectKBest(score_func=f_classif, k=min(10, X.shape[1]))
        selector.fit(X, y)
        scores = pd.Series(selector.scores_, index=X.columns).sort_values(ascending=False)
        plt.figure(figsize=(8, 4))
        sns.barplot(x=scores.values, y=scores.index, palette="crest")
        plt.title("Top Feature Importance (SelectKBest)")
        plt.xlabel("F-Score")
        plt.ylabel("Feature")
        plt.tight_layout()
        path = output_dir / "feature_importance_statistical.png"
        plt.savefig(path)
        plt.show()
        print(f"[INFO] Saved {path}")
    except (KeyError, ValueError, OSError) as exc:
        print("[ERROR] Failed to plot statistical feature importance")
        raise exc


def plot_outlier_detection(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot boxplots for selected features for outlier inspection."""
    try:
        ensure_dir(output_dir)
        candidate_features = ["loc", "v(g)", "ev(g)", "n"]
        available = [col for col in candidate_features if col in df.columns]
        if not available:
            print("[WARN] Outlier detection skipped: required columns not found")
            return
        plt.figure(figsize=(10, 4))
        sns.boxplot(data=df[available], orient="h")
        plt.title("Outlier Detection")
        plt.tight_layout()
        path = output_dir / "outliers_detection.png"
        plt.savefig(path)
        plt.show()
        print(f"[INFO] Saved {path}")
    except (KeyError, ValueError, OSError) as exc:
        print("[ERROR] Failed to plot outliers")
        raise exc


def print_statistical_summary(df: pd.DataFrame, target_col: str) -> None:
    """Print descriptive stats, skewness, kurtosis, and correlations."""
    try:
        numeric_df = df.select_dtypes(include=[np.number])
        print("[INFO] Statistical Summary:")
        print(numeric_df.describe())
        print("\n[INFO] Skewness:")
        print(numeric_df.skew())
        print("\n[INFO] Kurtosis:")
        print(numeric_df.kurtosis())
        print("\n[INFO] Pearson Correlation with Target:")
        corr = numeric_df.corr()[target_col].sort_values(ascending=False)
        print(corr)
    except (KeyError, ValueError) as exc:
        print("[ERROR] Failed to print statistical summary")
        raise exc


def run_eda(df: pd.DataFrame, target_col: str, output_dir: Path) -> None:
    """Run full exploratory data analysis workflow."""
    try:
        plot_class_distribution(df, target_col, output_dir)
        plot_correlation_heatmap(df, output_dir)
        plot_feature_histograms(df, target_col, output_dir)
        plot_boxplots_by_class(df, target_col, output_dir)
        plot_feature_importance_statistical(df, target_col, output_dir)
        plot_outlier_detection(df, output_dir)
        print_statistical_summary(df, target_col)
        print("[INFO] EDA completed")
    except (KeyError, ValueError, OSError) as exc:
        print("[ERROR] EDA failed")
        raise exc
