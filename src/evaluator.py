"""Model evaluation utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from .utils import ensure_dir


def evaluate_model(
    model_name: str,
    y_true: pd.Series,
    y_pred: pd.Series,
    y_proba: pd.Series,
    output_dir: Path,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Compute metrics, print report, and save confusion matrix."""
    try:
        ensure_dir(output_dir)
        print(f"\n[MODEL] {model_name}")
        print(classification_report(y_true, y_pred, zero_division=0))

        metrics = {
            "Model": model_name,
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred, zero_division=0),
            "F1": f1_score(y_true, y_pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_true, y_proba),
        }

        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix - {model_name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        cm_path = output_dir / f"cm_{model_name.lower().replace(' ', '_')}.png"
        plt.savefig(cm_path)
        plt.show()
        print(f"[INFO] Saved {cm_path}")

        fpr, tpr, _thresholds = roc_curve(y_true, y_proba)
        roc_info = {"Model": model_name, "fpr": fpr, "tpr": tpr, "roc_auc": metrics["ROC-AUC"]}
        return metrics, roc_info
    except (ValueError, KeyError, OSError) as exc:
        print("[ERROR] Evaluation failed")
        raise exc


def plot_roc_curves(roc_curves: List[Dict[str, Any]], output_dir: Path) -> None:
    """Plot and save ROC curves for all models."""
    try:
        ensure_dir(output_dir)
        plt.figure(figsize=(7, 5))
        for roc_data in roc_curves:
            plt.plot(roc_data["fpr"], roc_data["tpr"], label=f"{roc_data['Model']} (AUC={roc_data['roc_auc']:.2f})")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curves Comparison")
        plt.legend(loc="lower right")
        plt.tight_layout()
        path = output_dir / "roc_curves_comparison.png"
        plt.savefig(path)
        plt.show()
        print(f"[INFO] Saved {path}")
    except (ValueError, OSError) as exc:
        print("[ERROR] Failed to plot ROC curves")
        raise exc


def save_model_comparison(results: List[Dict[str, float]], output_dir: Path) -> pd.DataFrame:
    """Save model comparison table to CSV."""
    try:
        ensure_dir(output_dir)
        df = pd.DataFrame(results)
        df = df[["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]]
        path = output_dir / "model_comparison.csv"
        df.to_csv(path, index=False)
        print(f"[INFO] Saved {path}")
        return df
    except (ValueError, OSError, KeyError) as exc:
        print("[ERROR] Failed to save model comparison")
        raise exc
