"""Model training and comparison module."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from .evaluator import evaluate_model, plot_roc_curves, save_model_comparison
from .utils import RANDOM_STATE, ensure_dir

try:
    from xgboost import XGBClassifier
except ImportError as exc:
    XGBClassifier = None
    print("[WARN] XGBoost not available. Install xgboost to enable.")

try:
    from lightgbm import LGBMClassifier
except ImportError as exc:
    LGBMClassifier = None
    print("[WARN] LightGBM not available. Install lightgbm to enable.")


@dataclass
class ModelArtifacts:
    """Container for model training outputs."""

    models: Dict[str, Any]
    comparison: pd.DataFrame
    best_model_name: str
    best_model: Any


def _build_models() -> Dict[str, Any]:
    """Initialize all models with required parameters."""
    try:
        models: Dict[str, Any] = {
            "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
            "SVM": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            "Naive Bayes": GaussianNB(),
        }
        if XGBClassifier is not None:
            models["XGBoost"] = XGBClassifier(
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=RANDOM_STATE,
            )
        if LGBMClassifier is not None:
            models["LightGBM"] = LGBMClassifier(random_state=RANDOM_STATE, verbose=-1)
        return models
    except (TypeError, ValueError) as exc:
        print("[ERROR] Failed to build model registry")
        raise exc


def _get_proba(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Get probability predictions for the positive class."""
    try:
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)[:, 1]
        if hasattr(model, "decision_function"):
            scores = model.decision_function(X)
            denom = scores.max() - scores.min()
            if denom == 0:
                return np.zeros_like(scores, dtype=float)
            return (scores - scores.min()) / denom
        raise AttributeError("Model does not support probability predictions.")
    except (AttributeError, ValueError) as exc:
        print("[ERROR] Unable to compute probability estimates")
        raise exc


def _pick_best_models(comparison: pd.DataFrame) -> Tuple[str, str]:
    """Select the best two models based on recall and ROC-AUC."""
    try:
        ranked = comparison.sort_values(by=["Recall", "ROC-AUC"], ascending=False)
        best = ranked.iloc[0]["Model"]
        second = ranked.iloc[1]["Model"] if len(ranked) > 1 else ranked.iloc[0]["Model"]
        return str(best), str(second)
    except (KeyError, IndexError, ValueError) as exc:
        print("[ERROR] Failed to select best models")
        raise exc


def _grid_params(model_name: str) -> Dict[str, List[Any]]:
    """Return parameter grid based on model name."""
    try:
        grids = {
            "Random Forest": {
                "n_estimators": [100, 200],
                "max_depth": [None, 5, 10],
                "min_samples_split": [2, 5],
            },
            "Logistic Regression": {
                "C": [0.1, 1.0, 10.0],
                "penalty": ["l2"],
                "solver": ["lbfgs"],
            },
            "SVM": {
                "C": [0.5, 1.0, 2.0],
                "gamma": ["scale", "auto"],
            },
            "Decision Tree": {
                "max_depth": [3, 5, 7, None],
                "min_samples_split": [2, 5],
            },
            "XGBoost": {
                "n_estimators": [100, 200],
                "max_depth": [3, 5],
                "learning_rate": [0.05, 0.1],
            },
            "LightGBM": {
                "n_estimators": [100, 200],
                "num_leaves": [31, 63],
                "learning_rate": [0.05, 0.1],
            },
        }
        return grids.get(model_name, {})
    except (TypeError, KeyError) as exc:
        print("[ERROR] Failed to build parameter grid")
        raise exc


def _cross_validate_best(model: Any, X: pd.DataFrame, y: pd.Series) -> None:
    """Run 5-fold stratified cross-validation on the best model."""
    try:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        scoring = {
            "accuracy": "accuracy",
            "precision": "precision",
            "recall": "recall",
            "f1": "f1",
            "roc_auc": "roc_auc",
        }
        results = cross_validate(model, X, y, cv=skf, scoring=scoring, n_jobs=-1)
        print("\n[INFO] Cross-Validation Results:")
        for metric, values in results.items():
            if metric.startswith("test_"):
                mean_val = np.mean(values)
                std_val = np.std(values)
                print(f"{metric.replace('test_', '').upper()}: {mean_val:.4f} ± {std_val:.4f}")
    except (ValueError, TypeError) as exc:
        print("[ERROR] Cross-validation failed")
        raise exc


def _grid_search(models: Dict[str, Any], X: pd.DataFrame, y: pd.Series, model_names: List[str]) -> None:
    """Perform GridSearchCV on the top 2 models."""
    try:
        for name in model_names:
            grid = _grid_params(name)
            if not grid:
                print(f"[WARN] No grid configured for {name}")
                continue
            gs = GridSearchCV(
                models[name],
                grid,
                scoring="recall",
                cv=3,
                n_jobs=-1,
            )
            gs.fit(X, y)
            print(f"[INFO] Best params for {name}: {gs.best_params_}")
    except (ValueError, TypeError, KeyError) as exc:
        print("[ERROR] Grid search failed")
        raise exc


def _plot_feature_importance(model: Any, feature_names: List[str], output_dir: Path, filename: str) -> None:
    """Plot and save feature importance."""
    try:
        if not hasattr(model, "feature_importances_"):
            print(f"[WARN] Feature importance not available for {filename}")
            return
        importances = model.feature_importances_
        sorted_idx = np.argsort(importances)[::-1][:15]
        top_features = np.array(feature_names)[sorted_idx]
        top_importances = importances[sorted_idx]

        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(8, 4))
        sns.barplot(x=top_importances, y=top_features, palette="mako")
        plt.title(f"Feature Importance - {filename}")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.tight_layout()
        path = output_dir / filename
        plt.savefig(path)
        plt.show()
        print(f"[INFO] Saved {path}")
    except (ValueError, OSError) as exc:
        print("[ERROR] Failed to plot feature importance")
        raise exc


def train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: Path,
    feature_names: List[str],
) -> ModelArtifacts:
    """Train all models and evaluate performance."""
    try:
        ensure_dir(output_dir)
        models = _build_models()
        results: List[Dict[str, float]] = []
        roc_curves: List[Dict[str, Any]] = []

        for name, model in models.items():
            print(f"[INFO] Training {name}")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = _get_proba(model, X_test)
            metrics, roc_info = evaluate_model(
                model_name=name,
                y_true=y_test,
                y_pred=y_pred,
                y_proba=y_proba,
                output_dir=output_dir,
            )
            results.append(metrics)
            roc_curves.append(roc_info)

        comparison = save_model_comparison(results, output_dir.parent)
        plot_roc_curves(roc_curves, output_dir)

        best_model_name, second_model_name = _pick_best_models(comparison)
        best_model = models[best_model_name]
        print(f"[INFO] Best model based on recall: {best_model_name}")

        _cross_validate_best(best_model, X_train, y_train)
        _grid_search(models, X_train, y_train, [best_model_name, second_model_name])

        if "Random Forest" in models:
            _plot_feature_importance(models["Random Forest"], feature_names, output_dir, "feature_importance_rf.png")
        if "XGBoost" in models:
            _plot_feature_importance(models["XGBoost"], feature_names, output_dir, "feature_importance_xgb.png")

        return ModelArtifacts(
            models=models,
            comparison=comparison,
            best_model_name=best_model_name,
            best_model=best_model,
        )
    except (ValueError, TypeError, KeyError, OSError) as exc:
        print("[ERROR] Model training failed")
        raise exc
