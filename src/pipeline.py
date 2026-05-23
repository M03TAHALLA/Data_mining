"""Shared pipeline runner for software defect prediction."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd

from .data_loader import load_or_download
from .preprocessor import preprocess_data, PreprocessResult
from .eda import run_eda
from .models import train_models, ModelArtifacts
from .dss import build_priority_queue, build_risk_report, print_alerts
from .utils import ensure_dir, print_step, save_json, save_model


def run_pipeline(
    base_dir: Path,
    df: Optional[pd.DataFrame] = None,
    target_col: str = "defects",
    filename: str = "KC1.arff",
    url: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the full defect prediction pipeline."""
    try:
        data_dir = base_dir / "data"
        processed_dir = data_dir / "processed"
        reports_dir = base_dir / "reports"
        figures_dir = reports_dir / "figures"

        ensure_dir(data_dir / "raw")
        ensure_dir(processed_dir)
        ensure_dir(reports_dir)
        ensure_dir(figures_dir)

        if df is None:
            print_step(1, "Load raw data")
            df = load_or_download(data_dir=data_dir, filename=filename, url=url)
        else:
            print_step(1, "Load uploaded data")

        print_step(2, "Preprocess data")
        preprocess_result: PreprocessResult = preprocess_data(df, output_dir=processed_dir, target_col=target_col)

        print_step(3, "Run EDA")
        run_eda(preprocess_result.clean_df, target_col, figures_dir)

        print_step(4, "Train models")
        model_artifacts: ModelArtifacts = train_models(
            X_train=preprocess_result.X_train,
            y_train=preprocess_result.y_train,
            X_test=preprocess_result.X_test,
            y_test=preprocess_result.y_test,
            output_dir=figures_dir,
            feature_names=preprocess_result.feature_names,
        )

        print_step(5, "Evaluate and compare models")
        comparison_path = reports_dir / "model_comparison.csv"
        print(f"[INFO] Model comparison saved to {comparison_path}")

        print_step(6, "Generate DSS risk report")
        risk_report = build_risk_report(
            model=model_artifacts.best_model,
            X=preprocess_result.full_features,
            output_dir=reports_dir,
            module_ids=preprocess_result.full_features.index.tolist(),
        )
        print_alerts(risk_report)
        build_priority_queue(risk_report)

        print_step(7, "Save outputs")
        save_model(model_artifacts.best_model, reports_dir / "best_model.joblib")
        save_json({"feature_names": preprocess_result.feature_names}, reports_dir / "feature_names.json")

        print_step(8, "Pipeline summary")
        print(f"[INFO] Best model: {model_artifacts.best_model_name}")
        print(f"[INFO] Reports saved to {reports_dir}")

        return {
            "preprocess_result": preprocess_result,
            "model_artifacts": model_artifacts,
            "risk_report": risk_report,
        }
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print("[ERROR] Pipeline failed")
        raise exc
