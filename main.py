"""Main pipeline runner for software defect prediction."""

from __future__ import annotations

from pathlib import Path

from src.data_loader import load_or_download
from src.preprocessor import preprocess_data
from src.eda import run_eda
from src.models import train_models
from src.dss import build_priority_queue, build_risk_report, print_alerts
from src.utils import ensure_dir, print_step, save_json, save_model


def run_pipeline() -> None:
    """Run the full defect prediction pipeline."""
    try:
        base_dir = Path(__file__).resolve().parent
        data_dir = base_dir / "data"
        processed_dir = data_dir / "processed"
        reports_dir = base_dir / "reports"
        figures_dir = reports_dir / "figures"

        ensure_dir(data_dir / "raw")
        ensure_dir(processed_dir)
        ensure_dir(reports_dir)
        ensure_dir(figures_dir)

        print_step(1, "Load raw data")
        df = load_or_download(data_dir=data_dir)

        print_step(2, "Preprocess data")
        preprocess_result = preprocess_data(df, output_dir=processed_dir, target_col="defects")

        print_step(3, "Run EDA")
        run_eda(preprocess_result.clean_df, "defects", figures_dir)

        print_step(4, "Train models")
        model_artifacts = train_models(
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
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print("[ERROR] Pipeline failed")
        raise exc


if __name__ == "__main__":
    run_pipeline()
