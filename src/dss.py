"""Decision Support System (DSS) for defect prediction."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List

import pandas as pd

from .utils import ensure_dir, save_dataframe


def calculate_risk_scores(model: Any, X: pd.DataFrame) -> pd.Series:
    """Calculate risk scores as probabilities scaled to 0-100."""
    try:
        proba = model.predict_proba(X)[:, 1]
        return pd.Series(proba * 100, index=X.index, name="Risk_Score")
    except (AttributeError, ValueError) as exc:
        print("[ERROR] Failed to calculate risk scores")
        raise exc


def classify_risk_level(score: float) -> str:
    """Classify risk score into labeled risk levels."""
    try:
        if score <= 30:
            return "🟢 LOW RISK"
        if score <= 60:
            return "🟡 MEDIUM RISK"
        if score <= 80:
            return "🟠 HIGH RISK"
        return "🔴 CRITICAL RISK"
    except (TypeError, ValueError) as exc:
        print("[ERROR] Failed to classify risk level")
        raise exc


def _recommendation_for_feature(feature: str) -> str:
    """Generate recommendation based on dominant feature."""
    try:
        feature_lower = feature.lower()
        if "v(g)" in feature_lower or "complex" in feature_lower:
            return "Reduce cyclomatic complexity"
        if "loc" in feature_lower:
            return "Split into smaller functions"
        if "locode" in feature_lower:
            return "Refactor large code blocks"
        if "coverage" in feature_lower:
            return "Increase unit test coverage"
        if "branch" in feature_lower:
            return "Simplify branching logic"
        return "Review and refactor high-risk metrics"
    except (AttributeError, TypeError) as exc:
        print("[ERROR] Failed to create recommendation")
        raise exc


def recommendation_for_feature(feature: str) -> str:
    """Public wrapper for recommendations engine."""
    try:
        return _recommendation_for_feature(feature)
    except (AttributeError, TypeError) as exc:
        print("[ERROR] Failed to generate recommendation")
        raise exc


def build_risk_report(
    model: Any,
    X: pd.DataFrame,
    output_dir: Path,
    module_ids: List[Any] | None = None,
) -> pd.DataFrame:
    """Build the DSS risk ranking report."""
    try:
        ensure_dir(output_dir)
        risk_scores = calculate_risk_scores(model, X)
        module_series = pd.Series(module_ids if module_ids is not None else X.index, name="Module_ID")
        top_factors = X.abs().idxmax(axis=1)
        report = pd.DataFrame(
            {
                "Module_ID": module_series,
                "Risk_Score": risk_scores.values,
                "Risk_Level": [classify_risk_level(score) for score in risk_scores.values],
                "Top_Risk_Factor": top_factors.values,
                "Recommendation": [recommendation_for_feature(feature) for feature in top_factors.values],
            }
        )
        report = report.sort_values(by="Risk_Score", ascending=False).reset_index(drop=True)
        save_dataframe(report, output_dir / "risk_ranking.csv")
        print("[INFO] DSS risk report saved")
        return report
    except (ValueError, TypeError, OSError) as exc:
        print("[ERROR] Failed to build DSS report")
        raise exc


def print_alerts(report: pd.DataFrame) -> None:
    """Print alerts for critical modules."""
    try:
        critical = report[report["Risk_Level"].str.contains("CRITICAL", na=False)]
        for _, row in critical.iterrows():
            print(f"⚠️ ALERT: Module {row['Module_ID']} — Risk: {row['Risk_Score']:.1f}% → Immediate review required")
    except (KeyError, ValueError) as exc:
        print("[ERROR] Failed to print alerts")
        raise exc


def build_priority_queue(report: pd.DataFrame) -> List[Any]:
    """Create a prioritized list of module IDs for QA."""
    try:
        queue = report.sort_values(by="Risk_Score", ascending=False)["Module_ID"].tolist()
        print("[INFO] QA Priority Queue:")
        print(queue)
        return queue
    except (KeyError, ValueError) as exc:
        print("[ERROR] Failed to build priority queue")
        raise exc
