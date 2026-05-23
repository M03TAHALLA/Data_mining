"""Streamlit dashboard for software defect prediction."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st
from scipy.io import arff

import sys

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))

from src.dss import classify_risk_level, recommendation_for_feature
from src.data_loader import load_raw_data_bytes
from src.pipeline import run_pipeline
from src.utils import load_model, load_scaler


st.set_page_config(page_title="Software Defect Prediction DSS", layout="wide")

if "analysis_ready" not in st.session_state:
    st.session_state.analysis_ready = False
if "last_run_signature" not in st.session_state:
    st.session_state.last_run_signature = None


@st.cache_data
def load_datasets() -> Optional[Dict[str, pd.DataFrame]]:
    """Load processed datasets and reports."""
    data_dir = BASE_DIR / "data" / "processed"
    reports_dir = BASE_DIR / "reports"
    try:
        processed = pd.read_csv(data_dir / "processed_full.csv").reset_index(drop=True)
        comparison = pd.read_csv(reports_dir / "model_comparison.csv")
        risk = pd.read_csv(reports_dir / "risk_ranking.csv")
        if "Module_ID" in risk.columns:
            risk["Module_ID"] = pd.to_numeric(risk["Module_ID"], errors="coerce").fillna(0).astype(int)
        return {"processed": processed, "comparison": comparison, "risk": risk}
    except (OSError, pd.errors.ParserError) as exc:
        return None


@st.cache_resource
def load_artifacts() -> Optional[Dict[str, Any]]:
    """Load model and scaler artifacts."""
    reports_dir = BASE_DIR / "reports"
    processed_dir = BASE_DIR / "data" / "processed"
    try:
        model = load_model(reports_dir / "best_model.joblib")
        scaler = load_scaler(processed_dir / "scaler.joblib")
        feature_meta = json.loads((reports_dir / "feature_names.json").read_text(encoding="utf-8"))
        return {"model": model, "scaler": scaler, "feature_names": feature_meta["feature_names"]}
    except (OSError, json.JSONDecodeError, KeyError) as exc:
        return None


@st.cache_data
def parse_uploaded_dataset(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Parse an uploaded dataset into a DataFrame."""
    return load_raw_data_bytes(file_bytes, filename)


def _default_target_index(columns: List[str]) -> int:
    """Choose a default target column index."""
    for idx, col in enumerate(columns):
        if col.strip().lower() == "defects":
            return idx
    return 0


st.title("Software Defect Prediction — Decision Support System")

st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV or ARFF", type=["csv", "arff"])
uploaded_df: Optional[pd.DataFrame] = None
target_col: Optional[str] = None

if uploaded_file is None:
    if not st.session_state.analysis_ready:
        st.info("Upload a dataset to start the analysis.")
        st.stop()
else:
    try:
        file_bytes = uploaded_file.getvalue()
        uploaded_df = parse_uploaded_dataset(file_bytes, uploaded_file.name)
        default_index = _default_target_index(uploaded_df.columns.tolist())
        target_col = st.sidebar.selectbox(
            "Target column",
            uploaded_df.columns.tolist(),
            index=default_index,
        )
        st.sidebar.caption(f"Rows: {uploaded_df.shape[0]} | Columns: {uploaded_df.shape[1]}")
    except (OSError, ValueError, pd.errors.ParserError, arff.ArffError) as exc:
        st.sidebar.error("Failed to read uploaded file.")
        st.stop()

    if uploaded_df is not None:
        signature = (hashlib.sha256(file_bytes).hexdigest(), target_col)
        if signature != st.session_state.last_run_signature:
            with st.spinner("Running full analysis..."):
                try:
                    run_pipeline(BASE_DIR, df=uploaded_df, target_col=target_col or "defects")
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    st.error("Pipeline failed. Check the console logs for details.")
                    st.stop()
            st.cache_data.clear()
            st.cache_resource.clear()
            st.session_state.last_run_signature = signature
            st.session_state.analysis_ready = True
            st.success("Analysis complete. Dashboard updated.")

datasets = load_datasets()
artifacts = load_artifacts()

if datasets is None or artifacts is None:
    st.info("Run the pipeline (main.py) or upload a dataset to generate analysis outputs.")
    st.stop()

df = datasets["processed"]
comparison = datasets["comparison"]
risk = datasets["risk"]

feature_names: List[str] = artifacts["feature_names"]
model = artifacts["model"]
scaler = artifacts["scaler"]

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["📊 Data Overview", "🔍 Exploratory Analysis", "🤖 Model Performance", "🚨 Risk Dashboard", "🔎 Module Inspector", "📈 Feature Importance"]
)

with tab1:
    st.subheader("Dataset Overview")
    st.write(f"Shape: {df.shape}")
    st.dataframe(df.dtypes.rename("dtype").to_frame())

    st.subheader("Missing Values")
    st.dataframe(df.isna().sum().to_frame("Missing Values"))

    st.subheader("Class Distribution")
    class_counts = df["defects"].value_counts().reset_index()
    class_counts.columns = ["defects", "count"]
    fig = px.bar(class_counts, x="defects", y="count", text="count", title="Class Distribution")
    st.plotly_chart(fig, width='stretch')

    st.subheader("Summary Statistics")
    st.dataframe(df.describe())

with tab2:
    st.subheader("Correlation Heatmap")
    corr = df.corr(numeric_only=True)
    heatmap_fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu", title="Correlation Heatmap")
    st.plotly_chart(heatmap_fig, width='stretch')

    st.subheader("Feature Histogram by Class")
    feature = st.selectbox("Select Feature", feature_names)
    hist_fig = px.histogram(df, x=feature, color="defects", barmode="overlay", title=f"Histogram: {feature}")
    st.plotly_chart(hist_fig, width='stretch')

    st.subheader("Boxplot by Class")
    box_fig = px.box(df, x="defects", y=feature, color="defects", title=f"Boxplot: {feature}")
    st.plotly_chart(box_fig, width='stretch')

with tab3:
    st.subheader("Model Metrics")
    model_choice = st.selectbox("Select Model", comparison["Model"].tolist())
    row = comparison[comparison["Model"] == model_choice].iloc[0]

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Accuracy", f"{row['Accuracy']:.3f}")
    col2.metric("Precision", f"{row['Precision']:.3f}")
    col3.metric("Recall", f"{row['Recall']:.3f}")
    col4.metric("F1", f"{row['F1']:.3f}")
    col5.metric("ROC-AUC", f"{row['ROC-AUC']:.3f}")

    st.subheader("Confusion Matrix")
    cm_path = BASE_DIR / "reports" / "figures" / f"cm_{model_choice.lower().replace(' ', '_')}.png"
    if cm_path.exists():
        st.image(str(cm_path))
    else:
        st.warning("Confusion matrix image not found.")

    st.subheader("ROC Curve Comparison")
    roc_path = BASE_DIR / "reports" / "figures" / "roc_curves_comparison.png"
    if roc_path.exists():
        st.image(str(roc_path))
    else:
        st.warning("ROC curve image not found.")

    st.subheader("All Models Comparison")
    comp_fig = px.bar(
        comparison.melt(id_vars="Model", var_name="Metric", value_name="Value"),
        x="Model",
        y="Value",
        color="Metric",
        barmode="group",
        title="Model Comparison",
    )
    st.plotly_chart(comp_fig, width='stretch')

with tab4:
    st.subheader("Risk KPIs")
    total_modules = len(risk)
    defective = int(df["defects"].sum())
    critical = int(risk["Risk_Level"].str.contains("CRITICAL").sum())
    high = int(risk["Risk_Level"].str.contains("HIGH").sum())

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Modules", total_modules)
    k2.metric("Defective", defective)
    k3.metric("Critical Risk", critical)
    k4.metric("High Risk", high)

    st.subheader("Risk Ranking Table")
    def highlight_risk(row: pd.Series) -> List[str]:
        """Apply colors based on risk level."""
        try:
            color_map = {
                "🟢 LOW RISK": "background-color: #d4f8d4",
                "🟡 MEDIUM RISK": "background-color: #fff3cd",
                "🟠 HIGH RISK": "background-color: #ffe0b2",
                "🔴 CRITICAL RISK": "background-color: #f8d7da",
            }
            return [color_map.get(row["Risk_Level"], "")] * len(row)
        except KeyError as exc:
            st.warning("Risk level missing for row.")
            raise exc

    st.dataframe(risk.style.apply(highlight_risk, axis=1))

    st.subheader("Risk Distribution")
    risk_counts = risk["Risk_Level"].value_counts().reset_index()
    risk_counts.columns = ["Risk_Level", "Count"]
    pie_fig = px.pie(risk_counts, names="Risk_Level", values="Count", title="Risk Distribution")
    st.plotly_chart(pie_fig, width='stretch')

    st.subheader("Top 10 Most At-Risk Modules")
    top_risk = risk.head(10)
    bar_fig = px.bar(top_risk, x="Module_ID", y="Risk_Score", color="Risk_Level", title="Top 10 Risk Modules")
    st.plotly_chart(bar_fig, width='stretch')

with tab5:
    st.subheader("Module Inspector")
    module_id = st.selectbox("Select Module ID", risk["Module_ID"].tolist())
    module_row = df.iloc[int(module_id)][feature_names]
    proba = model.predict_proba(module_row.to_frame().T)[:, 1][0] * 100
    risk_level = classify_risk_level(proba)
    top_feature = module_row.abs().idxmax()
    recommendation = recommendation_for_feature(str(top_feature))

    st.write(f"Predicted Defective: {'Yes' if proba >= 50 else 'No'}")
    gauge_fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=proba,
            title={"text": "Risk Score"},
            gauge={"axis": {"range": [0, 100]}},
        )
    )
    st.plotly_chart(gauge_fig, width='stretch')
    st.write(f"Risk Level: {risk_level}")
    st.write(f"Top Risk Factor: {top_feature}")
    st.write(f"Recommendation: {recommendation}")

    if st.checkbox("Manual Metric Input"):
        with st.form("manual_input_form"):
            inputs: Dict[str, float] = {}
            for feat in feature_names:
                inputs[feat] = st.number_input(feat, value=float(df[feat].mean()))
            submitted = st.form_submit_button("Predict")
            if submitted:
                input_df = pd.DataFrame([inputs])[feature_names]
                scaled = scaler.transform(input_df)
                proba_manual = model.predict_proba(scaled)[:, 1][0] * 100
                st.write(f"Predicted Defective: {'Yes' if proba_manual >= 50 else 'No'}")
                st.write(f"Risk Score: {proba_manual:.2f}")
                st.write(f"Risk Level: {classify_risk_level(proba_manual)}")

with tab6:
    st.subheader("Feature Importance (Random Forest)")
    rf_path = BASE_DIR / "reports" / "figures" / "feature_importance_rf.png"
    if rf_path.exists():
        st.image(str(rf_path))
    else:
        st.warning("Random Forest feature importance not found.")

    st.subheader("Feature Importance (XGBoost)")
    xgb_path = BASE_DIR / "reports" / "figures" / "feature_importance_xgb.png"
    if xgb_path.exists():
        st.image(str(xgb_path))
    else:
        st.warning("XGBoost feature importance not found.")

    st.subheader("SHAP Summary (if available)")
    try:
        import shap

        sample = df[feature_names].sample(n=min(200, len(df)), random_state=42)
        explainer = shap.Explainer(model, sample)
        shap_values = explainer(sample)
        plt.figure()
        shap.summary_plot(shap_values, sample, show=False)
        st.pyplot(plt.gcf())
    except ImportError:
        st.info("SHAP is not installed. Install shap to view explanations.")
    except (ValueError, RuntimeError, TypeError) as exc:
        st.warning("SHAP plot could not be generated.")
