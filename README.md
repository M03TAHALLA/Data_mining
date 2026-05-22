# Software Defect Prediction System

End-to-end, production-ready pipeline to predict software defects using NASA PROMISE KC1/JM1 metrics, with full preprocessing, EDA, ML model training, evaluation, decision support, and a Streamlit dashboard.

## Project Structure

```
software_defect_prediction/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Preprocessing.ipynb
│   ├── 03_ML_Models.ipynb
│   ├── 04_DSS_Dashboard.ipynb
│   └── 05_Bonus_Rules_Clustering.ipynb
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessor.py
│   ├── eda.py
│   ├── models.py
│   ├── evaluator.py
│   ├── dss.py
│   └── utils.py
├── dashboard/
│   └── app.py
├── reports/
│   └── figures/
├── requirements.txt
├── README.md
└── main.py
```

## Installation

```bash
pip install -r requirements.txt
```

## How to Run

1. Run the full pipeline:
```bash
python main.py
```

2. Launch the dashboard:
```bash
streamlit run dashboard/app.py
```

## Dataset

- Source: https://promise.softwareengineering.ca/
- Recommended dataset: KC1 (NASA PROMISE)
- Target variable: `defects` (binary)
- Features: software metrics (complexity, Halstead metrics, LOC, branch counts)
- The pipeline will download `KC1.arff` automatically into `data/raw/` if it is missing.

## Models Used

- Decision Tree
- Random Forest
- SVM (RBF)
- Logistic Regression
- Naive Bayes
- XGBoost
- LightGBM

Model metrics are saved to `reports/model_comparison.csv`. ROC curves and confusion matrices are stored in `reports/figures/`.

## Results Summary

The pipeline compares all models using Accuracy, Precision, Recall, F1, and ROC-AUC. Recall is prioritized for defect detection. The best model is selected automatically and used in the Decision Support System.

## How to Test the System

- Run the pipeline end-to-end with `python main.py`
- Validate outputs in `reports/` and `data/processed/`
- Open the Streamlit dashboard and verify interactive outputs

## Screenshots

Add dashboard and report screenshots here.
