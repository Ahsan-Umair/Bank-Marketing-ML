# Bank Marketing Classification

A machine-learning study of whether a bank client will subscribe to a term deposit after a marketing campaign. The notebook focuses on class imbalance, leakage-aware feature selection, reliable cross-validation, regularized logistic regression, and an end-to-end preprocessing pipeline.

## Workflow

- Load the semicolon-delimited Bank Marketing dataset and inspect its distributions.
- Explore campaign outcomes across client and contact attributes.
- Prepare numeric and categorical columns with a `ColumnTransformer`.
- Compare ordinary K-fold and stratified K-fold validation.
- Compare baseline, L1-regularized, and L2-regularized logistic regression.
- Tune regularization strength with grid and randomized search using F1 score.
- Evaluate the selected pipeline with accuracy, precision, recall, F1, a classification report, and confusion matrix.
- Export one pipeline containing both preprocessing and classification.

## Dataset

The `Bank Marketing ML/` directory includes full and reduced variants of the UCI-style Bank Marketing data, together with their field descriptions. The notebook trains on `bank-full.csv`, which contains 45,211 client-contact records and a binary subscription target named `y`.

## Repository contents

| Path | Purpose |
| --- | --- |
| `Bank Marketing ML/Bank.ipynb` | Exploration, validation experiments, tuning, and final evaluation |
| `Bank Marketing ML/bank-full.csv` | Full dataset used by the notebook |
| `Bank Marketing ML/bank.csv` | Smaller dataset variant |
| `Bank Marketing ML/bank-additional*.csv` | Alternative dataset variants with economic indicators |
| `Bank Marketing ML/*-names.txt` | Dataset descriptions |
| `Bank Marketing ML/bank_marketing_logistic_pipeline.pkl` | Exported preprocessing and logistic-regression pipeline |

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install jupyter numpy pandas matplotlib scikit-learn joblib
jupyter lab "Bank Marketing ML/Bank.ipynb"
```

Run all cells in order. Because preprocessing is embedded in the exported pipeline, pass inference data with the original raw feature names and formats.
