
# GCC Warehousing & Fulfillment – Insights Dashboard

This Streamlit dashboard visualises survey data, runs machine‑learning analyses, and uncovers actionable insights for an AI‑driven, on‑demand warehousing and fulfilment platform in the GCC.

## Features
1. **Data Visualisation** – 10+ interactive charts & KPIs.
2. **Classification** – KNN, Decision Tree, Random Forest, GBRT with metrics, confusion matrices, multi‑model ROC curves, and prediction upload/download.
3. **Clustering** – K‑Means elbow chart, adjustable cluster count, persona table, download clustered data.
4. **Association Rule Mining** – Apriori with user‑set parameters, top‑10 rules.
5. **Regression** – Linear, Ridge, Lasso, Decision‑Tree regressors with performance metrics.

## Quick Start
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Deploy on Streamlit Cloud
1. Push all files in this repo to GitHub.
2. In Streamlit Cloud, create a new app linked to the repo. **Main file**: `streamlit_app.py`.
3. Launch! The default dataset (`synthetic_gcc_warehousing_survey.csv`) is bundled, but you can upload updated data anytime.

## Data
The bundled CSV contains 1,000 synthetic survey responses reflecting SME logistics behaviour in the GCC. Multi‑select columns are comma‑delimited for association rules.

## License
MIT
