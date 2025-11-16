# Advanced Feature Attribution Analysis using SHAP & LIME

This repository contains an end‑to‑end implementation for training an advanced ML model (XGBoost), performing SHAP global & local explanations, and comparing them with LIME.

## 📌 Project Structure
```
shap_lime_project/
│── README.md
│── main.py
│── requirements.txt
│── reports/
│     └── explanation_report.md
```

## 🚀 What This Project Does
- Trains an XGBoost classifier on a chosen Kaggle dataset.
- Computes SHAP global importance + summary plots.
- Generates SHAP force & waterfall plots for selected instances.
- Generates LIME explanations for the same 5 instances.
- Compares SHAP vs LIME consistency + discrepancies.
- Provides best‑practice guidelines for regulated‑industry ML interpretation.

## 🧩 Dataset Recommendation
Use **“Give Me Some Credit”** from Kaggle.

Dataset Name (exact):  
`Give Me Some Credit`  
https://www.kaggle.com/datasets/c/GiveMeSomeCredit

Reason: Tabular, imbalance, risk‑based classification, perfect for SHAP attribution.

## 🛠 Installation
```
pip install -r requirements.txt
```

## ▶️ Run
```
python main.py
```
