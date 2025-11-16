import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score
import shap
from lime.lime_tabular import LimeTabularExplainer

def load_data():
    # Placeholder – user downloads Kaggle dataset
    df = pd.read_csv("credit_train.csv")
    df = df.dropna()
    X = df.drop("SeriousDlqin2yrs", axis=1)
    y = df["SeriousDlqin2yrs"]
    return X, y

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBClassifier(
        max_depth=5,
        learning_rate=0.05,
        n_estimators=300,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss"
    )
    model.fit(X_train, y_train)

    preds = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, preds)
    f1 = f1_score(y_test, preds > 0.5)

    return model, X_train, X_test, y_test, auc, f1

def shap_global(model, X_train):
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_train)
    shap.summary_plot(shap_vals, X_train, show=False)
    return explainer, shap_vals

def lime_local(model, X_train, X_test):
    explainer = LimeTabularExplainer(
        X_train.values,
        feature_names=X_train.columns,
        class_names=["0", "1"],
        discretize_continuous=True
    )
    sample_ids = X_test.sample(5, random_state=42).index.tolist()
    explanations = {}
    for idx in sample_ids:
        exp = explainer.explain_instance(
            X_test.loc[idx].values,
            model.predict_proba
        )
        explanations[idx] = exp.as_list()
    return explanations

def main():
    X, y = load_data()
    model, X_train, X_test, y_test, auc, f1 = train_model(X, y)
    print("AUC:", auc)
    print("F1:", f1)

    shap_explainer, shap_vals = shap_global(model, X_train)
    lime_explanations = lime_local(model, X_train, X_test)

    print("Generated SHAP & LIME explanations successfully.")

if __name__ == "__main__":
    main()
