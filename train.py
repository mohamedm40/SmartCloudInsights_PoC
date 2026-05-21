import os
import json
import argparse
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score
)

MODELS_DIR = "models"
SCREENSHOTS_DIR = "screenshots"
DATA_DIR = "data"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)


def _warn(msg: str) -> None:
    print(f"[warn] {msg}")


def save_confusion(cm, labels, out_path, title):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print("Saved confusion matrix to:", out_path)


def train_student():
    data_path = os.path.join(DATA_DIR, "student.csv")
    print("Loading dataset:", data_path)
    df = pd.read_csv(data_path, sep=";")

    # Binary classification: at-risk if final grade G3 < 10
    df["at_risk"] = (df["G3"] < 10).astype(int)

    X = df.drop(columns=["G3", "at_risk"])
    y = df["at_risk"]

    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    pre = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])

    clf = LogisticRegression(max_iter=2000)
    pipe = Pipeline([("pre", pre), ("clf", clf)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training student model...")
    pipe.fit(X_train, y_train)

    proba = pipe.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    auc = float(roc_auc_score(y_test, proba))
    rep = classification_report(y_test, preds, output_dict=True)
    cm = confusion_matrix(y_test, preds)

    metrics = {
        "module": "student",
        "roc_auc": auc,
        "accuracy": rep["accuracy"],
        "f1_at_risk": rep["1"]["f1-score"],
        "precision_at_risk": rep["1"]["precision"],
        "recall_at_risk": rep["1"]["recall"],
        "support_test": int(len(y_test))
    }

    with open(os.path.join(MODELS_DIR, "student_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print("Saved metrics to:", os.path.join(MODELS_DIR, "student_metrics.json"))

    save_confusion(cm, ["Not Risk", "At Risk"],
                   os.path.join(SCREENSHOTS_DIR, "student_confusion_matrix.png"),
                   "Confusion Matrix - Student At-Risk Prediction")

    joblib.dump(pipe, os.path.join(MODELS_DIR, "student_risk_model.joblib"))
    print("Saved trained model to:", os.path.join(MODELS_DIR, "student_risk_model.joblib"))


def _synthetic_demand(n=2000, seed=7):
    rng = np.random.default_rng(seed)
    # Matches the feature set used by the REAL Bike Sharing dataset preparation
    df = pd.DataFrame({
        "season": rng.integers(1, 5, n),
        "holiday": rng.integers(0, 2, n),
        "workingday": rng.integers(0, 2, n),
        "weathersit": rng.integers(1, 5, n),
        "weekday": rng.integers(0, 7, n),
        "temp": rng.uniform(0.2, 0.9, n),
        "hum": rng.uniform(0.2, 0.95, n),
        "windspeed": rng.uniform(0.0, 0.8, n),
    })
    demand_score = (
        (df["temp"] * 2.0)
        - (df["hum"] * 0.8)
        - (df["windspeed"] * 0.5)
        + (df["workingday"] * 0.3)
        + rng.normal(0, 0.25, n)
    )
    df["high_demand"] = (demand_score > np.median(demand_score)).astype(int)
    return df


def train_demand():
    data_path = os.path.join(DATA_DIR, "demand.csv")
    if os.path.exists(data_path):
        print("Loading demand dataset:", data_path)
        df = pd.read_csv(data_path)
        if "high_demand" not in df.columns:
            # If user provides numeric demand, create label
            if "demand" in df.columns:
                df["high_demand"] = (df["demand"] > df["demand"].median()).astype(int)
            else:
                raise ValueError("Demand dataset must include 'high_demand' or 'demand' column.")
    else:
        # Try to auto-download/prepare the real dataset
        try:
            from download_datasets import prepare_demand_from_bike
            prepare_demand_from_bike()
        except Exception as e:
            _warn(f"Demand dataset not found and auto-download failed: {e}")

        if os.path.exists(data_path):
            print("Loading demand dataset:", data_path)
            df = pd.read_csv(data_path)
        else:
            print("Demand dataset still missing. Generating synthetic dataset for PoC:", data_path)
            df = _synthetic_demand()

    y = df["high_demand"].astype(int)
    X = df.drop(columns=["high_demand"])

    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    pre = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])

    clf = LogisticRegression(max_iter=2000)
    pipe = Pipeline([("pre", pre), ("clf", clf)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training demand model...")
    pipe.fit(X_train, y_train)

    proba = pipe.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    auc = float(roc_auc_score(y_test, proba))
    rep = classification_report(y_test, preds, output_dict=True)
    cm = confusion_matrix(y_test, preds)

    metrics = {
        "module": "demand",
        "roc_auc": auc,
        "accuracy": rep["accuracy"],
        "f1_high_demand": rep["1"]["f1-score"],
        "support_test": int(len(y_test))
    }
    with open(os.path.join(MODELS_DIR, "demand_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print("Saved metrics to:", os.path.join(MODELS_DIR, "demand_metrics.json"))

    save_confusion(cm, ["Low Demand", "High Demand"],
                   os.path.join(SCREENSHOTS_DIR, "demand_confusion_matrix.png"),
                   "Confusion Matrix - Product Demand (High vs Low)")

    joblib.dump(pipe, os.path.join(MODELS_DIR, "demand_model.joblib"))
    print("Saved trained model to:", os.path.join(MODELS_DIR, "demand_model.joblib"))


def _synthetic_health(n=2500, seed=11):
    rng = np.random.default_rng(seed)
    # Matches the feature set used by the REAL Early Stage Diabetes dataset preparation
    df = pd.DataFrame({
        "age": rng.integers(16, 70, n),
        "gender": rng.choice(["male", "female"], n),
        "polyuria": rng.choice(["yes", "no"], n, p=[0.25, 0.75]),
        "polydipsia": rng.choice(["yes", "no"], n, p=[0.25, 0.75]),
        "sudden_weight_loss": rng.choice(["yes", "no"], n, p=[0.20, 0.80]),
        "weakness": rng.choice(["yes", "no"], n, p=[0.30, 0.70]),
        "polyphagia": rng.choice(["yes", "no"], n, p=[0.15, 0.85]),
        "obesity": rng.choice(["yes", "no"], n, p=[0.20, 0.80]),
    })
    score = (
        (df["age"] / 60.0)
        + (df["polyuria"].eq("yes") * 1.2)
        + (df["polydipsia"].eq("yes") * 1.0)
        + (df["sudden_weight_loss"].eq("yes") * 0.8)
        + (df["obesity"].eq("yes") * 0.6)
        + rng.normal(0, 0.25, n)
    )
    df["high_risk"] = (score > np.median(score)).astype(int)
    return df


def train_health():
    data_path = os.path.join(DATA_DIR, "health.csv")
    if os.path.exists(data_path):
        print("Loading health dataset:", data_path)
        df = pd.read_csv(data_path)
        # Accept common label aliases
        if "high_risk" not in df.columns:
            if "diabetes_positive" in df.columns:
                df["high_risk"] = df["diabetes_positive"].astype(int)
                df.drop(columns=["diabetes_positive"], inplace=True)
            elif "risk" in df.columns:
                df["high_risk"] = (df["risk"] > df["risk"].median()).astype(int)
            else:
                raise ValueError("Health dataset must include 'high_risk' (or 'diabetes_positive'/'risk') column.")
    else:
        # Try to auto-download/prepare the real dataset
        try:
            from download_datasets import prepare_health_from_diabetes
            prepare_health_from_diabetes()
        except Exception as e:
            _warn(f"Health dataset not found and auto-download failed: {e}")

        if os.path.exists(data_path):
            print("Loading health dataset:", data_path)
            df = pd.read_csv(data_path)
            if "high_risk" not in df.columns and "diabetes_positive" in df.columns:
                df["high_risk"] = df["diabetes_positive"].astype(int)
                df.drop(columns=["diabetes_positive"], inplace=True)
        else:
            print("Health dataset still missing. Generating synthetic dataset for PoC:", data_path)
            df = _synthetic_health()

    y = df["high_risk"].astype(int)
    X = df.drop(columns=["high_risk"])

    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    pre = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])

    clf = LogisticRegression(max_iter=2000)
    pipe = Pipeline([("pre", pre), ("clf", clf)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training health model...")
    pipe.fit(X_train, y_train)

    proba = pipe.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    auc = float(roc_auc_score(y_test, proba))
    rep = classification_report(y_test, preds, output_dict=True)
    cm = confusion_matrix(y_test, preds)

    metrics = {
        "module": "health",
        "roc_auc": auc,
        "accuracy": rep["accuracy"],
        "f1_high_risk": rep["1"]["f1-score"],
        "support_test": int(len(y_test))
    }
    with open(os.path.join(MODELS_DIR, "health_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print("Saved metrics to:", os.path.join(MODELS_DIR, "health_metrics.json"))

    save_confusion(cm, ["Lower Risk", "Higher Risk"],
                   os.path.join(SCREENSHOTS_DIR, "health_confusion_matrix.png"),
                   "Confusion Matrix - Health Risk (Higher vs Lower)")

    joblib.dump(pipe, os.path.join(MODELS_DIR, "health_risk_model.joblib"))
    print("Saved trained model to:", os.path.join(MODELS_DIR, "health_risk_model.joblib"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", choices=["student", "demand", "health", "all"], default="student")
    args = parser.parse_args()

    if args.module in ("student", "all"):
        train_student()

    if args.module in ("demand", "all"):
        train_demand()

    if args.module in ("health", "all"):
        train_health()


if __name__ == "__main__":
    main()
