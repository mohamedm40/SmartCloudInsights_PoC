import os
import json
import argparse
import pandas as pd
import numpy as np


def save_json(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    print("Saved defaults to:", path)
    print("Total columns:", len(obj))


def defaults_from_dataframe(df: pd.DataFrame) -> dict:
    defaults = {}
    for col in df.columns:
        if df[col].dtype == "object":
            defaults[col] = df[col].mode(dropna=True).iloc[0]
        else:
            s = pd.to_numeric(df[col], errors="coerce")
            defaults[col] = float(s.median())
    return defaults


def build_student_defaults(data_path: str) -> dict:
    df = pd.read_csv(data_path, sep=";")
    X = df.drop(columns=["G3"], errors="ignore")
    return defaults_from_dataframe(X)


def build_demand_defaults(data_path: str) -> dict:
    if not os.path.exists(data_path):
        # Try to prepare real dataset automatically
        try:
            from download_datasets import prepare_demand_from_bike
            prepare_demand_from_bike()
        except Exception:
            pass

    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        # Bike Sharing feature set
        cols = [
            c for c in [
                "season",
                "holiday",
                "workingday",
                "weathersit",
                "weekday",
                "temp",
                "hum",
                "windspeed",
            ]
            if c in df.columns
        ]
        X = df[cols].copy()
        return defaults_from_dataframe(X)

    # Fallback defaults (synthetic)
    return {
        "season": 2,
        "holiday": 0,
        "workingday": 1,
        "weathersit": 1,
        "weekday": 2,
        "temp": 0.5,
        "hum": 0.5,
        "windspeed": 0.2,
    }


def build_health_defaults(data_path: str) -> dict:
    if not os.path.exists(data_path):
        try:
            from download_datasets import prepare_health_from_diabetes
            prepare_health_from_diabetes()
        except Exception:
            pass

    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        cols = [
            c for c in [
                "age",
                "gender",
                "polyuria",
                "polydipsia",
                "sudden_weight_loss",
                "weakness",
                "polyphagia",
                "obesity",
            ]
            if c in df.columns
        ]
        X = df[cols].copy()
        return defaults_from_dataframe(X)

    return {
        "age": 30,
        "gender": "male",
        "polyuria": "no",
        "polydipsia": "no",
        "sudden_weight_loss": "no",
        "weakness": "no",
        "polyphagia": "no",
        "obesity": "no",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", choices=["student", "demand", "health", "all"], default="student")
    args = parser.parse_args()

    os.makedirs("models", exist_ok=True)

    if args.module in ("student", "all"):
        student_path = os.path.join("data", "student.csv")
        defaults = build_student_defaults(student_path)
        save_json(os.path.join("models", "student_feature_defaults.json"), defaults)
        # legacy name for backward compatibility with older app/web
        save_json(os.path.join("models", "feature_defaults.json"), defaults)

    if args.module in ("demand", "all"):
        demand_path = os.path.join("data", "demand.csv")
        defaults = build_demand_defaults(demand_path)
        save_json(os.path.join("models", "demand_feature_defaults.json"), defaults)

    if args.module in ("health", "all"):
        health_path = os.path.join("data", "health.csv")
        defaults = build_health_defaults(health_path)
        save_json(os.path.join("models", "health_feature_defaults.json"), defaults)


if __name__ == "__main__":
    main()
