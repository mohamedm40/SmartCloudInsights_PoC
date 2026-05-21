"""Download and prepare REAL public datasets for the SmartCloudInsights PoC.

This script downloads:
  - UCI Bike Sharing Dataset (id=275) -> prepares data/demand.csv
  - UCI Early Stage Diabetes Risk Prediction (id=529) -> prepares data/health.csv

It does NOT download the Student Performance dataset, because many students already
have it locally and it is semicolon-separated.

Run:
  python download_datasets.py
"""

from __future__ import annotations

import csv
import os
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd


DATA_DIR = Path("data")
DOWNLOAD_DIR = DATA_DIR / "_downloads"


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    url: str
    zip_filename: str


BIKE = DatasetSpec(
    name="bike_sharing",
    # UCI static download (dataset id 275)
    url="https://archive.ics.uci.edu/static/public/275/bike%2Bsharing%2Bdataset.zip",
    zip_filename="bike_sharing.zip",
)

DIABETES = DatasetSpec(
    name="early_stage_diabetes",
    # UCI static download (dataset id 529)
    url="https://archive.ics.uci.edu/static/public/529/early%2Bstage%2Bdiabetes%2Brisk%2Bprediction%2Bdataset.zip",
    zip_filename="early_stage_diabetes.zip",
)


def _download_zip(spec: DatasetSpec) -> Path:
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DOWNLOAD_DIR / spec.zip_filename
    if zip_path.exists() and zip_path.stat().st_size > 0:
        print(f"[skip] {spec.name}: already downloaded -> {zip_path}")
        return zip_path

    print(f"[download] {spec.name}: {spec.url}")
    urlretrieve(spec.url, zip_path)
    print(f"[ok] saved -> {zip_path}")
    return zip_path


def _extract_zip(zip_path: Path, target_dir: Path) -> None:
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(target_dir)


def _find_file(root: Path, filename: str) -> Path:
    for p in root.rglob(filename):
        return p
    raise FileNotFoundError(f"Could not find {filename} under {root}")


def prepare_demand_from_bike() -> None:
    """Prepare data/demand.csv from the UCI Bike Sharing *day.csv* file."""
    zip_path = _download_zip(BIKE)
    extract_dir = DOWNLOAD_DIR / "bike_sharing"
    _extract_zip(zip_path, extract_dir)

    day_csv = _find_file(extract_dir, "day.csv")
    # Keep a copy of the raw day.csv for provenance
    raw_copy = DATA_DIR / "bike_day.csv"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(day_csv, raw_copy)

    df = pd.read_csv(day_csv)
    # Use stable, interpretable features available in the dataset
    keep = [
        "season",
        "holiday",
        "workingday",
        "weathersit",
        "weekday",
        "temp",
        "hum",
        "windspeed",
        "cnt",
    ]
    df = df[keep].copy()

    # Define a simple classification target: "high demand" based on median daily rentals
    threshold = float(df["cnt"].median())
    df["high_demand"] = (df["cnt"] >= threshold).astype(int)
    df.drop(columns=["cnt"], inplace=True)

    out = DATA_DIR / "demand.csv"
    df.to_csv(out, index=False)
    print(f"[ok] demand dataset prepared -> {out} (rows={len(df)})")


def prepare_health_from_diabetes() -> None:
    """Prepare data/health.csv from the UCI Early Stage Diabetes dataset."""
    zip_path = _download_zip(DIABETES)
    extract_dir = DOWNLOAD_DIR / "early_stage_diabetes"
    _extract_zip(zip_path, extract_dir)

    # Many copies use this exact filename
    src_csv = _find_file(extract_dir, "diabetes_data_upload.csv")
    raw_copy = DATA_DIR / "diabetes_data_upload.csv"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src_csv, raw_copy)

    df = pd.read_csv(src_csv)

    # Normalise to lowercase strings for consistent UI inputs
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = df[c].astype(str).str.strip().str.lower()

    # Select a small, explainable subset of features (keeps the UI simple)
    features = [
        "age",
        "gender",
        "polyuria",
        "polydipsia",
        "sudden_weight_loss",
        "weakness",
        "polyphagia",
        "obesity",
    ]

    # Target column is typically named 'class'
    if "class" not in df.columns:
        raise ValueError("Expected 'class' column in diabetes dataset")

    out_df = df[features + ["class"]].copy()
    # Keep label name aligned with the rest of the PoC ("high_risk"),
    # while still representing diabetes-positive cases.
    out_df.rename(columns={"class": "high_risk"}, inplace=True)
    out_df["high_risk"] = (out_df["high_risk"] == "positive").astype(int)

    out = DATA_DIR / "health.csv"
    out_df.to_csv(out, index=False)
    print(f"[ok] health dataset prepared -> {out} (rows={len(out_df)})")


def main() -> None:
    print("Preparing real datasets...")
    try:
        prepare_demand_from_bike()
    except Exception as e:
        print("[warn] Could not prepare demand dataset:", e)
    try:
        prepare_health_from_diabetes()
    except Exception as e:
        print("[warn] Could not prepare health dataset:", e)

    print("Done.")


if __name__ == "__main__":
    main()
