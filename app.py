import os
import json
from datetime import datetime, timezone
from typing import Any, Dict

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware

from modules.registry import get_registry
from modules.base import ModuleNotReadyError

try:
    from azure.storage.blob import BlobServiceClient
    from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError
except Exception:
    BlobServiceClient = None
    ResourceExistsError = Exception
    ResourceNotFoundError = Exception


class PredictRequest(BaseModel):
    features: Dict[str, Any] = Field(default_factory=dict)


app = FastAPI(title="SmartCloud Insights API", version="1.0")

allowed_origins = os.getenv("ALLOWED_ORIGINS", "*").split(",") if os.getenv("ALLOWED_ORIGINS") else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in allowed_origins],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

REGISTRY = get_registry()

AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
STUDENT_HISTORY_CONTAINER = os.getenv("STUDENT_HISTORY_CONTAINER", "student-history")
STUDENT_HISTORY_BLOB = os.getenv("STUDENT_HISTORY_BLOB", "student_prediction_history.jsonl")


def module_status():
    status = {}
    for name, mod in REGISTRY.items():
        ok, msg = mod.is_ready()
        status[name] = {"ready": ok, "message": msg}
    return status


def get_blob_client():
    if not AZURE_STORAGE_CONNECTION_STRING or BlobServiceClient is None:
        return None

    service = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
    container_client = service.get_container_client(STUDENT_HISTORY_CONTAINER)

    try:
        container_client.create_container()
    except ResourceExistsError:
        pass
    except Exception:
        pass

    return container_client.get_blob_client(STUDENT_HISTORY_BLOB)


def save_student_history(features: Dict[str, Any], result: Dict[str, Any]):
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "module": "student",
        "features": features,
        "prediction_result": result
    }

    line = json.dumps(record, default=str) + "\n"

    blob_client = get_blob_client()

    if blob_client:
        try:
            try:
                blob_client.create_append_blob()
            except ResourceExistsError:
                pass

            blob_client.append_block(line.encode("utf-8"))
            return {"saved": True, "storage": "Azure Blob Storage"}
        except Exception as e:
            print(f"Azure Blob save error: {e}")

    os.makedirs("logs", exist_ok=True)
    with open("logs/student_prediction_history.jsonl", "a", encoding="utf-8") as f:
        f.write(line)

    return {"saved": True, "storage": "Local fallback file"}


def read_student_history(limit: int = 20):
    data = ""

    blob_client = get_blob_client()

    if blob_client:
        try:
            data = blob_client.download_blob().readall().decode("utf-8")
        except Exception:
            data = ""

    if not data:
        try:
            with open("logs/student_prediction_history.jsonl", "r", encoding="utf-8") as f:
                data = f.read()
        except Exception:
            data = ""

    history = []
    for line in data.splitlines():
        try:
            history.append(json.loads(line))
        except Exception:
            pass

    return history[-limit:]


@app.get("/health")
def health():
    return {"status": "ok", "modules": module_status()}


@app.get("/modules")
def list_modules():
    return {"modules": list(REGISTRY.keys()), "status": module_status()}


@app.get("/student/history")
def student_history(limit: int = 20):
    return {
        "module": "student",
        "storage": "Azure Blob Storage" if AZURE_STORAGE_CONNECTION_STRING else "Local fallback file",
        "container": STUDENT_HISTORY_CONTAINER,
        "blob": STUDENT_HISTORY_BLOB,
        "history": read_student_history(limit)
    }


@app.get("/{module}/features")
def get_features(module: str):
    if module not in REGISTRY:
        raise HTTPException(status_code=404, detail=f"Unknown module: {module}")
    mod = REGISTRY[module]
    try:
        defaults = mod.defaults()
    except ModuleNotReadyError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return {
        "module": module,
        "feature_columns": list(defaults.keys()),
        "defaults": defaults
    }


@app.get("/{module}/resources")
def get_resources(module: str):
    if module not in REGISTRY:
        raise HTTPException(status_code=404, detail=f"Unknown module: {module}")
    mod = REGISTRY[module]
    try:
        mod.load()
        return {"module": module, "resources": mod._resources}
    except ModuleNotReadyError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/{module}/predict")
def predict(module: str, req: PredictRequest):
    if module not in REGISTRY:
        raise HTTPException(status_code=404, detail=f"Unknown module: {module}")

    mod = REGISTRY[module]
    try:
        mod.load()
        defaults = mod.defaults()
    except ModuleNotReadyError as e:
        raise HTTPException(status_code=503, detail=str(e))

    unknown = [k for k in req.features.keys() if k not in mod.feature_columns]
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unknown feature(s): {unknown}")

    # Merge default values with user input
    # This allows {"features": {}} to work using stored default student values
    full_features = defaults.copy()
    full_features.update(req.features)

    try:
        result = mod.predict(full_features)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    if module == "student":
        save_student_history(full_features, result)

    return result

@app.post("/predict")
def predict_legacy(req: PredictRequest):
    result = predict("student", req)

    return {
        "at_risk_probability": result["score"],
        "risk_level": result["level"],
        "recommended_resources": result["recommended_resources"],
        "disclaimer": result["disclaimer"]
    }
