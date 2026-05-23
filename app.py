import os
import json
from datetime import datetime, timezone
from typing import Any, Dict
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from modules.registry import get_registry
from modules.base import ModuleNotReadyError

try:
    from azure.storage.blob import BlobServiceClient
    from azure.core.exceptions import ResourceExistsError
except Exception:
    BlobServiceClient = None
    ResourceExistsError = Exception


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


def clean_json(data: Any):
    """
    Converts numpy/pandas/model output values into JSON-safe values.
    This prevents FastAPI from crashing with Internal Server Error.
    """
    try:
        if hasattr(data, "item"):
            return data.item()

        if isinstance(data, dict):
            return {str(k): clean_json(v) for k, v in data.items()}

        if isinstance(data, list):
            return [clean_json(v) for v in data]

        if isinstance(data, tuple):
            return [clean_json(v) for v in data]

        if pd.isna(data):
            return None

        return data
    except Exception:
        try:
            return jsonable_encoder(data)
        except Exception:
            return str(data)


def module_status():
    status = {}
    for name, mod in REGISTRY.items():
        ok, msg = mod.is_ready()
        status[name] = {"ready": ok, "message": msg}
    return status


def get_blob_client():
    try:
        if not AZURE_STORAGE_CONNECTION_STRING or BlobServiceClient is None:
            return None

        service = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
        container_client = service.get_container_client(STUDENT_HISTORY_CONTAINER)

        try:
            container_client.create_container()
        except ResourceExistsError:
            pass
        except Exception as e:
            print(f"Container create/check error: {e}")

        return container_client.get_blob_client(STUDENT_HISTORY_BLOB)

    except Exception as e:
        print(f"Blob client error: {e}")
        return None


def save_student_history(features: Dict[str, Any], result: Dict[str, Any]):
 
    try:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "module": "student",
            "features": clean_json(features),
            "prediction_result": clean_json(result)
        }

        line = json.dumps(record, default=str) + "\n"

        blob_client = get_blob_client()

        if blob_client:
            try:
                try:
                    old_data = blob_client.download_blob().readall().decode("utf-8")
                except Exception:
                    old_data = ""

                blob_client.upload_blob(old_data + line, overwrite=True)

                return {
                    "saved": True,
                    "storage": "Azure Blob Storage",
                    "container": STUDENT_HISTORY_CONTAINER,
                    "blob": STUDENT_HISTORY_BLOB
                }

            except Exception as e:
                print(f"Azure Blob save error: {e}")
                return {
                    "saved": False,
                    "storage": "Azure Blob Storage",
                    "error": str(e)
                }

        return {
            "saved": False,
            "storage": "Azure Blob Storage",
            "error": "Blob client not available. Check environment variables or azure-storage-blob package."
        }

    except Exception as e:
        print(f"History save failed: {e}")
        return {"saved": False, "error": str(e)}


def read_student_history(limit: int = 20):
    data = ""

    blob_client = get_blob_client()

    if blob_client:
        try:
            data = blob_client.download_blob().readall().decode("utf-8")
        except Exception as e:
            print(f"Azure Blob read error: {e}")
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
        "storage": "Azure Blob Storage" if AZURE_STORAGE_CONNECTION_STRING else "Not configured",
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
        "defaults": clean_json(defaults)
    }


@app.get("/{module}/resources")
def get_resources(module: str):
    if module not in REGISTRY:
        raise HTTPException(status_code=404, detail=f"Unknown module: {module}")

    mod = REGISTRY[module]

    try:
        mod.load()
        return {"module": module, "resources": clean_json(mod._resources)}
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
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Module load/defaults failed: {str(e)}")

    unknown = [k for k in req.features.keys() if k not in mod.feature_columns]
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unknown feature(s): {unknown}")

    full_features = defaults.copy()
    full_features.update(req.features)

    try:
        result = mod.predict(full_features)
        result = clean_json(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    history_status = None

    if module == "student":
        history_status = save_student_history(full_features, result)

    if isinstance(result, dict):
        result["history_storage"] = history_status

    return result


@app.post("/predict")
def predict_legacy(req: PredictRequest):
    result = predict("student", req)

    return {
        "at_risk_probability": result.get("score"),
        "risk_level": result.get("level"),
        "recommended_resources": result.get("recommended_resources"),
        "disclaimer": result.get("disclaimer"),
        "history_storage": result.get("history_storage")
    }
