import os
from typing import Any, Dict

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware

from modules.registry import get_registry
from modules.base import ModuleNotReadyError


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


def module_status():
    status = {}
    for name, mod in REGISTRY.items():
        ok, msg = mod.is_ready()
        status[name] = {"ready": ok, "message": msg}
    return status


@app.get("/health")
def health():
    return {"status": "ok", "modules": module_status()}


@app.get("/modules")
def list_modules():
    return {"modules": list(REGISTRY.keys()), "status": module_status()}


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
        # Return full resource catalogue (low/medium/high)
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
    except ModuleNotReadyError as e:
        raise HTTPException(status_code=503, detail=str(e))

    unknown = [k for k in req.features.keys() if k not in mod.feature_columns]
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unknown feature(s): {unknown}")

    result = mod.predict(req.features)
    return result


# Backward compatible student endpoint:
@app.post("/predict")
def predict_legacy(req: PredictRequest):
    result = predict("student", req)
    # Map to old response keys for the existing student UI
    return {
        "at_risk_probability": result["score"],
        "risk_level": result["level"],
        "recommended_resources": result["recommended_resources"],
        "disclaimer": result["disclaimer"]
    }
