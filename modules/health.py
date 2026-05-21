import os
from typing import Any, Dict

from .base import BaseModule


class HealthTrendModule(BaseModule):
    name = "health"
    version = "1.0"
    model_path = os.path.join("models", "health_risk_model.joblib")
    defaults_path = os.path.join("models", "health_feature_defaults.json")
    resources_path = os.path.join("resources", "health_resources.json")

    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        X = self._build_row(features)
        proba = float(self._model.predict_proba(X)[:, 1][0])  # probability of higher risk

        if proba < 0.33:
            level = "low"
        elif proba < 0.66:
            level = "medium"
        else:
            level = "high"

        return {
            "module": self.name,
            "score_label": "health_risk_probability",
            "score": round(proba, 4),
            "level": level,
            "recommended_resources": self.resources_for_level(level),
            "disclaimer": "Decision-support only. This is not medical advice. Consult qualified healthcare professionals."
        }
