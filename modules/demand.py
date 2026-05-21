import os
from typing import Any, Dict

from .base import BaseModule


class ProductDemandModule(BaseModule):
    name = "demand"
    version = "1.0"
    model_path = os.path.join("models", "demand_model.joblib")
    defaults_path = os.path.join("models", "demand_feature_defaults.json")
    resources_path = os.path.join("resources", "demand_resources.json")

    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        X = self._build_row(features)

        # Classification: probability of high demand
        proba = float(self._model.predict_proba(X)[:, 1][0])

        if proba < 0.33:
            level = "low"
        elif proba < 0.66:
            level = "medium"
        else:
            level = "high"

        return {
            "module": self.name,
            "score_label": "high_demand_probability",
            "score": round(proba, 4),
            "level": level,
            "recommended_resources": self.resources_for_level(level),
            "disclaimer": "Decision-support only. Forecasts are approximate and should be validated with business context."
        }
