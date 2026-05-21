import os
from typing import Any, Dict

from .base import BaseModule


class StudentRiskModule(BaseModule):
    name = "student"
    version = "1.0"
    model_path = os.path.join("models", "student_risk_model.joblib")
    defaults_path = os.path.join("models", "student_feature_defaults.json")
    resources_path = os.path.join("resources", "student_resources.json")

    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        X = self._build_row(features)
        proba = float(self._model.predict_proba(X)[:, 1][0])  # at-risk probability

        if proba < 0.33:
            level = "low"
        elif proba < 0.66:
            level = "medium"
        else:
            level = "high"

        return {
            "module": self.name,
            "score_label": "at_risk_probability",
            "score": round(proba, 4),
            "level": level,
            "recommended_resources": self.resources_for_level(level),
            "disclaimer": "Decision-support only. Predictions may be imperfect and must not be used for automated decisions."
        }
