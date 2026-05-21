import os
import json
from typing import Any, Dict, List, Optional, Tuple

import joblib
import pandas as pd


class ModuleNotReadyError(RuntimeError):
    """Raised when a module model/artefacts are missing."""


class BaseModule:
    """Base interface for SmartCloud Insights prediction modules."""

    name: str = "base"
    version: str = "0.1"
    model_path: str = ""
    defaults_path: str = ""
    resources_path: str = ""
    feature_columns: List[str] = []

    def __init__(self) -> None:
        self._model = None
        self._defaults: Dict[str, Any] = {}
        self._resources: Dict[str, Any] = {}

    def is_ready(self) -> Tuple[bool, str]:
        if self.model_path and not os.path.exists(self.model_path):
            return False, f"Missing model: {self.model_path}"
        if self.defaults_path and not os.path.exists(self.defaults_path):
            return False, f"Missing defaults: {self.defaults_path}"
        if self.resources_path and not os.path.exists(self.resources_path):
            return False, f"Missing resources: {self.resources_path}"
        return True, "ok"

    def load(self) -> None:
        ok, msg = self.is_ready()
        if not ok:
            raise ModuleNotReadyError(msg)

        if self._model is None and self.model_path:
            self._model = joblib.load(self.model_path)

        if not self._defaults and self.defaults_path:
            with open(self.defaults_path, "r", encoding="utf-8") as f:
                self._defaults = json.load(f)
            self.feature_columns = list(self._defaults.keys())

        if not self._resources and self.resources_path:
            with open(self.resources_path, "r", encoding="utf-8") as f:
                self._resources = json.load(f)

    def defaults(self) -> Dict[str, Any]:
        self.load()
        return dict(self._defaults)

    def resources_for_level(self, level: str) -> List[Dict[str, str]]:
        self.load()
        return list(self._resources.get(level, []))

    def _build_row(self, features: Dict[str, Any]) -> pd.DataFrame:
        self.load()
        row = self._defaults.copy()
        row.update(features)
        X = pd.DataFrame([row], columns=self.feature_columns)
        return X

    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Return a dict response. Must be implemented by subclasses."""
        raise NotImplementedError
