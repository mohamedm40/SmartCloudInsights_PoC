from typing import Dict

from .student import StudentRiskModule
from .demand import ProductDemandModule
from .health import HealthTrendModule
from .base import BaseModule


def get_registry() -> Dict[str, BaseModule]:
    # Instances are lightweight; models are loaded lazily on first request.
    return {
        "student": StudentRiskModule(),
        "demand": ProductDemandModule(),
        "health": HealthTrendModule(),
    }
