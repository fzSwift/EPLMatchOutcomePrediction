"""API / UI catalog data (keep in sync with ml_core model keys)."""
from __future__ import annotations

from typing import Final, Literal

ModelId = Literal["rf", "log", "xgb"]

MODEL_IDS: Final[tuple[ModelId, ...]] = ("rf", "log", "xgb")

MODEL_CATALOG: Final[list[dict[str, str]]] = [
    {"id": "rf", "name": "Random Forest"},
    {"id": "log", "name": "Logistic Regression"},
    {"id": "xgb", "name": "XGBoost"},
]
