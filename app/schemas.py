from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ModelKey = Literal["rf", "log", "xgb"]


class HealthOut(BaseModel):
    ok: bool = True
    model_loaded: bool
    artifact: str


class ModelInfo(BaseModel):
    id: str
    name: str


class MetaOut(BaseModel):
    teams: list[str]
    models: list[ModelInfo]
    class_order: list[str]
    reference_order: list[str]


class PredictIn(BaseModel):
    home_team: str = Field(..., min_length=1, description="Home club name as in the dataset")
    away_team: str = Field(..., min_length=1)
    year: int = Field(..., ge=2015, le=2035)
    date: str = Field(..., description="ISO date YYYY-MM-DD")
    possession_home: int = Field(..., ge=0, le=100)
    possession_away: int = Field(..., ge=0, le=100)
    shots_home: int = Field(..., ge=0, le=100)
    shots_away: int = Field(..., ge=0, le=100)
    corners_home: int = Field(..., ge=0, le=30)
    corners_away: int = Field(..., ge=0, le=30)
    fouls_home: int = Field(..., ge=0, le=40)
    fouls_away: int = Field(..., ge=0, le=40)
    model: ModelKey = "rf"

    model_config = ConfigDict(extra="forbid")


class PredictOut(BaseModel):
    prediction: str
    prediction_display: str
    probabilities: dict[str, float]
    """Keys match label encoder class order."""
    class_order: list[str]
    model: str
