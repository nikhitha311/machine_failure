"""Pydantic schemas for API request/response validation."""

from typing import List
from pydantic import BaseModel, Field


class SensorReading(BaseModel):
    """Single machine sensor reading for prediction."""
    type: str = Field(..., description="Machine type: L, M, or H")
    air_temperature: float = Field(..., ge=0, le=500)
    process_temperature: float = Field(..., ge=0, le=500)
    rotational_speed: float = Field(..., ge=0, le=10000)
    torque: float = Field(..., ge=0, le=500)
    tool_wear: float = Field(..., ge=0, le=1000)


class PredictionResponse(BaseModel):
    """Prediction result with explanation."""
    failure_probability: float
    prediction: int
    risk_level: str
    explanation: List[str]
    recommendation: str