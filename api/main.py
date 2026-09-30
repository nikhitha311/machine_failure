 
"""
FastAPI endpoint for machine failure prediction.
OptiForge 2026 · Team code crafters (OPT-26-3407)
"""

from __future__ import annotations

import os
import time
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from api.schemas import SensorReading, PredictionResponse
from api.model_loader import get_model
from src.predict import predict_one


app = FastAPI(
    title="Robust Machine Failure Prediction API",
    description="OptiForge 2026 - Team code crafters (OPT-26-3407)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

if os.path.exists("web"):
    app.mount("/static", StaticFiles(directory="web"), name="static")


@app.get("/")
def root():
    return {
        "status": "healthy",
        "service": "machine-failure-prediction",
        "team": "code crafters",
        "team_id": "OPT-26-3407",
        "endpoints": ["/predict", "/health", "/docs", "/static/index.html"],
    }


@app.get("/health")
def health():
    return {"status": "ok", "timestamp": time.time()}


@app.post("/predict", response_model=PredictionResponse)
def predict(reading: SensorReading):
    model, feature_names = get_model()

    sample_df = pd.DataFrame([{
        "Type": {"L": 0, "M": 1, "H": 2}.get(reading.type.upper(), 0),
        "Air temperature [K]": reading.air_temperature,
        "Process temperature [K]": reading.process_temperature,
        "Rotational speed [rpm]": reading.rotational_speed,
        "Torque [Nm]": reading.torque,
        "Tool wear [min]": reading.tool_wear,
    }])

    if feature_names:
        sample_df = sample_df.reindex(columns=feature_names, fill_value=0)

    sensor_cols = {
        "air_temp": "Air temperature [K]",
        "process_temp": "Process temperature [K]",
        "rpm": "Rotational speed [rpm]",
        "torque": "Torque [Nm]",
        "tool_wear": "Tool wear [min]",
    }

    result = predict_one(
        model=model,
        sample_df=sample_df,
        feature_names=feature_names or list(sample_df.columns),
        sensor_cols=sensor_cols,
        top_k=5,
    )

    return PredictionResponse(
        failure_probability=round(result["probability"], 4),
        prediction=result["prediction"],
        risk_level=result["risk_level"],
        explanation=result["explanation"],
        recommendation=result["recommendation"],
    )


@app.get("/predict-demo")
def predict_demo():
    return predict(SensorReading(
        type="M",
        air_temperature=298.1,
        process_temperature=308.6,
        rotational_speed=1551,
        torque=42.8,
        tool_wear=0,
    ))


@app.get("/ui")
def ui():
    return FileResponse("web/index.html")