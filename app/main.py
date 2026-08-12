"""
Predictive Maintenance API
Serves the trained XGBoost model as a REST API, plus a small static demo
page so a non-technical person can try it in a browser.

Run locally:
    uvicorn app.main:app --reload

Docs (auto-generated):
    http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import logging
import time

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app import config
from app.model import get_model
from app.schemas import (
    BatchMachineReadings,
    BatchPredictionResponse,
    MachineReading,
    ModelInfoResponse,
    PredictionResponse,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("predictive_maintenance")

app = FastAPI(
    title=config.API_TITLE,
    description=config.API_DESCRIPTION,
    version=config.API_VERSION,
)

# Open CORS since this is meant to be called from any frontend (including
# the static demo page and third-party clients). Tighten allow_origins to
# your specific domain(s) once you know them.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def load_model_on_startup() -> None:
    # Triggers the cached load once, so the first real request isn't slow
    # and so we fail fast (loud crash on boot) if artifacts are missing.
    get_model()


@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Process-Time-Ms"] = f"{(time.perf_counter() - start) * 1000:.2f}"
    return response


# Static demo UI
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", tags=["demo"], include_in_schema=False)
def serve_demo():
    return FileResponse("static/index.html")


# Health
@app.get("/health", tags=["system"])
def health():
    model = get_model()
    return {
        "status": "ok",
        "model_loaded": model.model is not None,
        "model_type": type(model.model).__name__,
    }


# Prediction endpoints
@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
def predict(reading: MachineReading):
    """Score a single machine reading."""
    model = get_model()
    try:
        return model.predict_one(reading)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


@app.post(
    "/predict/batch", response_model=BatchPredictionResponse, tags=["prediction"]
)
def predict_batch(payload: BatchMachineReadings):
    """Score up to 500 machine readings in one call."""
    model = get_model()
    try:
        results = model.predict_batch(payload.readings)
        return {"results": results}
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception("Batch prediction failed")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {e}")


# Model metadata
@app.get("/model/info", response_model=ModelInfoResponse, tags=["system"])
def model_info():
    model = get_model()
    metrics = {}
    all_models = {}
    if config.MODEL_RESULTS_PATH.exists():
        df = pd.read_csv(config.MODEL_RESULTS_PATH, index_col=0)
        all_models = df.round(4).to_dict(orient="index")
        best_name = type(model.model).__name__
        # model_results.csv is indexed by human-readable model names
        # (e.g. "XGBoost"), so match on substring against the class name.
        for name, row in all_models.items():
            if name.replace(" ", "").lower() in best_name.lower():
                metrics = row
                break
        if not metrics and all_models:
            # Fall back to the top-performing row by F1-Score
            best_name_csv = df["F1-Score"].idxmax()
            metrics = all_models[best_name_csv]

    return {
        "model_name": type(model.model).__name__,
        "features_used": model.feature_names,
        "metrics": metrics,
        "all_models_compared": all_models,
    }
