import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from api.model_loader import ModelLoader
from api.monitoring_logger import log_inference
from api.preprocessing import clean_prediction_data
from api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    PredictionResponse,
    VideoInput,
)

model_loader = ModelLoader()
try:
    model_loader.load()
except Exception as e:
    print(f"[WARN] Immediate model loading deferred: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[INFO] Starting YouTube Prediction API...")
    if not model_loader.is_ready():
        model_loader.load()
    yield
    print("[INFO] Shutting down YouTube Prediction API...")


app = FastAPI(
    title="YouTube Trending Video Views Prediction API",
    description="Lab 10 - ML Model Serving and API Engineering for YouTube Video Views",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def root():
    return {
        "service": "YouTube Prediction API",
        "status": "online",
        "version": "1.0.0",
        "model": model_loader.model_name or "YouTube_Views_Production_Model",
        "docs_url": "/docs",
    }


@app.get("/health")
def health_check():
    is_ready = model_loader.is_ready()
    return {
        "status": "healthy" if is_ready else "unhealthy",
        "model_loaded": model_loader.model is not None,
        "preprocessor_loaded": model_loader.preprocessor is not None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/metadata")
def get_metadata():
    if not model_loader.is_ready():
        raise HTTPException(status_code=503, detail="Model is not ready.")
    return {
        "model_name": model_loader.model_name,
        "model_version": model_loader.model_version,
        "target": "views",
        "target_transformation": "log1p(views)",
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: VideoInput):
    if not model_loader.is_ready():
        raise HTTPException(status_code=503, detail="Model not loaded.")

    start_time = time.perf_counter()
    req_id = uuid.uuid4().hex[:12]

    df = pd.DataFrame([payload.model_dump()])
    X_clean = clean_prediction_data(df)
    X_trans = model_loader.preprocessor.transform(X_clean)

    pred_log = float(model_loader.model.predict(X_trans)[0])
    pred_views = int(round(max(float(np.expm1(pred_log)), 0.0)))

    latency_ms = (time.perf_counter() - start_time) * 1000
    log_inference(req_id, payload.model_dump(), pred_views, pred_log, latency_ms)

    return PredictionResponse(
        predicted_views=pred_views,
        predicted_log_views=round(pred_log, 4),
        model_version=str(model_loader.model_version or "1"),
        request_id=req_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(payload: BatchPredictionRequest):
    if not model_loader.is_ready():
        raise HTTPException(status_code=503, detail="Model not loaded.")

    start_time = time.perf_counter()
    records = [v.model_dump() for v in payload.videos]
    df = pd.DataFrame(records)
    X_clean = clean_prediction_data(df)
    X_trans = model_loader.preprocessor.transform(X_clean)

    preds_log = model_loader.model.predict(X_trans)
    preds_views = np.maximum(np.expm1(preds_log), 0).round(0).astype(int)

    responses = []
    for i, (pred_l, pred_v) in enumerate(zip(preds_log, preds_views)):
        req_id = uuid.uuid4().hex[:12]
        responses.append(
            PredictionResponse(
                predicted_views=int(pred_v),
                predicted_log_views=round(float(pred_l), 4),
                model_version=str(model_loader.model_version or "1"),
                request_id=req_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
        )

    proc_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
    return BatchPredictionResponse(
        predictions=responses,
        total_records=len(responses),
        processing_time_ms=proc_time_ms,
    )
