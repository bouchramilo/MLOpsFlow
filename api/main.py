# main.py
import os
import time
import logging
from contextlib import asynccontextmanager
from typing import List

import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram, Gauge

from api.schemas import (
    DiabetesInput,
    DiabetesBatchInput,
    PredictionResponse,
    BatchPredictionResponse,
    HealthResponse
)
from api.model_loader import model_loader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

PREDICTION_COUNT = Counter(
    'prediction_requests_total',
    'Total number of prediction requests',
    ['endpoint', 'status']
)
PREDICTION_LATENCY = Histogram(
    'prediction_latency_seconds',
    'Prediction request latency in seconds',
    ['endpoint']
)
INFERENCE_TIME = Histogram(
    'model_inference_time_seconds',
    'Model inference time in seconds'
)
MODEL_LOADED = Gauge(
    'model_loaded',
    'Whether the ML model is loaded (1) or not (0)'
)


@asynccontextmanager
async def lifespan(app: FastAPI):

    logger.info("Starting up the API...")
    success, message = model_loader.load_model()
    
    if success:
        logger.info(f"Model loaded successfully: {message}")
        MODEL_LOADED.set(1)
    else:
        logger.warning(f"Model loading failed: {message}")
        logger.warning("API will start but predictions will not be available until model is loaded.")
        MODEL_LOADED.set(0)
    
    yield
    
    logger.info("Shutting down the API...")
    MODEL_LOADED.set(0)


app = FastAPI(
    title="Diabetes Prediction API",
    version="1.0.0",
    lifespan=lifespan
)

instrumentator = Instrumentator(
    should_group_status_codes=False,
    should_ignore_untemplated=True,
    should_respect_env_var=True,
    should_instrument_requests_inprogress=True,
    excluded_handlers=["/metrics"],
    inprogress_name="http_requests_inprogress",
    inprogress_labels=True
)
instrumentator.instrument(app).expose(app)


def prepare_input_data(input_data: DiabetesInput) -> pd.DataFrame:

    return pd.DataFrame([{
        "Pregnancies": input_data.Pregnancies,
        "Glucose": input_data.Glucose,
        "BloodPressure": input_data.BloodPressure,
        "SkinThickness": input_data.SkinThickness,
        "Insulin": input_data.Insulin,
        "BMI": input_data.BMI,
        "DiabetesPedigreeFunction": input_data.DiabetesPedigreeFunction,
        "Age": input_data.Age
    }])


def prepare_batch_input_data(batch_input: DiabetesBatchInput) -> pd.DataFrame:

    data = []
    for instance in batch_input.instances:
        data.append({
            "Pregnancies": instance.Pregnancies,
            "Glucose": instance.Glucose,
            "BloodPressure": instance.BloodPressure,
            "SkinThickness": instance.SkinThickness,
            "Insulin": instance.Insulin,
            "BMI": instance.BMI,
            "DiabetesPedigreeFunction": instance.DiabetesPedigreeFunction,
            "Age": instance.Age
        })
    return pd.DataFrame(data)


@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "Welcome to the Diabetes Prediction API",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():

    model_info = model_loader.get_model_info()
    
    return HealthResponse(
        status="healthy" if model_loader.is_loaded() else "degraded",
        model_loaded=model_loader.is_loaded(),
        model_version=model_info.get("model_version")
    )


@app.get("/model/info", tags=["Model"])
async def get_model_info():
    return model_loader.get_model_info()


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(input_data: DiabetesInput):
    start_time = time.time()
    
    if not model_loader.is_loaded():
        PREDICTION_COUNT.labels(endpoint="/predict", status="error").inc()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please try again later."
        )
    
    try:
        df = prepare_input_data(input_data)
        
        inference_start = time.time()
        prediction = model_loader.predict(df)
        probabilities = model_loader.predict_proba(df)
        inference_time = time.time() - inference_start
        
        INFERENCE_TIME.observe(inference_time)
        PREDICTION_COUNT.labels(endpoint="/predict", status="success").inc()
        
        response = PredictionResponse(
            prediction=int(prediction[0]),
            probability=probabilities[0].tolist() if probabilities is not None else None,
            status="success"
        )
        
        total_time = time.time() - start_time
        PREDICTION_LATENCY.labels(endpoint="/predict").observe(total_time)
        
        logger.info(f"Prediction made: {response.prediction} (inference time: {inference_time:.4f}s)")
        
        return response
        
    except Exception as e:
        PREDICTION_COUNT.labels(endpoint="/predict", status="error").inc()
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(batch_input: DiabetesBatchInput):
    start_time = time.time()
    
    if not model_loader.is_loaded():
        PREDICTION_COUNT.labels(endpoint="/predict/batch", status="error").inc()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please try again later."
        )
    
    try:
        df = prepare_batch_input_data(batch_input)
        
        inference_start = time.time()
        predictions = model_loader.predict(df)
        probabilities = model_loader.predict_proba(df)
        inference_time = time.time() - inference_start
        
        INFERENCE_TIME.observe(inference_time)
        PREDICTION_COUNT.labels(endpoint="/predict/batch", status="success").inc()
        
        response = BatchPredictionResponse(
            predictions=[int(p) for p in predictions],
            probabilities=[p.tolist() for p in probabilities] if probabilities is not None else None,
            count=len(predictions),
            status="success"
        )
        
        total_time = time.time() - start_time
        PREDICTION_LATENCY.labels(endpoint="/predict/batch").observe(total_time)
        
        logger.info(f"Batch prediction made for {len(predictions)} instances (inference time: {inference_time:.4f}s)")
        
        return response
        
    except Exception as e:
        PREDICTION_COUNT.labels(endpoint="/predict/batch", status="error").inc()
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


@app.post("/model/reload", tags=["Model"])
async def reload_model():
    success, message = model_loader.load_model()
    
    if success:
        MODEL_LOADED.set(1)
        return {"status": "success", "message": message}
    else:
        MODEL_LOADED.set(0)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=message
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )