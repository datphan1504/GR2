"""FastAPI route handlers for Website Fingerprinting inference."""

import numpy as np
from fastapi import APIRouter, HTTPException

from app.api.schemas import (
    TraceInput, PredictionResult, BatchTraceInput,
    BatchPredictionResult, HealthResponse
)

router = APIRouter(prefix="/api/v1", tags=["Inference"])

# State placeholder attached on application startup
app_state = {
    "model": None,
    "config": None,
    "model_loaded": False
}


@router.get("/health", response_model=HealthResponse)
def health_check():
    """Returns the API service health and model status."""
    return HealthResponse(
        status="ok",
        model_name=app_state.get("config", {}).get("model_name", "var-cnn") if app_state.get("config") else "none",
        model_loaded=app_state.get("model_loaded", False)
    )


@router.post("/predict", response_model=PredictionResult)
def predict_trace(payload: TraceInput):
    """Classifies a single traffic trace into a monitored website or unmonitored."""
    model = app_state.get("model")
    config = app_state.get("config")

    if model is None or config is None:
        raise HTTPException(status_code=503, detail="Model is not loaded or weights not found.")

    seq_length = config.get("seq_length", 5000)
    num_mon_sites = config.get("num_mon_sites", 100)

    # Prepare dir input
    dir_arr = np.zeros((1, seq_length, 1), dtype=np.float32)
    dir_input_list = payload.dir_seq[:seq_length]
    dir_arr[0, :len(dir_input_list), 0] = dir_input_list

    # Prepare inputs dictionary
    inputs = {'dir_input': dir_arr}

    if payload.time_seq is not None:
        time_arr = np.zeros((1, seq_length, 1), dtype=np.float32)
        time_input_list = payload.time_seq[:seq_length]
        time_arr[0, :len(time_input_list), 0] = time_input_list
        inputs['time_input'] = time_arr

    if payload.metadata is not None:
        meta_arr = np.zeros((1, 7), dtype=np.float32)
        meta_len = min(len(payload.metadata), 7)
        meta_arr[0, :meta_len] = payload.metadata[:meta_len]
        inputs['metadata_input'] = meta_arr
    elif 'metadata_input' in [inp.name.split(':')[0] for inp in model.inputs]:
        inputs['metadata_input'] = np.zeros((1, 7), dtype=np.float32)

    # Perform inference
    try:
        preds = model.predict(inputs)
        softmax = preds[0]
        predicted_class = int(np.argmax(softmax))
        confidence = float(softmax[predicted_class])
        is_mon = (predicted_class < num_mon_sites)

        return PredictionResult(
            predicted_site_id=predicted_class,
            confidence=confidence,
            is_monitored=is_mon,
            probabilities=softmax.tolist()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@router.post("/predict/batch", response_model=BatchPredictionResult)
def predict_batch_traces(payload: BatchTraceInput):
    """Performs batch prediction on a list of traces."""
    results = [predict_trace(item) for item in payload.traces]
    return BatchPredictionResult(predictions=results)
