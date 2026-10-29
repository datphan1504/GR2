"""Pydantic schemas for REST API input/output validation."""

from typing import List, Optional
from pydantic import BaseModel, Field


class TraceInput(BaseModel):
    """Network trace input features."""
    dir_seq: List[int] = Field(
        ...,
        description="Packet direction sequence (+1: outgoing, -1: incoming). Typically length 5000."
    )
    time_seq: Optional[List[float]] = Field(
        None,
        description="Packet timestamp or inter-arrival sequence. Typically length 5000."
    )
    metadata: Optional[List[float]] = Field(
        None,
        description="7-element summary feature vector (total_packets, in, out, ratios, total_time, time_per_pkt)."
    )


class PredictionResult(BaseModel):
    """Inference output prediction."""
    predicted_site_id: int
    confidence: float
    is_monitored: bool
    probabilities: Optional[List[float]] = None


class BatchTraceInput(BaseModel):
    """Batch input for multiple traffic traces."""
    traces: List[TraceInput]


class BatchPredictionResult(BaseModel):
    """Batch prediction results."""
    predictions: List[PredictionResult]


class HealthResponse(BaseModel):
    """API health and readiness status."""
    status: str
    model_name: str
    model_loaded: bool
