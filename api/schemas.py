"""
Pydantic schemas defining the API's request/response contracts.
"""
from pydantic import BaseModel


class TumorStats(BaseModel):
    necrotic_core_pct: float
    edema_pct: float
    enhancing_tumor_pct: float
    total_tumor_pct: float


class PredictionResponse(BaseModel):
    status: str
    stats: TumorStats
    mask_shape: list[int]
    message: str | None = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
