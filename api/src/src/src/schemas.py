# =============================================================================
# BIBLIOTECAS E MÓDULOS
# =============================================================================

from typing import List

from pydantic import BaseModel

# =============================================================================
# SCHEMAS
# =============================================================================


class Message(BaseModel):
    message: str


class Prediction(BaseModel):
    message: str
    prediction: List[List[float]]


class TimeSeries(BaseModel):
    series: List[List[float]]
