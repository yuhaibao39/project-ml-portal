"""
Pydantic models/schemas for the Property Value Estimator API.

Defines request and response models with validation appropriate
for California housing data.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field


class PropertyFeatures(BaseModel):
    """Input features for a single property valuation prediction.

    All values correspond to the California housing dataset features.
    """

    MedInc: float = Field(
        ...,
        ge=0.0,
        le=20.0,
        description="Median income in block group (tens of thousands of USD)",
        examples=[3.5],
    )
    HouseAge: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Median house age in block group (years)",
        examples=[25.0],
    )
    AveRooms: float = Field(
        ...,
        ge=1.0,
        le=50.0,
        description="Average number of rooms per household",
        examples=[6.0],
    )
    AveBedrms: float = Field(
        ...,
        ge=0.0,
        le=20.0,
        description="Average number of bedrooms per household",
        examples=[1.5],
    )
    Population: float = Field(
        ...,
        ge=0.0,
        le=50_000.0,
        description="Block group population",
        examples=[1200.0],
    )
    AveOccup: float = Field(
        ...,
        ge=0.0,
        le=10_000.0,
        description="Average number of household members",
        examples=[3.0],
    )
    Latitude: float = Field(
        ...,
        ge=32.0,
        le=42.0,
        description="Latitude of block group centroid (degrees)",
        examples=[34.5],
    )
    Longitude: float = Field(
        ...,
        ge=-125.0,
        le=-113.0,
        description="Longitude of block group centroid (degrees)",
        examples=[-118.5],
    )

    class Config:
        json_schema_extra = {
            "example": {
                "MedInc": 3.5,
                "HouseAge": 25.0,
                "AveRooms": 6.0,
                "AveBedrms": 1.5,
                "Population": 1200.0,
                "AveOccup": 3.0,
                "Latitude": 34.5,
                "Longitude": -118.5,
            }
        }


class PredictionResult(BaseModel):
    """Result of a single property valuation prediction."""

    predicted_value: float = Field(
        ...,
        description="Predicted median house value in hundreds of thousands of USD",
        examples=[2.5],
    )
    confidence_interval: Optional[tuple[float, float]] = Field(
        None,
        description="Approximate 95 % confidence interval for the prediction",
    )
    features_used: dict[str, float] = Field(
        ...,
        description="Feature names mapped to their importance scores",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "predicted_value": 2.5,
                "confidence_interval": (2.1, 2.9),
                "features_used": {
                    "MedInc": 0.6,
                    "HouseAge": 0.05,
                    "AveRooms": 0.1,
                },
            }
        }


class PredictionHistory(BaseModel):
    """A record of a previously made prediction."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    features: PropertyFeatures
    result: PredictionResult
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )

    class Config:
        json_schema_extra = {
            "example": {
                "id": "a1b2c3d4e5f6",
                "features": {
                    "MedInc": 3.5,
                    "HouseAge": 25.0,
                    "AveRooms": 6.0,
                    "AveBedrms": 1.5,
                    "Population": 1200.0,
                    "AveOccup": 3.0,
                    "Latitude": 34.5,
                    "Longitude": -118.5,
                },
                "result": {
                    "predicted_value": 2.5,
                    "confidence_interval": [2.1, 2.9],
                    "features_used": {"MedInc": 0.6},
                },
                "timestamp": "2025-01-15T12:00:00Z",
            }
        }


class BatchPredictionRequest(BaseModel):
    """Request body for batch property predictions."""

    properties: list[PropertyFeatures] = Field(
        ...,
        min_length=1,
        max_length=10_000,
        description="List of properties to predict",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "properties": [
                    {
                        "MedInc": 3.5,
                        "HouseAge": 25.0,
                        "AveRooms": 6.0,
                        "AveBedrms": 1.5,
                        "Population": 1200.0,
                        "AveOccup": 3.0,
                        "Latitude": 34.5,
                        "Longitude": -118.5,
                    }
                ],
            }
        }


class BatchPredictionResponse(BaseModel):
    """Response body for batch property predictions."""

    predictions: list[PredictionResult]


class ErrorResponse(BaseModel):
    """Standard error response body."""

    error: str = Field(..., description="Short error code or message")
    detail: Optional[str] = Field(None, description="Detailed error information")


class HealthResponse(BaseModel):
    """Response body for the health check endpoint."""

    status: str
    model_loaded: bool
    model_version: Optional[str] = None
    uptime_seconds: Optional[float] = None


class ModelInfoResponse(BaseModel):
    """Response body for model metadata."""

    model_type: str
    features: list[str]
    n_features: int
    r2_score: Optional[float] = None
    training_date: Optional[str] = None
    feature_importance: Optional[dict[str, float]] = None
