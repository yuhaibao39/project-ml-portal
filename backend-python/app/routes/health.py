"""
Health and model-info route handlers for the Property Value Estimator.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends

from app.models.schemas import HealthResponse, ModelInfoResponse
from app.services.model_service import ModelService

logger = logging.getLogger(__name__)

router = APIRouter()

# ---------------------------------------------------------------------------
# Dependency: acquire the singleton model service from app.state
# ---------------------------------------------------------------------------


def _get_model_service(request: "Request") -> ModelService:  # type: ignore[name-defined]  # noqa: F821
    """Return the ModelService instance stored on the application.

    This function is used as a FastAPI dependency so route handlers can
    access the shared model instance.
    """
    return request.app.state.model_service


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Returns the current health status of the API and ML model.",
)
async def health_check(
    model_service: ModelService = Depends(_get_model_service),
) -> HealthResponse:
    """Check API health and whether the ML model is loaded."""
    return HealthResponse(
        status="healthy",
        model_loaded=model_service.is_loaded(),
        model_version=model_service.version() if model_service.is_loaded() else None,
        uptime_seconds=round(model_service.uptime_seconds(), 2),
    )


@router.get(
    "/api/v1/model/info",
    response_model=ModelInfoResponse,
    summary="Model metadata",
    description="Returns detailed metadata about the trained ML model.",
)
async def model_info(
    model_service: ModelService = Depends(_get_model_service),
) -> ModelInfoResponse:
    """Return model metadata including type, features, R², and importance."""
    info = model_service.get_model_info()
    return ModelInfoResponse(
        model_type=info["model_type"],
        features=info["features"],
        n_features=info["n_features"],
        r2_score=info["r2_score"],
        training_date=info["training_date"],
        feature_importance=info["feature_importance"],
    )
