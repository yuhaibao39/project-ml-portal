"""
ML model service for the Property Value Estimator.

Handles loading/training of the scikit-learn regressor and provides
prediction methods with proper scaling and error handling.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from app.models.schemas import PredictionResult, PropertyFeatures

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODEL_PATH: str = os.getenv(
    "MODEL_PATH", "app/ml_model/housing_model.pkl"
)
SCALER_PATH: str = os.getenv(
    "SCALER_PATH", "app/ml_model/scaler.pkl"
)

# ---------------------------------------------------------------------------
# Feature metadata
# ---------------------------------------------------------------------------
FEATURE_NAMES: list[str] = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]

TARGET_NAME: str = "MedHouseVal"

# Default training date used when model is freshly trained
TRAINING_DATE_STR: str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


class ModelService:
    """Service that manages the ML regression model for house price prediction.

    On construction the model is either loaded from a ``.pkl`` file on disk
    or trained on the California housing dataset automatically.
    """

    def __init__(self) -> None:
        self.model: Optional[RandomForestRegressor] = None
        self.scaler: Optional[StandardScaler] = None
        self.r2_score_value: Optional[float] = None
        self.training_date: str = TRAINING_DATE_STR
        self.feature_importance: dict[str, float] = {}
        self._uptime: float = time.time()
        self._loaded: bool = False
        self._version: str = "1.0.0"

        self.load_or_train_model()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_or_train_model(self) -> None:
        """Load a pickled model from disk or train a fresh one.

        The routine first attempts to load ``housing_model.pkl`` and
        ``scaler.pkl``.  If either file is missing it falls back to
        training a ``RandomForestRegressor`` on the California housing
        dataset, persisting both artifacts afterwards.
        """
        if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
            try:
                self.model = joblib.load(MODEL_PATH)
                self.scaler = joblib.load(SCALER_PATH)
                # Attempt to read sidecar metadata if it exists
                meta_path = MODEL_PATH.replace(".pkl", "_meta.joblib")
                if os.path.exists(meta_path):
                    meta: dict[str, Any] = joblib.load(meta_path)
                    self.r2_score_value = meta.get("r2_score")
                    self.training_date = meta.get(
                        "training_date", TRAINING_DATE_STR
                    )
                    self._version = meta.get("version", "1.0.0")
                    self.feature_importance = meta.get(
                        "feature_importance", {}
                    )
                else:
                    # Derive importance from the loaded model if possible
                    self._extract_feature_importance()

                self._loaded = True
                logger.info(
                    "Model loaded from disk (R²=%.4f)", self.r2_score_value
                )
                return
            except Exception:
                logger.warning(
                    "Failed to load model from disk, retraining.",
                    exc_info=True,
                )

        logger.info("Training new RandomForestRegressor on California housing data.")
        self._train_model()
        self._loaded = True

    def predict(self, features: PropertyFeatures) -> PredictionResult:
        """Predict the median house value for a single property.

        Parameters
        ----------
        features : PropertyFeatures
            Validated input features from the API request.

        Returns
        -------
        PredictionResult
            Predicted value (in hundreds of thousands of USD), confidence
            interval, and feature importance dictionary.
        """
        if self.model is None or self.scaler is None:
            raise RuntimeError("Model has not been initialised.")

        input_df = self._features_to_dataframe([features])
        scaled = self.scaler.transform(input_df)
        pred: float = float(self.model.predict(scaled)[0])

        # Approximate 95 % confidence interval based on the training R²
        ci = self._confidence_interval(pred)

        return PredictionResult(
            predicted_value=round(pred, 4),
            confidence_interval=ci,
            features_used=self.feature_importance,
        )

    def predict_batch(
        self, features_list: list[PropertyFeatures]
    ) -> list[PredictionResult]:
        """Predict median house values for multiple properties.

        Parameters
        ----------
        features_list : list[PropertyFeatures]
            A list of validated feature sets.

        Returns
        -------
        list[PredictionResult]
            One result per input property.
        """
        if self.model is None or self.scaler is None:
            raise RuntimeError("Model has not been initialised.")

        if not features_list:
            return []

        input_df = self._features_to_dataframe(features_list)
        scaled = self.scaler.transform(input_df)
        raw_preds: np.ndarray = self.model.predict(scaled)

        results: list[PredictionResult] = []
        for val in raw_preds:
            ci = self._confidence_interval(float(val))
            results.append(
                PredictionResult(
                    predicted_value=round(float(val), 4),
                    confidence_interval=ci,
                    features_used=self.feature_importance,
                )
            )
        return results

    def get_model_info(self) -> dict[str, Any]:
        """Return metadata about the currently loaded model.

        Returns
        -------
        dict
            Keys include *model_type*, *features*, *n_features*,
            *r2_score*, *training_date*, *feature_importance*.
        """
        return {
            "model_type": type(self.model).__name__
            if self.model
            else "None",
            "features": FEATURE_NAMES,
            "n_features": len(FEATURE_NAMES),
            "r2_score": self.r2_score_value,
            "training_date": self.training_date,
            "feature_importance": self.feature_importance,
        }

    def is_loaded(self) -> bool:
        """Return whether the model has been successfully loaded/trained."""
        return self._loaded and self.model is not None

    def version(self) -> str:
        """Return the model version string."""
        return self._version

    def uptime_seconds(self) -> float:
        """Return seconds since this service was instantiated."""
        return time.time() - self._uptime

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _train_model(self) -> None:
        """Train a RandomForestRegressor on the California housing dataset.

        The trained model and scaler are persisted to disk afterwards,
        along with a metadata sidecar file.
        """
        # Load dataset
        housing = fetch_california_housing()
        X: np.ndarray = housing.data
        y: np.ndarray = housing.target

        # Train / test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train regressor
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred = model.predict(X_test_scaled)
        r2 = r2_score(y_test, y_pred)

        self.model = model
        self.scaler = scaler
        self.r2_score_value = round(r2, 4)
        self.training_date = datetime.now(timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        )
        self._extract_feature_importance()

        # Persist
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)

        meta: dict[str, Any] = {
            "r2_score": self.r2_score_value,
            "training_date": self.training_date,
            "version": self._version,
            "feature_importance": self.feature_importance,
        }
        meta_path = MODEL_PATH.replace(".pkl", "_meta.joblib")
        joblib.dump(meta, meta_path)

        logger.info(
            "Model trained and saved (R²=%.4f) to %s",
            self.r2_score_value,
            MODEL_PATH,
        )

    def _extract_feature_importance(self) -> None:
        """Extract feature importance from the trained model if available."""
        if self.model is None:
            self.feature_importance = {}
            return

        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
            self.feature_importance = {
                name: round(float(imp), 4)
                for name, imp in zip(FEATURE_NAMES, importances)
            }
        elif hasattr(self.model, "coef_"):
            coefs = self.model.coef_
            self.feature_importance = {
                name: round(float(abs(c)), 4)
                for name, c in zip(FEATURE_NAMES, coefs)
            }
        else:
            self.feature_importance = {name: 0.0 for name in FEATURE_NAMES}

    def _features_to_dataframe(
        self, features_list: list[PropertyFeatures]
    ) -> pd.DataFrame:
        """Convert a list of Pydantic feature models to a DataFrame.

        The column order strictly follows ``FEATURE_NAMES`` so that the
        scaler and model receive data in the expected layout.
        """
        records = [f.model_dump() for f in features_list]
        df = pd.DataFrame(records, columns=FEATURE_NAMES)

        # Safely clip extreme outliers that may pass validation
        df["AveBedrms"] = df["AveBedrms"].clip(lower=0.1)
        df["AveRooms"] = df["AveRooms"].clip(lower=1.0)

        return df

    def _confidence_interval(
        self, prediction: float
    ) -> Optional[tuple[float, float]]:
        """Return an approximate 95 % confidence interval.

        Uses the model's R² as a rough proxy: a model with a higher R²
        has a narrower interval.  The base margin is 0.5 (in target
        units of hundreds of thousands of USD).
        """
        if self.r2_score_value is None:
            return None

        # Wider margin for worse models, tighter for better ones
        margin = 0.5 * (1.0 + (1.0 - self.r2_score_value))
        return (round(prediction - margin, 4), round(prediction + margin, 4))
