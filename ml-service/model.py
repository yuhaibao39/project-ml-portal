"""
Model training and prediction logic for California Housing price prediction.

This module handles:
- Loading the California housing dataset from sklearn
- Training a RandomForestRegressor model
- Scaling features with StandardScaler
- Making single and batch predictions
- Providing feature importance and model metadata
"""

import os
import logging
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Paths for saved artifacts
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "california_housing_model.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")

# Feature names in the California housing dataset
FEATURE_NAMES = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]

# Global state populated by load_model()
_model = None
_scaler = None
_metrics = None
_training_date = None
_feature_importance = None


def train_model():
    """
    Train a RandomForestRegressor on the California housing dataset.

    Steps:
        1. Load the California housing dataset
        2. Split into training (80%) and test (20%) sets
        3. Standardize features using StandardScaler
        4. Train a RandomForestRegressor with 100 estimators
        5. Evaluate on the test set (R², MAE, RMSE)
        6. Persist model, scaler, and metrics to disk

    Returns:
        tuple: (model, scaler, metrics_dict)
    """
    global _training_date
    logger.info("Loading California housing dataset...")
    data = fetch_california_housing()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target

    logger.info(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")

    # Train/test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    logger.info(
        f"Train set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples"
    )

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train RandomForestRegressor
    logger.info("Training RandomForestRegressor with n_estimators=100...")
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    logger.info(f"Model evaluation: R²={r2:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}")

    # Record training date
    _training_date = datetime.now(timezone.utc).isoformat()

    # Feature importance
    global _feature_importance
    importances = model.feature_importances_
    _feature_importance = dict(
        sorted(
            zip(FEATURE_NAMES, importances),
            key=lambda x: x[1],
            reverse=True,
        )
    )
    logger.info(f"Feature importances: {_feature_importance}")

    # Package metrics
    metrics = {
        "r2_score": round(float(r2), 4),
        "mae": round(float(mae), 4),
        "rmse": round(float(rmse), 4),
        "model_type": "RandomForestRegressor",
        "n_estimators": 100,
        "n_features": len(FEATURE_NAMES),
        "n_samples": X.shape[0],
        "training_date": _training_date,
    }

    # Save artifacts to disk
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    import json
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Model saved to {MODEL_PATH}")
    logger.info(f"Scaler saved to {SCALER_PATH}")
    logger.info(f"Metrics saved to {METRICS_PATH}")

    return model, scaler, metrics


def load_model():
    """
    Load the trained model and scaler from disk, or train if no saved model exists.

    This populates the global module state so that predict functions
    can reference _model, _scaler, _metrics, etc.

    Returns:
        tuple: (model, scaler, metrics_dict)
    """
    global _model, _scaler, _metrics, _training_date, _feature_importance

    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        logger.info("Loading saved model and scaler from disk...")
        _model = joblib.load(MODEL_PATH)
        _scaler = joblib.load(SCALER_PATH)

        import json
        if os.path.exists(METRICS_PATH):
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                _metrics = json.load(f)
            _training_date = _metrics.get("training_date")
        else:
            _metrics = {"model_type": "RandomForestRegressor", "n_estimators": 100}

        # Recompute feature importance from loaded model
        importances = _model.feature_importances_
        _feature_importance = dict(
            sorted(
                zip(FEATURE_NAMES, importances),
                key=lambda x: x[1],
                reverse=True,
            )
        )
        logger.info("Model and scaler loaded successfully.")
    else:
        logger.info("No saved model found. Training new model...")
        _model, _scaler, _metrics = train_model()

    # Ensure _metrics includes model_type and n_estimators
    if _metrics:
        _metrics.setdefault("model_type", "RandomForestRegressor")
        _metrics.setdefault("n_estimators", 100)
        _metrics.setdefault("n_features", len(FEATURE_NAMES))

    return _model, _scaler, _metrics


def predict(features):
    """
    Predict median house value for a single sample.

    Args:
        features: array-like of shape (8,) or (1, 8) with values in the order
                  [MedInc, HouseAge, AveRooms, AveBedrms, Population,
                   AveOccup, Latitude, Longitude]

    Returns:
        dict: {
            "predicted_value": float (median house value in $100,000s),
            "confidence_interval": [lower_bound, upper_bound],
            "feature_names": list of feature name strings,
            "feature_importance": dict mapping feature name to importance score
        }
    """
    global _model, _scaler, _feature_importance

    if _model is None or _scaler is None:
        load_model()

    # Ensure 2D array
    features_arr = np.array(features).reshape(1, -1)

    # Validate feature count
    if features_arr.shape[1] != len(FEATURE_NAMES):
        raise ValueError(
            f"Expected {len(FEATURE_NAMES)} features, got {features_arr.shape[1]}. "
            f"Features order: {FEATURE_NAMES}"
        )

    # Scale features using the fitted scaler
    features_scaled = _scaler.transform(features_arr)

    # Predict
    pred = _model.predict(features_scaled)[0]
    pred = round(float(pred), 4)

    # Compute confidence interval using individual tree predictions
    # Get predictions from all trees in the forest
    tree_preds = np.array([tree.predict(features_scaled)[0] for tree in _model.estimators_])
    std_err = np.std(tree_preds)
    margin = 1.96 * std_err  # 95% confidence interval
    ci_lower = round(float(pred - margin), 4)
    ci_upper = round(float(pred + margin), 4)

    # Ensure lower bound is not negative (house value can't be negative)
    ci_lower = max(0.0, ci_lower)

    return {
        "predicted_value": pred,
        "confidence_interval": [ci_lower, ci_upper],
        "feature_names": FEATURE_NAMES,
        "feature_importance": _feature_importance,
    }


def predict_batch(features_list):
    """
    Predict median house values for multiple samples.

    Args:
        features_list: list of array-like, each of shape (8,) with feature values
                       in the order [MedInc, HouseAge, AveRooms, AveBedrms,
                       Population, AveOccup, Latitude, Longitude]

    Returns:
        list of dict: each element has the same structure as predict() output
    """
    global _model, _scaler

    if _model is None or _scaler is None:
        load_model()

    # Convert to 2D array
    features_arr = np.array(features_list)

    if features_arr.ndim == 1:
        features_arr = features_arr.reshape(1, -1)

    if features_arr.shape[1] != len(FEATURE_NAMES):
        raise ValueError(
            f"Expected {len(FEATURE_NAMES)} features per sample, "
            f"got {features_arr.shape[1]}."
        )

    # Scale
    features_scaled = _scaler.transform(features_arr)

    # Predict
    preds = _model.predict(features_scaled)

    # Build individual tree predictions for confidence intervals (vectorized)
    # Collect predictions from each tree for all samples
    all_tree_preds = np.array([
        tree.predict(features_scaled) for tree in _model.estimators_
    ])  # shape: (n_estimators, n_samples)

    results = []
    for i, pred in enumerate(preds):
        pred_val = round(float(pred), 4)
        tree_preds_i = all_tree_preds[:, i]
        std_err = np.std(tree_preds_i)
        margin = 1.96 * std_err
        ci_lower = max(0.0, round(float(pred_val - margin), 4))
        ci_upper = round(float(pred_val + margin), 4)

        results.append({
            "predicted_value": pred_val,
            "confidence_interval": [ci_lower, ci_upper],
            "feature_names": FEATURE_NAMES,
            "feature_importance": _feature_importance,
        })

    return results


def get_feature_importance():
    """
    Get the feature importance scores from the trained model.

    Returns:
        dict: Maps feature name (str) to importance score (float),
              sorted in descending order of importance.
              Returns None if the model has not been loaded yet.
    """
    return _feature_importance


def get_model_info():
    """
    Get comprehensive model metadata.

    Returns:
        dict: Model metadata including feature names, importance scores,
              model type, R² score, training date, etc.
    """
    global _metrics, _feature_importance

    if _metrics is None:
        load_model()

    info = {
        "model_type": _metrics.get("model_type", "RandomForestRegressor"),
        "n_estimators": _metrics.get("n_estimators", 100),
        "n_features": _metrics.get("n_features", len(FEATURE_NAMES)),
        "feature_names": FEATURE_NAMES,
        "feature_importance": _feature_importance,
        "r2_score": _metrics.get("r2_score"),
        "mae": _metrics.get("mae"),
        "rmse": _metrics.get("rmse"),
        "training_date": _metrics.get("training_date", _training_date),
        "n_samples": _metrics.get("n_samples"),
    }
    return info
