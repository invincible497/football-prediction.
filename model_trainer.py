"""
Machine Learning Training, Validation, and Evaluation Module.
Trains, evaluates, and compares:
1. Logistic Regression
2. Random Forest Classifier
3. Gradient Boosting Classifier
Uses chronological train/test splitting, comprehensive multi-class metrics,
Brier score, log loss, confusion matrices, and saves artifacts via Joblib.
"""

import os
import json
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    log_loss, confusion_matrix
)

from database import get_all_matches, DB_FILE
from feature_engineering import (
    build_match_features_dataset, FEATURE_COLUMNS,
    TARGET_INV_MAP, TARGET_LABEL_MAP
)

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

METRICS_FILE = os.path.join(MODELS_DIR, "model_metrics.json")
BEST_MODEL_FILE = os.path.join(MODELS_DIR, "best_model.joblib")
SCALER_FILE = os.path.join(MODELS_DIR, "scaler.joblib")


def compute_multiclass_brier_score(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """
    Computes multi-class Brier score:
    BS = (1/N) * sum_i sum_k (p_ik - y_ik)^2
    A score of 0 is perfect prediction.
    """
    n_samples, n_classes = y_proba.shape
    one_hot = np.zeros((n_samples, n_classes))
    for i, label in enumerate(y_true):
        one_hot[i, int(label)] = 1.0
    return float(np.mean(np.sum((y_proba - one_hot) ** 2, axis=1)))


def train_and_evaluate_all_models(
    db_path: str = DB_FILE,
    test_ratio: float = 0.20
) -> Dict[str, Any]:
    """
    Loads all matches from database, engineers temporal leak-free features,
    performs chronological train/test split, trains and evaluates the 3 algorithms,
    selects the best model, and saves artifacts.
    """
    matches_df = get_all_matches(db_path)
    if len(matches_df) < 50:
        raise ValueError("Insufficient match data in database to train models.")

    # 1. Feature Engineering (Strictly chronological)
    X_full, y_full = build_match_features_dataset(matches_df)
    
    # Extract only feature columns
    X_features = X_full[FEATURE_COLUMNS].copy()

    # Handle any potential remaining NaNs
    X_features = X_features.fillna(X_features.median())

    # 2. Chronological Train-Test Split (NO random shuffling to prevent time leakage)
    split_idx = int(len(X_features) * (1.0 - test_ratio))
    X_train = X_features.iloc[:split_idx]
    X_test = X_features.iloc[split_idx:]
    y_train = y_full.iloc[:split_idx].values
    y_test = y_full.iloc[split_idx:].values

    # 3. Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Initialize Models
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            C=0.5,
            class_weight='balanced',
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=8,
            min_samples_split=8,
            class_weight='balanced',
            random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=120,
            learning_rate=0.07,
            max_depth=4,
            subsample=0.85,
            random_state=42
        )
    }

    results = {}
    best_model_name = None
    best_f1_score = -1.0

    for name, model in models.items():
        # Train
        model.fit(X_train_scaled, y_train)

        # Predict
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec_w = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec_w = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1_w = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
        loss = log_loss(y_test, y_proba)
        brier = compute_multiclass_brier_score(y_test, y_proba)
        cm = confusion_matrix(y_test, y_pred).tolist()

        # Feature importances
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_.tolist()
        elif hasattr(model, 'coef_'):
            importances = np.mean(np.abs(model.coef_), axis=0).tolist()
        else:
            importances = [0.0] * len(FEATURE_COLUMNS)

        feature_imp_dict = {
            col: round(imp, 5) for col, imp in zip(FEATURE_COLUMNS, importances)
        }
        # Sort by importance descending
        sorted_imp = dict(sorted(feature_imp_dict.items(), key=lambda item: item[1], reverse=True))

        results[name] = {
            "accuracy": round(float(acc), 4),
            "precision_weighted": round(float(prec_w), 4),
            "recall_weighted": round(float(rec_w), 4),
            "f1_weighted": round(float(f1_w), 4),
            "f1_macro": round(float(f1_macro), 4),
            "log_loss": round(float(loss), 4),
            "brier_score": round(float(brier), 4),
            "confusion_matrix": cm,
            "feature_importances": sorted_imp
        }

        # Save individual model
        slug = name.lower().replace(" ", "_")
        joblib.dump(model, os.path.join(MODELS_DIR, f"{slug}.joblib"))

        # Determine best model based on weighted F1 score
        if f1_w > best_f1_score:
            best_f1_score = f1_w
            best_model_name = name

    # Save best model and scaler
    best_model_obj = models[best_model_name]
    joblib.dump(best_model_obj, BEST_MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)

    summary = {
        "best_model": best_model_name,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "feature_columns": FEATURE_COLUMNS,
        "models": results,
        "classes": [TARGET_LABEL_MAP[i] for i in range(3)]
    }

    with open(METRICS_FILE, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def load_model_and_scaler(model_name: Optional[str] = None):
    """Loads a specific model (or the best model) and the fitted scaler."""
    if not os.path.exists(SCALER_FILE):
        return None, None

    scaler = joblib.load(SCALER_FILE)

    if model_name:
        slug = model_name.lower().replace(" ", "_")
        target_path = os.path.join(MODELS_DIR, f"{slug}.joblib")
        if os.path.exists(target_path):
            return joblib.load(target_path), scaler

    if os.path.exists(BEST_MODEL_FILE):
        return joblib.load(BEST_MODEL_FILE), scaler

    return None, scaler


def get_stored_metrics() -> Optional[Dict[str, Any]]:
    """Reads saved model metrics JSON."""
    if os.path.exists(METRICS_FILE):
        try:
            with open(METRICS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return None
    return None
