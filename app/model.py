"""
Loads the trained model + scaler + feature order once at startup and
exposes a simple predict interface. Keeping this as a class (rather than
module-level globals) makes it easy to mock in tests.

Note: SMOTE is intentionally NOT used here. SMOTE is a training-time-only
technique for rebalancing classes before fitting; it must never be applied
to real inference data.
"""
from __future__ import annotations

import logging
from functools import lru_cache

import joblib
import pandas as pd

from app import config
from app.features import engineer_features

logger = logging.getLogger("predictive_maintenance")


class ModelNotReadyError(RuntimeError):
    """Raised if a prediction is requested before artifacts finished loading."""


class PredictiveMaintenanceModel:
    def __init__(self) -> None:
        self.model = None
        self.scaler = None
        self.feature_names: list[str] | None = None
        self._load()

    def _load(self) -> None:
        missing = [
            p
            for p in (
                config.BEST_MODEL_PATH,
                config.SCALER_PATH,
                config.FEATURE_NAMES_PATH,
            )
            if not p.exists()
        ]
        if missing:
            raise FileNotFoundError(
                f"Missing model artifact(s): {[str(p) for p in missing]}. "
                f"Make sure best_model.pkl, scaler.pkl and feature_names.pkl "
                f"are present in {config.MODEL_DIR}."
            )

        logger.info("Loading model artifacts from %s", config.MODEL_DIR)
        self.model = joblib.load(config.BEST_MODEL_PATH)
        self.scaler = joblib.load(config.SCALER_PATH)
        self.feature_names = joblib.load(config.FEATURE_NAMES_PATH)
        logger.info(
            "Loaded %s expecting %d features",
            type(self.model).__name__,
            len(self.feature_names),
        )

    def _to_row(self, reading) -> pd.DataFrame:
        """Turn one MachineReading (pydantic model) into a model-ready row."""
        engineered = engineer_features(
            machine_type=reading.type,
            air_temperature=reading.air_temperature,
            process_temperature=reading.process_temperature,
            rotational_speed=reading.rotational_speed,
            torque=reading.torque,
            tool_wear=reading.tool_wear,
        )
        row = pd.DataFrame([engineered])
        # Reindex to the exact column order the scaler/model were fit on.
        # This is what protects us from any column-ordering assumptions.
        row = row.reindex(columns=self.feature_names)
        if row.isnull().any(axis=None):
            missing_cols = row.columns[row.isnull().any()].tolist()
            raise ValueError(
                f"Feature engineering did not produce required column(s): {missing_cols}"
            )
        return row

    def risk_level(self, probability: float) -> str:
        t = config.RISK_THRESHOLDS
        if probability < t["low"]:
            return "Low"
        if probability < t["medium"]:
            return "Medium"
        if probability < t["high"]:
            return "High"
        return "Critical"

    def predict_one(self, reading) -> dict:
        row = self._to_row(reading)
        scaled = self.scaler.transform(row)
        pred = int(self.model.predict(scaled)[0])
        proba = float(self.model.predict_proba(scaled)[0][1])
        return {
            "prediction": pred,
            "label": "Failure" if pred == 1 else "Normal",
            "failure_probability": round(proba, 6),
            "risk_level": self.risk_level(proba),
        }

    def predict_batch(self, readings: list) -> list[dict]:
        rows = pd.concat([self._to_row(r) for r in readings], ignore_index=True)
        scaled = self.scaler.transform(rows)
        preds = self.model.predict(scaled)
        probas = self.model.predict_proba(scaled)[:, 1]
        results = []
        for pred, proba in zip(preds, probas):
            proba = float(proba)
            results.append(
                {
                    "prediction": int(pred),
                    "label": "Failure" if pred == 1 else "Normal",
                    "failure_probability": round(proba, 6),
                    "risk_level": self.risk_level(proba),
                }
            )
        return results


@lru_cache
def get_model() -> PredictiveMaintenanceModel:
    """Cached singleton so the (relatively expensive) load happens once."""
    return PredictiveMaintenanceModel()
