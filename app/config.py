from pathlib import Path
import os

# Root of the project (one level up from this file's app/ folder)
BASE_DIR = Path(__file__).resolve().parent.parent

# Model artifacts directory (override with MODEL_DIR env var if you deploy
# the models somewhere else, e.g. a mounted volume or cloud bucket)
MODEL_DIR = Path(os.getenv("MODEL_DIR", BASE_DIR / "models"))

BEST_MODEL_PATH = MODEL_DIR / "best_model.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
FEATURE_NAMES_PATH = MODEL_DIR / "feature_names.pkl"
MODEL_RESULTS_PATH = MODEL_DIR / "model_results.csv"

# Risk banding for the predicted failure probability. Tune these based on
# how conservative you want alerts to be.
RISK_THRESHOLDS = {
    "low": 0.25,       # < 0.25  -> Low
    "medium": 0.5,      # < 0.5   -> Medium
    "high": 0.75,       # < 0.75  -> High
    # >= 0.75 -> Critical
}

# Valid product quality types from the AI4I 2020 dataset the model was
# trained on (Low / Medium / High quality variant of the product).
VALID_TYPES = ["L", "M", "H"]

API_TITLE = "Predictive Maintenance API"
API_DESCRIPTION = (
    "Predicts the probability of machine failure from live sensor readings "
    "(air/process temperature, rotational speed, torque, tool wear) using a "
    "gradient-boosted model trained on the AI4I 2020 dataset."
)
API_VERSION = "1.0.0"
