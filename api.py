# api.py
"""
Production-ready FastAPI backend for the Federated Learning Diabetes Prediction system.

Architecture
────────────
  • Model   : DiabetesNet  (8 → 64 → 32 → 16 → 1, raw logit)
  • Sigmoid applied here at inference — NOT baked into the model's forward()
  • Scaler  : StandardScaler fitted on training data only  (artifacts/scaler.pkl)
  • Checkpoint: artifacts/global_model.pth

Endpoints
─────────
  GET  /health   – liveness probe
  POST /predict  – accepts 8 clinical features, returns prediction + probability

Run
───
  uvicorn api:app --reload --port 8000
"""

import os
import sys
import logging
import traceback
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import torch
import joblib

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator, model_validator

# ── Project root on sys.path so "models.model" resolves ──────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from models.model import DiabetesNet

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("fl-api")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

MODEL_PATH  = ROOT / "artifacts" / "global_model.pth"
SCALER_PATH = ROOT / "artifacts" / "scaler.pkl"
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────────────────────
# Global model state  (loaded once at startup, reused for every request)
# ─────────────────────────────────────────────────────────────────────────────

_model:  Optional[DiabetesNet] = None
_scaler = None


# ─────────────────────────────────────────────────────────────────────────────
# Model & scaler loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_model(path: Path = MODEL_PATH) -> DiabetesNet:
    """
    Instantiate DiabetesNet and load saved weights from *path*.

    Raises
    ------
    FileNotFoundError  – checkpoint does not exist
    RuntimeError       – weights incompatible with current architecture
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {path}\n"
            "Train the federated model first:\n"
            "  python run_project.py\n"
            "or:\n"
            "  python train_federated.py --mode server --rounds 10 --clients 3"
        )

    net = DiabetesNet(input_dim=8).to(DEVICE)

    # map_location handles CPU-only machines even if model was saved on GPU
    state_dict = torch.load(str(path), map_location=DEVICE)
    net.load_state_dict(state_dict)
    net.eval()   # switch BatchNorm / Dropout to inference mode

    log.info("Model loaded from %s  (device=%s)", path, DEVICE)
    return net


def load_scaler(path: Path = SCALER_PATH):
    """
    Load the fitted StandardScaler from *path*.

    The scaler was fit ONLY on training data to prevent data leakage.

    Raises
    ------
    FileNotFoundError  – scaler file does not exist
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Scaler not found: {path}\n"
            "Generate federated datasets first:\n"
            "  python train_federated.py --mode prepare --num_clients 3"
        )

    scaler = joblib.load(str(path))
    log.info("Scaler loaded from %s", path)
    return scaler


# ─────────────────────────────────────────────────────────────────────────────
# Application lifespan  (startup / shutdown hooks)
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and scaler exactly once when the server starts."""
    global _model, _scaler

    log.info("=" * 55)
    log.info("  FL Diabetes Prediction API  —  starting up")
    log.info("=" * 55)

    try:
        _model  = load_model()
        _scaler = load_scaler()
        log.info("Startup complete.  Listening for requests.")
    except FileNotFoundError as exc:
        # Keep the server alive so /health can describe the problem.
        # POST /predict will return 503 until artifacts exist.
        log.warning("Startup warning — artifact not found:\n%s", exc)
        log.warning("POST /predict will return 503 until training is complete.")

    yield  # ← server is live here

    log.info("Shutting down FL Diabetes API.")


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI application
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "FL Diabetes Prediction API",
    description = "Federated Learning backend for Type 2 Diabetes risk prediction.",
    version     = "1.0.0",
    lifespan    = lifespan,
)

# ── CORS ─────────────────────────────────────────────────────────────────────
# Allows the Next.js dev server (localhost:3000) to call this API.
# In production replace "*" with the actual frontend domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Global exception handler
# Ensures the API never leaks a raw Python traceback to the client.
# ─────────────────────────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    log.error(
        "Unhandled exception on %s:\n%s",
        request.url.path,
        traceback.format_exc(),
    )
    return JSONResponse(
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR,
        content     = {
            "error":  "Internal server error",
            "detail": str(exc),
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic schemas
# ─────────────────────────────────────────────────────────────────────────────

class PatientInput(BaseModel):
    """
    Eight clinical features sent by the Next.js prediction form.

    Accepted field name formats
    ───────────────────────────
    Both camelCase (frontend default) and snake_case are accepted.
    The pre-validator normalizes the raw dict before Pydantic touches it,
    so every downstream validator always sees the canonical camelCase name.

    Canonical   ←  also accepted
    ──────────────────────────────────────────────────
    bloodPressure  ←  blood_pressure
    skinThickness  ←  skin_thickness
    dpf            ←  diabetes_pedigree, diabetes_pedigree_function
    """

    pregnancies:   float = Field(..., description="Number of pregnancies  (0 – 20)")
    glucose:       float = Field(..., description="Plasma glucose in mg/dL  (0 – 300)")
    bloodPressure: float = Field(..., description="Diastolic blood pressure in mm Hg  (0 – 200)")
    skinThickness: float = Field(..., description="Triceps skinfold thickness in mm  (0 – 100)")
    insulin:       float = Field(..., description="2-hour serum insulin in µU/mL  (0 – 900)")
    bmi:           float = Field(..., description="Body mass index in kg/m²  (10 – 70)")
    dpf:           float = Field(..., description="Diabetes pedigree function score  (0 – 3)")
    age:           float = Field(..., description="Age in years  (1 – 120)")

    # ── Input normalizer: runs BEFORE any field validator ─────────────────────
    #
    # mode="before" means Pydantic passes us the raw incoming dict.
    # We rewrite snake_case aliases to the canonical camelCase keys so that
    # all @field_validator methods only ever need to handle one name each.
    #
    # Priority: if a caller somehow sends both forms (e.g. bloodPressure AND
    # blood_pressure), the camelCase value wins because we only write the alias
    # when the canonical key is absent.

    @model_validator(mode="before")
    @classmethod
    def normalize_field_names(cls, data: dict) -> dict:
        """
        Rewrite snake_case aliases to canonical camelCase field names.

        This lets callers use either convention without any behaviour change.
        """
        if not isinstance(data, dict):
            # Pydantic will raise a descriptive error on its own for non-dict input
            return data

        # Alias map: snake_case (and any other accepted variants) → canonical key
        ALIASES: dict[str, str] = {
            "blood_pressure":            "bloodPressure",
            "skin_thickness":            "skinThickness",
            "diabetes_pedigree":         "dpf",
            "diabetes_pedigree_function":"dpf",
        }

        normalized: dict = {}
        for key, value in data.items():
            canonical = ALIASES.get(key, key)   # remap or keep as-is
            # Only write the canonical key if it hasn't already been set by a
            # camelCase field earlier in the dict (camelCase wins on conflict).
            if canonical not in normalized:
                normalized[canonical] = value

        return normalized

    # ── Per-field range validators ────────────────────────────────────────────

    @field_validator("pregnancies")
    @classmethod
    def v_pregnancies(cls, v: float) -> float:
        if v < 0 or v > 20:
            raise ValueError("pregnancies must be between 0 and 20")
        return v

    @field_validator("glucose")
    @classmethod
    def v_glucose(cls, v: float) -> float:
        if v < 0 or v > 300:
            raise ValueError("glucose must be between 0 and 300 mg/dL")
        return v

    @field_validator("bloodPressure")
    @classmethod
    def v_blood_pressure(cls, v: float) -> float:
        if v < 0 or v > 200:
            raise ValueError("bloodPressure must be between 0 and 200 mm Hg")
        return v

    @field_validator("skinThickness")
    @classmethod
    def v_skin_thickness(cls, v: float) -> float:
        if v < 0 or v > 100:
            raise ValueError("skinThickness must be between 0 and 100 mm")
        return v

    @field_validator("insulin")
    @classmethod
    def v_insulin(cls, v: float) -> float:
        if v < 0 or v > 900:
            raise ValueError("insulin must be between 0 and 900 µU/mL")
        return v

    @field_validator("bmi")
    @classmethod
    def v_bmi(cls, v: float) -> float:
        if v < 10 or v > 70:
            raise ValueError("bmi must be between 10 and 70 kg/m²")
        return v

    @field_validator("dpf")
    @classmethod
    def v_dpf(cls, v: float) -> float:
        if v < 0 or v > 3:
            raise ValueError("dpf (diabetes pedigree function) must be between 0 and 3")
        return v

    @field_validator("age")
    @classmethod
    def v_age(cls, v: float) -> float:
        if v < 1 or v > 120:
            raise ValueError("age must be between 1 and 120 years")
        return v

    # ── Cross-field sanity check ──────────────────────────────────────────────

    @model_validator(mode="after")
    def cross_field_checks(self) -> "PatientInput":
        """
        Catch combinations that are physiologically impossible.
        A glucose of 0 usually means a missing value in the dataset, not a
        true measurement.  Raise a clear error rather than silently passing
        zeroed data through the model.
        """
        if self.glucose == 0 and self.insulin > 0:
            raise ValueError(
                "A glucose reading of 0 combined with non-zero insulin is "
                "physiologically implausible.  Glucose = 0 typically means a "
                "missing measurement — please check the input data."
            )
        return self

    class Config:
        json_schema_extra = {
            # Both examples are valid — the API normalizes either format.
            "examples": {
                "camelCase (frontend default)": {
                    "summary": "camelCase field names — sent by the Next.js form",
                    "value": {
                        "pregnancies":   2,
                        "glucose":       148,
                        "bloodPressure": 72,
                        "skinThickness": 35,
                        "insulin":       0,
                        "bmi":           33.6,
                        "dpf":           0.627,
                        "age":           50,
                    },
                },
                "snake_case (Python / curl default)": {
                    "summary": "snake_case field names — also accepted",
                    "value": {
                        "pregnancies":      10,
                        "glucose":          98,
                        "blood_pressure":   100,
                        "skin_thickness":   59,
                        "insulin":          500,
                        "bmi":              24.9,
                        "diabetes_pedigree":1.2,
                        "age":              35,
                    },
                },
            }
        }


class PredictionResponse(BaseModel):
    """Response body returned to the Next.js frontend."""
    prediction:      int    # 1 = Diabetic, 0 = Non-Diabetic
    probability:     float  # sigmoid output — 0.0 to 1.0
    risk_level:      str    # "Low" | "Moderate" | "Elevated" | "High"
    recommendation:  str    # clinical recommendation text
    threshold:       float  # decision threshold used
    model_version:   str    # model identifier


# ─────────────────────────────────────────────────────────────────────────────
# Inference pipeline
# ─────────────────────────────────────────────────────────────────────────────

def predict_diabetes(patient: PatientInput) -> PredictionResponse:
    """
    Full inference pipeline for one patient.

    Steps
    ─────
    1. Extract the 8 feature values in the exact column order used during training.
    2. Reshape into a (1, 8) float32 NumPy array.
    3. Apply the fitted StandardScaler — same mean/std as training data.
    4. Convert to a PyTorch float32 tensor on the inference device (CPU or CUDA).
    5. Forward pass through DiabetesNet — produces a raw scalar logit.
    6. Apply sigmoid — converts logit to probability in [0, 1].
    7. Threshold at 0.5: probability > 0.5 → "Diabetic".

    The feature ORDER below must match the column order in diabetes.csv /
    the order passed to scaler.fit_transform() during data preparation.
    (Pima Indians standard: Pregnancies, Glucose, BloodPressure, SkinThickness,
     Insulin, BMI, DiabetesPedigreeFunction, Age)
    """

    # ── Step 1: ordered feature vector ───────────────────────────────────────
    raw_values: list = [
        patient.pregnancies,
        patient.glucose,
        patient.bloodPressure,   # frontend: bloodPressure → training col: BloodPressure
        patient.skinThickness,
        patient.insulin,
        patient.bmi,
        patient.dpf,             # frontend: dpf           → training col: DiabetesPedigreeFunction
        patient.age,
    ]

    # ── Step 2: NumPy array, shape (1, 8) ────────────────────────────────────
    feature_array: np.ndarray = np.array(raw_values, dtype=np.float32).reshape(1, -1)

    # ── Step 3: scale using training statistics ───────────────────────────────
    scaled_array: np.ndarray = _scaler.transform(feature_array)

    # ── Step 4: PyTorch tensor ────────────────────────────────────────────────
    tensor: torch.Tensor = torch.tensor(scaled_array, dtype=torch.float32).to(DEVICE)

    # ── Step 5: forward pass — no gradient needed during inference ────────────
    with torch.no_grad():
        logit: torch.Tensor = _model(tensor)          # shape: (1, 1)

    # ── Step 6: sigmoid → probability ────────────────────────────────────────
    probability: float = torch.sigmoid(logit).squeeze().item()

    # ── Step 7: risk level from probability ───────────────────────────────────
    THRESHOLD = 0.5
    if probability < 0.30:
        risk_level     = "Low"
        recommendation = (
            "Your risk appears low. Maintain a healthy lifestyle with regular "
            "exercise, balanced diet, and routine check-ups."
        )
    elif probability < 0.50:
        risk_level     = "Moderate"
        recommendation = (
            "Moderate risk detected. Consider lifestyle improvements — increase "
            "physical activity, reduce sugar intake, and monitor your weight. "
            "Schedule a check-up with your doctor."
        )
    elif probability < 0.70:
        risk_level     = "Elevated"
        recommendation = (
            "Elevated risk detected. Consult your healthcare provider for a "
            "glucose tolerance test. Focus on diet control and regular monitoring."
        )
    else:
        risk_level     = "High"
        recommendation = (
            "High risk detected. Seek medical consultation promptly. A fasting "
            "blood glucose test and HbA1c screening are strongly recommended."
        )

    prediction_int: int = 1 if probability > THRESHOLD else 0
    label: str          = "Diabetic" if prediction_int == 1 else "Non-Diabetic"

    log.info(
        "Prediction: %-14s | risk=%-8s | probability=%.4f | "
        "glucose=%.0f  bmi=%.1f  age=%.0f",
        label,
        risk_level,
        probability,
        patient.glucose,
        patient.bmi,
        patient.age,
    )

    return PredictionResponse(
        prediction     = prediction_int,
        probability    = round(probability, 4),
        risk_level     = risk_level,
        recommendation = recommendation,
        threshold      = THRESHOLD,
        model_version  = "FL-DiabetesNet v1.0",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get(
    "/health",
    summary = "API health check",
    tags    = ["System"],
)
async def health_check():
    """
    Returns whether the API is alive and whether the model is ready to serve.

    The Next.js frontend can poll this endpoint to display a
    "backend offline" warning banner when the API or model is unavailable.
    """
    model_ready  = _model  is not None
    scaler_ready = _scaler is not None

    return {
        "status":        "ok" if (model_ready and scaler_ready) else "degraded",
        "model_loaded":  model_ready,
        "scaler_loaded": scaler_ready,
        "device":        str(DEVICE),
        "model_path":    str(MODEL_PATH),
        "scaler_path":   str(SCALER_PATH),
    }


@app.post(
    "/predict",
    response_model = PredictionResponse,
    summary        = "Predict diabetes risk",
    tags           = ["Prediction"],
    responses      = {
        200: {"description": "Successful prediction"},
        422: {"description": "Validation error — invalid or out-of-range input field"},
        503: {"description": "Model not ready — run federated training first"},
        500: {"description": "Unexpected inference error"},
    },
)
async def predict_endpoint(patient: PatientInput):
    """
    Accept 8 clinical features and return a diabetes prediction.

    **Request body** — all fields required:

    | Field           | Unit    | Valid range |
    |-----------------|---------|-------------|
    | pregnancies     | count   | 0 – 20      |
    | glucose         | mg/dL   | 0 – 300     |
    | bloodPressure   | mm Hg   | 0 – 200     |
    | skinThickness   | mm      | 0 – 100     |
    | insulin         | µU/mL   | 0 – 900     |
    | bmi             | kg/m²   | 10 – 70     |
    | dpf             | score   | 0 – 3       |
    | age             | years   | 1 – 120     |

    **Response**:
    - `prediction`     : `1` (Diabetic) or `0` (Non-Diabetic)
    - `probability`    : sigmoid output in [0.0, 1.0]
    - `risk_level`     : `"Low"` (<30%) | `"Moderate"` (30–50%) | `"Elevated"` (50–70%) | `"High"` (≥70%)
    - `recommendation` : clinical guidance based on risk level
    - `threshold`      : decision threshold (0.5)
    - `model_version`  : model identifier string
    """

    # Guard: artifacts missing (training hasn't been run yet)
    if _model is None or _scaler is None:
        raise HTTPException(
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE,
            detail      = (
                "Model is not loaded. "
                "Run federated training first:  python run_project.py  "
                "then restart the API server:   uvicorn api:app --reload --port 8000"
            ),
        )

    # Run inference — catch unexpected numpy/torch errors gracefully
    try:
        result = predict_diabetes(patient)
    except Exception as exc:
        log.error("Inference error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail      = f"Inference failed: {exc}",
        )

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Dev entry point
# Run directly:  python api.py
# Production:    uvicorn api:app --reload --port 8000
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    print("\n" + "=" * 55)
    print("  FL Diabetes Prediction API")
    print("  http://localhost:8000")
    print("  Interactive docs  →  http://localhost:8000/docs")
    print("=" * 55 + "\n")

    uvicorn.run(
        "api:app",
        host      = "0.0.0.0",
        port      = 8000,
        reload    = True,
        log_level = "info",
    )
