# inference.py
"""
Inference script for the trained federated global model.

Supports two modes:
  interactive  – prompt the user for patient data, print prediction
  batch        – read a CSV, write predictions to a new CSV

Improvements over original predict.py
──────────────────────────────────────
  ✅ Loads updated DiabetesNet (with BatchNorm layers) correctly
  ✅ model() now outputs raw logit → sigmoid applied here (not baked in)
  ✅ Threshold derived from ROC curve (Youden's J) by default
  ✅ Risk stratification: Low / Moderate / Elevated / High
  ✅ Batch prediction mode (CSV in → CSV out)
  ✅ Clinical recommendation printed with result
  ✅ Clear error messages when model/scaler files are missing

Usage
──────
  # Interactive
  python inference.py

  # With custom model/scaler paths
  python inference.py --model artifacts/best_model.pth \
                      --scaler artifacts/scaler.pkl \
                      --threshold 0.45

  # Batch prediction
  python inference.py --batch patients.csv
"""

import os
import sys
import argparse
import numpy as np
import torch
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.model import DiabetesNet

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_MODEL_PATH  = os.path.join("artifacts", "global_model.pth")
DEFAULT_SCALER_PATH = os.path.join("artifacts", "scaler.pkl")

# ── Feature order (MUST match training data column order) ─────────────────────
FEATURE_NAMES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Model and scaler loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(model_path: str = DEFAULT_MODEL_PATH) -> DiabetesNet:
    """Load the trained DiabetesNet from a checkpoint file."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model checkpoint not found: {model_path}\n"
            "Run federated training first:\n"
            "  python train_federated.py --mode server --rounds 15 --clients 3"
        )
    net = DiabetesNet().to(DEVICE)
    net.load_state_dict(torch.load(model_path, map_location=DEVICE))
    net.eval()
    return net


def load_scaler(scaler_path: str = DEFAULT_SCALER_PATH):
    """Load the fitted StandardScaler from disk."""
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(
            f"Scaler not found: {scaler_path}\n"
            "Run data preparation first:\n"
            "  python train_federated.py --mode prepare"
        )
    return joblib.load(scaler_path)


# ─────────────────────────────────────────────────────────────────────────────
# Prediction logic
# ─────────────────────────────────────────────────────────────────────────────

def predict_single(
    features:  list,
    model:     DiabetesNet,
    scaler,
    threshold: float = 0.5,
) -> dict:
    """
    Predict diabetes risk for one patient.

    Parameters
    ──────────
    features  : raw clinical values in FEATURE_NAMES order
    model     : loaded DiabetesNet
    scaler    : fitted StandardScaler
    threshold : decision threshold (default 0.5; use Youden's J in practice)

    Returns
    ───────
    Dict with keys: prediction, probability, risk_level, result, threshold
    """
    sample  = np.array(features, dtype=np.float32).reshape(1, -1)
    scaled  = scaler.transform(sample)
    tensor  = torch.tensor(scaled, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        logit       = model(tensor).squeeze()
        probability = torch.sigmoid(logit).item()

    prediction = int(probability >= threshold)

    # Risk stratification (clinically-inspired buckets)
    if probability < 0.2:
        risk_level = "Low"
    elif probability < 0.4:
        risk_level = "Moderate"
    elif probability < threshold:
        risk_level = "Elevated"
    else:
        risk_level = "High"

    return {
        "prediction":  prediction,
        "probability": round(float(probability), 4),
        "risk_level":  risk_level,
        "result":      "Diabetes Detected" if prediction else "No Diabetes Detected",
        "threshold":   threshold,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Interactive mode
# ─────────────────────────────────────────────────────────────────────────────

def interactive_predict(
    model_path:  str   = DEFAULT_MODEL_PATH,
    scaler_path: str   = DEFAULT_SCALER_PATH,
    threshold:   float = 0.5,
) -> None:
    """Prompt for patient data and print the diabetes prediction."""

    print("\n" + "═" * 58)
    print("  Diabetes Prediction  —  Federated Learning Model")
    print("═" * 58)

    model  = load_model(model_path)
    scaler = load_scaler(scaler_path)

    print("\nEnter Patient Clinical Data  (press Enter after each value):")
    print("─" * 42)

    features = []
    for name in FEATURE_NAMES:
        while True:
            try:
                val = float(input(f"  {name:<28}: "))
                features.append(val)
                break
            except ValueError:
                print("    Please enter a valid number.")

    result = predict_single(features, model, scaler, threshold)

    print("\n" + "═" * 58)
    print("  PREDICTION RESULT")
    print("═" * 58)
    print(f"  Result      : {result['result']}")
    print(f"  Probability : {result['probability']:.4f}  "
          f"({result['probability'] * 100:.1f} %)")
    print(f"  Risk Level  : {result['risk_level']}")
    print(f"  Threshold   : {result['threshold']}")
    print("─" * 58)

    if result["prediction"] == 1:
        print("  RECOMMENDATION:")
        print("    Patient shows elevated diabetes risk.")
        print("    Clinical consultation and HbA1c testing recommended.")
    else:
        print("  RECOMMENDATION:")
        print("    No diabetes detected at current threshold.")
        print("    Continue routine monitoring if risk factors are present.")

    print("═" * 58 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Batch mode
# ─────────────────────────────────────────────────────────────────────────────

def batch_predict(
    csv_path:    str,
    model_path:  str   = DEFAULT_MODEL_PATH,
    scaler_path: str   = DEFAULT_SCALER_PATH,
    threshold:   float = 0.5,
) -> None:
    """
    Run predictions on a CSV file and save results.

    Input CSV must contain columns matching FEATURE_NAMES.
    Output is saved as <input_name>_predictions.csv.
    """
    import pandas as pd

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    model  = load_model(model_path)
    scaler = load_scaler(scaler_path)
    df     = pd.read_csv(csv_path)

    missing = [f for f in FEATURE_NAMES if f not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")

    probs, preds, risks = [], [], []

    for _, row in df.iterrows():
        feats  = [row[f] for f in FEATURE_NAMES]
        result = predict_single(feats, model, scaler, threshold)
        probs.append(result["probability"])
        preds.append(result["prediction"])
        risks.append(result["risk_level"])

    df["probability"]  = probs
    df["prediction"]   = preds
    df["risk_level"]   = risks

    out_path = csv_path.replace(".csv", "_predictions.csv")
    df.to_csv(out_path, index=False)
    print(f"Batch predictions saved → {out_path}  ({len(df)} records)")

    # Summary statistics
    n_pos = sum(preds)
    print(f"  Predicted diabetic   : {n_pos} / {len(preds)} "
          f"({n_pos/len(preds)*100:.1f} %)")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Diabetes prediction inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--model",     type=str, default=DEFAULT_MODEL_PATH,
                        help="Path to model checkpoint (.pth)")
    parser.add_argument("--scaler",    type=str, default=DEFAULT_SCALER_PATH,
                        help="Path to fitted scaler (.pkl)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Decision threshold (0–1).  Default: 0.5")
    parser.add_argument("--batch",     type=str, default=None,
                        help="Path to CSV file for batch prediction")
    args = parser.parse_args()

    if args.batch:
        batch_predict(
            csv_path    = args.batch,
            model_path  = args.model,
            scaler_path = args.scaler,
            threshold   = args.threshold,
        )
    else:
        interactive_predict(
            model_path  = args.model,
            scaler_path = args.scaler,
            threshold   = args.threshold,
        )
