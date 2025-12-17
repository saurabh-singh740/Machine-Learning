import torch
import joblib
from model import DiabetesNet

# -------------------------------
# Load trained global model
# -------------------------------
model = DiabetesNet()
model.load_state_dict(torch.load("global_model.pth"))
model.eval()

# -------------------------------
# Load the SAME scaler used during training
# -------------------------------
scaler = joblib.load("scaler.pkl")

# ---- USER INPUT ----
print("\nInput Patient Data:")

glucose = float(input("Glucose = "))
bmi = float(input("BMI = "))
age = float(input("Age = "))

# Fixed / demo values for remaining features
pregnancies = 2
bp = 72
skin = 35
insulin = 0
dpf = 0.62

# -------------------------------
# Create input array (RAW values)
# Order MUST match training data
# -------------------------------
sample = [[
    pregnancies,
    glucose,
    bp,
    skin,
    insulin,
    bmi,
    dpf,
    age
]]

# -------------------------------
# Apply SAME standardization
# -------------------------------
sample = scaler.transform(sample)
sample = torch.tensor(sample, dtype=torch.float32)

# -------------------------------
# Prediction
# -------------------------------
with torch.no_grad():
    prob = model(sample).item()

print("\nPrediction:")
if prob > 0.5:
    print(f"Diabetes Detected (Probability: {prob:.2f})")
else:
    print(f"No Diabetes Detected (Probability: {prob:.2f})")
