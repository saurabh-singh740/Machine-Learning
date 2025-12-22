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
# Load SAME scaler used in training
# -------------------------------
scaler = joblib.load("scaler.pkl")

# -------------------------------
# USER INPUT (ALL 8 FEATURES)
# -------------------------------
print("\nEnter Patient Clinical Data:")

pregnancies = float(input("Pregnancies = "))
glucose = float(input("Glucose = "))
bp = float(input("Blood Pressure = "))
skin = float(input("Skin Thickness = "))
insulin = float(input("Insulin = "))
bmi = float(input("BMI = "))
dpf = float(input("Diabetes Pedigree Function = "))
age = float(input("Age = "))

# -------------------------------
# Create input array (RAW values)
# ORDER MUST MATCH training data
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
threshold = 0.4

with torch.no_grad():
    prob = model(sample).item()

prob = round(prob, 2)

print("\nPrediction Result:")
if prob >= threshold:
    print(f"Diabetes Detected (Probability: {prob:.2f})")
else:
    print(f"No Diabetes Detected (Probability: {prob:.2f})")
