import torch
from model import DiabetesNet

# Load trained global model
model = DiabetesNet()
model.load_state_dict(torch.load("global_model.pth"))
model.eval()

# ---- USER INPUT ----
print("\nInput Patient Data:")

glucose = float(input("Glucose = "))
bmi = float(input("BMI = "))
age = float(input("Age = "))

# Dummy values for remaining features (safe for demo)
bp = 72
skin = 35
insulin = 0
dpf = 0.62
pregnancies = 2

# Create input tensor (same order as dataset)
sample = torch.tensor(
    [[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]],
    dtype=torch.float32
)

# Prediction
with torch.no_grad():
    prob = model(sample).item()

print("\nPrediction:")
if prob > 0.5:
    print(f"Diabetes Detected (Probability: {prob:.2f})")
else:
    print(f"No Diabetes Detected (Probability: {prob:.2f})")
