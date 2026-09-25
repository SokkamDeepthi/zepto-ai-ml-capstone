import pandas as pd
import joblib

from pathlib import Path

# Project folder
BASE_DIR = Path(__file__).resolve().parent.parent

# Load saved model and encoder
MODEL_PATH = BASE_DIR / "models" / "titanic_model.pkl"
ENCODER_PATH = BASE_DIR / "models" / "sex_encoder.pkl"

model = joblib.load(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)

# New passenger data
passenger = pd.DataFrame({
    "pclass": [3],
    "sex": ["female"],
    "age": [25],
    "fare": [20]
})

# Encode sex
passenger["sex"] = encoder.transform(passenger["sex"])

# Prediction
prediction = model.predict(passenger)[0]

# Probability
probability = model.predict_proba(passenger)[0]

print("Passenger Details:")
print(passenger)

print("\nPrediction:")
if prediction == 1:
    print("Survived")
else:
    print("Did Not Survive")

print("\nPrediction Probability:")
print("Did Not Survive:", round(probability[0] * 100, 2), "%")
print("Survived:", round(probability[1] * 100, 2), "%")