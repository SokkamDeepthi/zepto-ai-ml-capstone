import pandas as pd
import joblib
from pathlib import Path


# =========================================================
# 1. Paths
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "best_random_forest_pipeline.pkl"


# =========================================================
# 2. Load Complete Fitted Pipeline
# =========================================================

pipeline = joblib.load(MODEL_PATH)

print("===== MODEL RELOADED =====")
print("Model:", MODEL_PATH)


# =========================================================
# 3. Raw Input
# =========================================================

raw_input = pd.DataFrame([
    {
        "pclass": 3,
        "sex": "female",
        "age": 25,
        "sibsp": 0,
        "parch": 0,
        "fare": 20.0,
        "embarked": "S"
    }
])


print("\n===== RAW INPUT =====")
print(raw_input)


# =========================================================
# 4. Prediction
# =========================================================

prediction = pipeline.predict(raw_input)
probability = pipeline.predict_proba(raw_input)


# =========================================================
# 5. Display Result
# =========================================================

print("\n===== PREDICTION =====")

if prediction[0] == 1:
    print("Predicted Survival: Yes")
else:
    print("Predicted Survival: No")

print(f"Survival Probability: {probability[0][1]:.4f}")
print(f"Non-Survival Probability: {probability[0][0]:.4f}")


# =========================================================
# 6. Completion
# =========================================================

print("\n===== COMPLETED =====")
print("Complete fitted pipeline successfully reloaded.")
print("Prediction successfully generated from raw input.")