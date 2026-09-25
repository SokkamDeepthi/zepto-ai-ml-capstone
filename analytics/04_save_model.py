import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Project folder
BASE_DIR = Path(__file__).resolve().parent.parent

# Load dataset
DATA_PATH = BASE_DIR / "analytics" / "titanic.csv"
df = pd.read_csv(DATA_PATH)

# Select columns
df = df[["survived", "pclass", "sex", "age", "fare"]]

# Handle missing values
df["age"] = df["age"].fillna(df["age"].median())

# Convert sex to numbers
le = LabelEncoder()
df["sex"] = le.fit_transform(df["sex"])

# Features and target
X = df[["pclass", "sex", "age", "fare"]]
y = df["survived"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Create models folder
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

# Save model
joblib.dump(model, MODEL_DIR / "titanic_model.pkl")

# Save encoder too
joblib.dump(le, MODEL_DIR / "sex_encoder.pkl")

print("Model trained successfully!")
print("Model saved to:", MODEL_DIR / "titanic_model.pkl")
print("Encoder saved to:", MODEL_DIR / "sex_encoder.pkl")