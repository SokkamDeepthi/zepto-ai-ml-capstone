import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

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

# Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Model Evaluation")
print("----------------")
print("Accuracy:", accuracy)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Not Survived", "Survived"]
)

disp.plot()
plt.title("Titanic Model - Confusion Matrix")
plt.tight_layout()

# Save confusion matrix
OUTPUT_DIR = BASE_DIR / "analytics" / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

plt.savefig(OUTPUT_DIR / "confusion_matrix.png")
plt.show()

# Feature importance
importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=True)

importance.plot(kind="barh")
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.tight_layout()

# Save feature importance
plt.savefig(OUTPUT_DIR / "feature_importance.png")
plt.show()

print("\nEvaluation completed successfully!")
print("Charts saved in:", OUTPUT_DIR)