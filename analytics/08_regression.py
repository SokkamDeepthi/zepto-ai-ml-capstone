import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# =========================================================
# 1. Paths
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "titanic.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)


# =========================================================
# 2. Load Titanic Dataset
# =========================================================

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("Shape:", df.shape)


# =========================================================
# 3. Select Features and Target
# =========================================================

# Fare is the regression target
y = df["fare"]

# Use other available passenger-related features
X = df[
    [
        "survived",
        "pclass",
        "sex",
        "age",
        "sibsp",
        "parch",
        "embarked"
    ]
].copy()


# =========================================================
# 4. Train-Test Split
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\n===== TRAIN / TEST SPLIT =====")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# =========================================================
# 5. Feature Groups
# =========================================================

numeric_features = [
    "survived",
    "pclass",
    "age",
    "sibsp",
    "parch"
]

categorical_features = [
    "sex",
    "embarked"
]


# =========================================================
# 6. Preprocessing
# =========================================================

numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_transformer,
            numeric_features
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ]
)


# =========================================================
# 7. Multivariate Linear Regression
# =========================================================

regression_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "regressor",
            LinearRegression()
        )
    ]
)


# =========================================================
# 8. Train Model
# =========================================================

regression_pipeline.fit(
    X_train,
    y_train
)

print("\nLinear Regression trained successfully!")


# =========================================================
# 9. Predictions
# =========================================================

y_pred = regression_pipeline.predict(
    X_test
)


# =========================================================
# 10. Regression Metrics
# =========================================================

mae = mean_absolute_error(
    y_test,
    y_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred
    )
)

r2 = r2_score(
    y_test,
    y_pred
)


# =========================================================
# 11. Adjusted R-Squared
# =========================================================

n = len(y_test)

# Number of predictors after preprocessing
X_test_transformed = (
    regression_pipeline
    .named_steps["preprocessor"]
    .transform(X_test)
)

p = X_test_transformed.shape[1]

if n - p - 1 > 0:
    adjusted_r2 = (
        1
        - ((1 - r2) * (n - 1))
        / (n - p - 1)
    )
else:
    adjusted_r2 = np.nan


# =========================================================
# 12. Print Results
# =========================================================

print("\n===== REGRESSION RESULTS =====")

print(f"MAE          : {mae:.4f}")
print(f"RMSE         : {rmse:.4f}")
print(f"R-Squared    : {r2:.4f}")
print(f"Adjusted R²  : {adjusted_r2:.4f}")

print(f"\nNumber of observations (n): {n}")
print(f"Number of predictors (p): {p}")


# =========================================================
# 13. Residuals
# =========================================================

residuals = y_test - y_pred


# =========================================================
# 14. Residual Plot
# =========================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    y_pred,
    residuals,
    alpha=0.6
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel("Predicted Fare")
plt.ylabel("Residuals")
plt.title("Residual Plot - Fare Regression")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "regression_residual_plot.png"
)

plt.close()


# =========================================================
# 15. Simple Heteroscedasticity Check
# =========================================================

# Compare residual spread in lower and higher
# predicted-fare groups.

residual_data = pd.DataFrame(
    {
        "Predicted": y_pred,
        "Residual": residuals
    }
)

median_prediction = residual_data[
    "Predicted"
].median()

low_group = residual_data[
    residual_data["Predicted"] <= median_prediction
]["Residual"]

high_group = residual_data[
    residual_data["Predicted"] > median_prediction
]["Residual"]


low_std = low_group.std()
high_std = high_group.std()

print("\n===== HETEROSCEDASTICITY CHECK =====")

print(
    f"Residual Std - Lower predicted fares: "
    f"{low_std:.4f}"
)

print(
    f"Residual Std - Higher predicted fares: "
    f"{high_std:.4f}"
)


if high_std > low_std * 1.25:
    conclusion = (
        "Residual spread increases at higher "
        "predicted fare values, suggesting "
        "heteroscedasticity."
    )
elif low_std > high_std * 1.25:
    conclusion = (
        "Residual spread is larger at lower "
        "predicted fare values, suggesting "
        "heteroscedasticity."
    )
else:
    conclusion = (
        "Residual spread is relatively similar "
        "across predicted fare values; strong "
        "heteroscedasticity is not evident from "
        "this check."
    )


print("\nConclusion:")
print(conclusion)


# =========================================================
# 16. Save Regression Results
# =========================================================

results = pd.DataFrame(
    {
        "Metric": [
            "MAE",
            "RMSE",
            "R-Squared",
            "Adjusted R-Squared"
        ],
        "Value": [
            mae,
            rmse,
            r2,
            adjusted_r2
        ]
    }
)


results.to_csv(
    OUTPUT_DIR / "regression_results.csv",
    index=False
)


# =========================================================
# 17. Save Regression Pipeline
# =========================================================

import joblib

MODEL_DIR = BASE_DIR.parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

joblib.dump(
    regression_pipeline,
    MODEL_DIR / "fare_regression_pipeline.pkl"
)


# =========================================================
# 18. Completion
# =========================================================

print("\n===== COMPLETED =====")

print(
    "Multivariate Linear Regression completed."
)

print(
    "Residual plot saved to:"
)

print(
    OUTPUT_DIR / "regression_residual_plot.png"
)

print(
    "Regression results saved to:"
)

print(
    OUTPUT_DIR / "regression_results.csv"
)

print(
    "Regression pipeline saved to:"
)

print(
    MODEL_DIR / "fare_regression_pipeline.pkl"
)