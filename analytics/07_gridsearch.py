import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# =========================================================
# 1. Paths
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "titanic.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)


# =========================================================
# 2. Load Dataset
# =========================================================

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("Shape:", df.shape)


# =========================================================
# 3. Features and Target
# =========================================================

X = df[
    [
        "pclass",
        "sex",
        "age",
        "sibsp",
        "parch",
        "fare",
        "embarked"
    ]
].copy()

y = df["survived"]


# =========================================================
# 4. Stratified Train-Test Split
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n===== DATA SPLIT =====")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# =========================================================
# 5. Preprocessing
# =========================================================

numeric_features = [
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare"
]

categorical_features = [
    "sex",
    "embarked"
]


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
            OneHotEncoder(handle_unknown="ignore")
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
# 6. Random Forest with OOB Score
# =========================================================

rf = RandomForestClassifier(
    random_state=42,
    oob_score=True,
    n_jobs=-1
)


pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            rf
        )
    ]
)


# =========================================================
# 7. GridSearchCV Parameter Grid
# =========================================================

param_grid = {
    "classifier__n_estimators": [
        100,
        200,
        300
    ],

    "classifier__max_depth": [
        None,
        5,
        10
    ],

    "classifier__max_features": [
        "sqrt",
        "log2"
    ]
}


# =========================================================
# 8. GridSearchCV
# =========================================================

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1,
    verbose=1
)


print("\n===== STARTING GRID SEARCH =====")

grid_search.fit(
    X_train,
    y_train
)


# =========================================================
# 9. Best Parameters
# =========================================================

print("\n===== BEST PARAMETERS =====")

print(
    grid_search.best_params_
)

print("\nBest Cross-Validation F1 Score:")

print(
    f"{grid_search.best_score_:.4f}"
)


# =========================================================
# 10. Best Pipeline
# =========================================================

best_pipeline = grid_search.best_estimator_

best_rf = (
    best_pipeline
    .named_steps["classifier"]
)


# =========================================================
# 11. OOB Score
# =========================================================

print("\n===== OOB SCORE =====")

print(
    f"OOB Score: {best_rf.oob_score_:.4f}"
)


# =========================================================
# 12. Test Set Evaluation
# =========================================================

y_pred = best_pipeline.predict(
    X_test
)

y_probability = best_pipeline.predict_proba(
    X_test
)[:, 1]


accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)


print("\n===== BEST RANDOM FOREST TEST RESULTS =====")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")


# =========================================================
# 13. Save Results
# =========================================================

results = pd.DataFrame(
    {
        "Metric": [
            "Best CV F1",
            "OOB Score",
            "Test Accuracy",
            "Test Precision",
            "Test Recall",
            "Test F1",
            "Test ROC-AUC"
        ],

        "Value": [
            grid_search.best_score_,
            best_rf.oob_score_,
            accuracy,
            precision,
            recall,
            f1,
            roc_auc
        ]
    }
)


results.to_csv(
    OUTPUT_DIR / "gridsearch_results.csv",
    index=False
)


# =========================================================
# 14. Save Best Pipeline
# =========================================================

import joblib

MODEL_DIR = BASE_DIR.parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

joblib.dump(
    best_pipeline,
    MODEL_DIR / "best_random_forest_pipeline.pkl"
)


print("\n===== COMPLETED =====")

print(
    "GridSearchCV completed successfully."
)

print(
    "Best pipeline saved to:"
)

print(
    MODEL_DIR / "best_random_forest_pipeline.pkl"
)

print(
    "Results saved to:"
)

print(
    OUTPUT_DIR / "gridsearch_results.csv"
)