from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier


# --------------------------------------------------
# 1. Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "outputs" / "titanic_cleaned.csv"

OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = BASE_DIR.parent / "models"

OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# 2. Load cleaned dataset
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("TITANIC CLASSIFICATION MODELING")
print("=" * 60)

print("\nDataset shape:", df.shape)


# --------------------------------------------------
# 3. Select features
# --------------------------------------------------

target = "survived"

features = [
    "pclass",
    "sex",
    "age",
    "sibsp",
    "parch",
    "fare",
    "embarked",
]

X = df[features]
y = df[target]


# --------------------------------------------------
# 4. Train-test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# --------------------------------------------------
# 5. Preprocessing
# --------------------------------------------------

numeric_features = [
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare",
]

categorical_features = [
    "sex",
    "embarked",
]


numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ]
)


categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent"),
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
        ),
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)


# --------------------------------------------------
# 6. Define models
# --------------------------------------------------

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42,
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=5,
        random_state=42,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=42,
    ),
}


# --------------------------------------------------
# 7. Train and evaluate models
# --------------------------------------------------

results = []
trained_pipelines = {}


for model_name, model in models.items():

    print("\n" + "-" * 60)
    print("Training:", model_name)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    cv_scores = cross_val_score(
        pipeline,
        X,
        y,
        cv=5,
        scoring="f1",
    )

    cv_f1 = cv_scores.mean()

    results.append(
        {
            "model": model_name,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "cv_f1_mean": cv_f1,
        }
    )

    trained_pipelines[model_name] = pipeline

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"CV F1    : {cv_f1:.4f}")


# --------------------------------------------------
# 8. Model comparison
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="f1_score",
    ascending=False,
)

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(results_df.to_string(index=False))


results_df.to_csv(
    OUTPUT_DIR / "model_comparison.csv",
    index=False,
)


# --------------------------------------------------
# 9. Select model for deployment
# --------------------------------------------------

best_model_name = results_df.iloc[0]["model"]

best_pipeline = trained_pipelines[best_model_name]

print("\nSelected model:", best_model_name)


# --------------------------------------------------
# 10. Save model
# --------------------------------------------------

model_path = MODEL_DIR / "titanic_model.pkl"

joblib.dump(
    best_pipeline,
    model_path,
)

print("\nModel saved to:")
print(model_path)


# --------------------------------------------------
# 11. Save model information
# --------------------------------------------------

model_info = pd.DataFrame(
    {
        "selected_model": [best_model_name],
        "accuracy": [
            results_df.iloc[0]["accuracy"]
        ],
        "precision": [
            results_df.iloc[0]["precision"]
        ],
        "recall": [
            results_df.iloc[0]["recall"]
        ],
        "f1_score": [
            results_df.iloc[0]["f1_score"]
        ],
        "cv_f1_mean": [
            results_df.iloc[0]["cv_f1_mean"]
        ],
    }
)

model_info.to_csv(
    OUTPUT_DIR / "selected_model.csv",
    index=False,
)


# --------------------------------------------------
# 12. Final message
# --------------------------------------------------

print("\n" + "=" * 60)
print("MODELING COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nGenerated files:")
print("- model_comparison.csv")
print("- selected_model.csv")
print("- titanic_model.pkl")