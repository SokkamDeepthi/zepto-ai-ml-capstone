import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt


# =========================================================
# 1. Load cleaned Titanic dataset
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "titanic.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("Shape:", df.shape)


# =========================================================
# 2. Select features and target
# =========================================================

# Target
y = df["survived"]

# Features
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


# =========================================================
# 3. Stratified Train-Test Split
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n===== TRAIN / TEST SPLIT =====")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

print("\nOverall class distribution:")
print(y.value_counts(normalize=True))

print("\nTraining class distribution:")
print(y_train.value_counts(normalize=True))

print("\nTesting class distribution:")
print(y_test.value_counts(normalize=True))


# =========================================================
# 4. Preprocessing
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
# 5. Three Classification Models
# =========================================================

models = {

    "Logistic Regression":
        LogisticRegression(
            max_iter=1000,
            random_state=42
        ),

    "Decision Tree":
        DecisionTreeClassifier(
            max_depth=5,
            random_state=42
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=200,
            random_state=42
        )
}


results = []

trained_pipelines = {}


# =========================================================
# 6. Train and Evaluate Three Models
# =========================================================

print("\n===== MODEL RESULTS =====")

for model_name, model in models.items():

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "classifier",
                model
            )
        ]
    )

    # Fit ONLY on training data
    pipeline.fit(X_train, y_train)

    # Prediction
    y_pred = pipeline.predict(X_test)

    # Probability for ROC-AUC
    y_probability = pipeline.predict_proba(X_test)[:, 1]

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

    results.append(
        {
            "Model": model_name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "ROC_AUC": roc_auc
        }
    )

    trained_pipelines[model_name] = pipeline

    print(f"\n{model_name}")
    print("-" * 30)
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )


# =========================================================
# 7. Comparison Table
# =========================================================

results_df = pd.DataFrame(results)

print("\n===== MODEL COMPARISON =====")
print(results_df.to_string(index=False))

results_df.to_csv(
    OUTPUT_DIR / "model_comparison.csv",
    index=False
)


# =========================================================
# 8. Confusion Matrix for All Three Models
# =========================================================

for model_name, pipeline in trained_pipelines.items():

    y_pred = pipeline.predict(X_test)

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    print(f"\n===== CONFUSION MATRIX: {model_name} =====")
    print(cm)

    safe_name = (
        model_name
        .lower()
        .replace(" ", "_")
    )

    plt.figure(figsize=(6, 5))

    plt.imshow(cm)

    plt.title(
        f"Confusion Matrix - {model_name}"
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.xticks(
        [0, 1],
        ["Did Not Survive", "Survived"]
    )

    plt.yticks(
        [0, 1],
        ["Did Not Survive", "Survived"]
    )

    for i in range(2):
        for j in range(2):
            plt.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center"
            )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR /
        f"{safe_name}_confusion_matrix.png"
    )

    plt.close()


# =========================================================
# 9. Decision Tree Visualization
# =========================================================

decision_tree_pipeline = trained_pipelines[
    "Decision Tree"
]

decision_tree_model = (
    decision_tree_pipeline
    .named_steps["classifier"]
)

fitted_preprocessor = (
    decision_tree_pipeline
    .named_steps["preprocessor"]
)

feature_names = (
    fitted_preprocessor
    .get_feature_names_out()
)

plt.figure(figsize=(22, 12))

plot_tree(
    decision_tree_model,
    feature_names=feature_names,
    class_names=[
        "Did Not Survive",
        "Survived"
    ],
    filled=True,
    rounded=True,
    max_depth=4
)

plt.title("Decision Tree - Titanic Survival")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "decision_tree.png"
)

plt.close()


# =========================================================
# 10. Random Forest Feature Importance
# =========================================================

rf_pipeline = trained_pipelines[
    "Random Forest"
]

rf_model = (
    rf_pipeline
    .named_steps["classifier"]
)

rf_preprocessor = (
    rf_pipeline
    .named_steps["preprocessor"]
)

rf_feature_names = (
    rf_preprocessor
    .get_feature_names_out()
)

feature_importance = pd.DataFrame(
    {
        "Feature": rf_feature_names,
        "Importance": rf_model.feature_importances_
    }
).sort_values(
    "Importance",
    ascending=False
)

print("\n===== RANDOM FOREST FEATURE IMPORTANCE =====")
print(
    feature_importance
    .head(10)
    .to_string(index=False)
)

feature_importance.to_csv(
    OUTPUT_DIR / "feature_importance.csv",
    index=False
)

plt.figure(figsize=(10, 6))

top_features = feature_importance.head(10)

plt.barh(
    top_features["Feature"][::-1],
    top_features["Importance"][::-1]
)

plt.title(
    "Top 10 Random Forest Feature Importances"
)

plt.xlabel("Importance")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "feature_importance.png"
)

plt.close()


# =========================================================
# 11. Save the three fitted pipelines
# =========================================================

import joblib

joblib.dump(
    trained_pipelines["Logistic Regression"],
    BASE_DIR.parent / "models" / "logistic_pipeline.pkl"
)

joblib.dump(
    trained_pipelines["Decision Tree"],
    BASE_DIR.parent / "models" / "decision_tree_pipeline.pkl"
)

joblib.dump(
    trained_pipelines["Random Forest"],
    BASE_DIR.parent / "models" / "random_forest_pipeline.pkl"
)


print("\n===== COMPLETED =====")
print("Three classification pipelines trained successfully.")
print("Pipelines saved in models/ folder.")
print("Charts and comparison files saved in analytics/outputs/")