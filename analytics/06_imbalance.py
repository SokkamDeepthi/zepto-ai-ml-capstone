import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


# =========================================================
# 1. Load Dataset
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "titanic.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)


# =========================================================
# 2. Features and Target
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
# 3. Stratified Train-Test Split
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\n===== CLASS BALANCE =====")

print("\nOriginal dataset:")
print(y.value_counts())

print("\nOriginal percentages:")
print(
    (y.value_counts(normalize=True) * 100).round(2)
)

print("\nTraining dataset:")
print(y_train.value_counts())

print("\nTest dataset:")
print(y_test.value_counts())


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
# 5. Three Random Forest Strategies
# =========================================================

models = {

    "Baseline":
        RandomForestClassifier(
            n_estimators=200,
            random_state=42
        ),

    "Class Weight Balanced":
        RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42
        )
}


results = []


# =========================================================
# 6. Baseline and Class Weight Balanced
# =========================================================

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

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

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

    results.append(
        {
            "Strategy": model_name,
            "Precision": precision,
            "Recall": recall,
            "F1": f1
        }
    )


# =========================================================
# 7. SMOTE
# =========================================================

smote_pipeline = ImbPipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "smote",
            SMOTE(
                random_state=42
            )
        ),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                random_state=42
            )
        )
    ]
)


# SMOTE is fitted only on training data
smote_pipeline.fit(
    X_train,
    y_train
)


y_pred_smote = smote_pipeline.predict(
    X_test
)


precision_smote = precision_score(
    y_test,
    y_pred_smote,
    zero_division=0
)

recall_smote = recall_score(
    y_test,
    y_pred_smote,
    zero_division=0
)

f1_smote = f1_score(
    y_test,
    y_pred_smote,
    zero_division=0
)


results.append(
    {
        "Strategy": "SMOTE",
        "Precision": precision_smote,
        "Recall": recall_smote,
        "F1": f1_smote
    }
)


# =========================================================
# 8. Comparison
# =========================================================

results_df = pd.DataFrame(results)

print("\n===== IMBALANCE COMPARISON =====")

print(
    results_df.to_string(index=False)
)


# Save results
results_df.to_csv(
    OUTPUT_DIR / "imbalance_comparison.csv",
    index=False
)


# =========================================================
# 9. Training Class Distribution After SMOTE
# =========================================================

# Transform training data using fitted preprocessor
X_train_processed = preprocessor.fit_transform(
    X_train
)

smote = SMOTE(
    random_state=42
)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train_processed,
    y_train
)

print("\n===== AFTER SMOTE =====")

print(
    pd.Series(y_train_smote).value_counts()
)

print("\nSMOTE completed only on training data.")

print("\n===== COMPLETED =====")
print(
    "Baseline, Class Weight Balanced and SMOTE comparison completed."
)
print(
    "Results saved to analytics/outputs/imbalance_comparison.csv"
)