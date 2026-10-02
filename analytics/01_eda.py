from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# --------------------------------------------------
# 1. Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "titanic.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# 2. Load dataset
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("TITANIC DATASET - EDA")
print("=" * 60)

print("\nDataset shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())


# --------------------------------------------------
# 3. Data types
# --------------------------------------------------

print("\n" + "=" * 60)
print("DATA TYPES")
print("=" * 60)

print(df.dtypes)


# --------------------------------------------------
# 4. Missing values
# --------------------------------------------------

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)

missing = df.isnull().sum()

print(missing)

missing.to_csv(OUTPUT_DIR / "missing_values.csv", header=["missing_count"])


# --------------------------------------------------
# 5. Basic statistics
# --------------------------------------------------

print("\n" + "=" * 60)
print("DESCRIPTIVE STATISTICS")
print("=" * 60)

print(df.describe())

df.describe().to_csv(OUTPUT_DIR / "descriptive_statistics.csv")


# --------------------------------------------------
# 6. Target distribution
# --------------------------------------------------

print("\n" + "=" * 60)
print("SURVIVAL DISTRIBUTION")
print("=" * 60)

print(df["survived"].value_counts())

print("\nSurvival percentage:")
print(df["survived"].value_counts(normalize=True) * 100)


# --------------------------------------------------
# 7. Data cleaning
# --------------------------------------------------

# Make a copy so the original dataset remains unchanged
clean_df = df.copy()

# Fill missing numerical values with median
clean_df["age"] = clean_df["age"].fillna(clean_df["age"].median())
clean_df["fare"] = clean_df["fare"].fillna(clean_df["fare"].median())

# Fill missing categorical values with mode
clean_df["embarked"] = clean_df["embarked"].fillna(
    clean_df["embarked"].mode()[0]
)

# Drop columns with excessive missing values
clean_df = clean_df.drop(columns=["deck"])

print("\n" + "=" * 60)
print("AFTER CLEANING")
print("=" * 60)

print("Shape:", clean_df.shape)

print("\nRemaining missing values:")
print(clean_df.isnull().sum())


# Save cleaned dataset
clean_df.to_csv(OUTPUT_DIR / "titanic_cleaned.csv", index=False)


# --------------------------------------------------
# 8. Visualization - Survival count
# --------------------------------------------------

plt.figure(figsize=(7, 5))

sns.countplot(data=clean_df, x="survived")

plt.title("Titanic Survival Distribution")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Number of Passengers")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "survival_distribution.png")
plt.close()


# --------------------------------------------------
# 9. Visualization - Survival by gender
# --------------------------------------------------

plt.figure(figsize=(7, 5))

sns.countplot(data=clean_df, x="sex", hue="survived")

plt.title("Survival by Gender")
plt.xlabel("Gender")
plt.ylabel("Number of Passengers")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "survival_by_gender.png")
plt.close()


# --------------------------------------------------
# 10. Visualization - Survival by passenger class
# --------------------------------------------------

plt.figure(figsize=(7, 5))

sns.countplot(data=clean_df, x="pclass", hue="survived")

plt.title("Survival by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Number of Passengers")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "survival_by_class.png")
plt.close()


# --------------------------------------------------
# 11. Age distribution
# --------------------------------------------------

plt.figure(figsize=(8, 5))

sns.histplot(data=clean_df, x="age", bins=30, kde=True)

plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Passengers")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "age_distribution.png")
plt.close()


# --------------------------------------------------
# 12. Fare distribution
# --------------------------------------------------

plt.figure(figsize=(8, 5))

sns.histplot(data=clean_df, x="fare", bins=30, kde=True)

plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.ylabel("Number of Passengers")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fare_distribution.png")
plt.close()


# --------------------------------------------------
# 13. Correlation heatmap
# --------------------------------------------------

numeric_df = clean_df.select_dtypes(include="number")

plt.figure(figsize=(9, 7))

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "correlation_heatmap.png")
plt.close()


# --------------------------------------------------
# 14. Final summary
# --------------------------------------------------

print("\n" + "=" * 60)
print("EDA COMPLETED SUCCESSFULLY")
print("=" * 60)

print("Cleaned dataset:")
print(OUTPUT_DIR / "titanic_cleaned.csv")

print("\nCharts saved in:")
print(OUTPUT_DIR)

print("\nGenerated files:")

for file in sorted(OUTPUT_DIR.iterdir()):
    print("-", file.name)