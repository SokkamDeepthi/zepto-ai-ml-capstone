import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ---------------------------------------------------------
# 1. Load Titanic dataset
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "titanic.csv"
CHART_DIR = BASE_DIR / "charts"

CHART_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)

print("\n===== DATASET PROFILE =====")
print("\nShape:")
print(df.shape)

print("\nInfo:")
df.info()

print("\nDescribe:")
print(df.describe(include="all"))

# ---------------------------------------------------------
# 2. Missing value analysis
# ---------------------------------------------------------

print("\n===== MISSING VALUE PERCENTAGES =====")

missing_percent = (df.isnull().mean() * 100).round(2)

missing_table = pd.DataFrame({
    "Missing_Count": df.isnull().sum(),
    "Missing_Percentage": missing_percent
})

print(missing_table[missing_table["Missing_Count"] > 0])

# ---------------------------------------------------------
# 3. Cleaning
# ---------------------------------------------------------

cleaned_df = df.copy()

# Age: missing percentage is between 5% and 30%
# Therefore, median imputation is used.
cleaned_df["age"] = cleaned_df["age"].fillna(
    cleaned_df["age"].median()
)

# Embarked: very small percentage of missing values
# Mode imputation is used.
cleaned_df["embarked"] = cleaned_df["embarked"].fillna(
    cleaned_df["embarked"].mode()[0]
)

# ---------------------------------------------------------
# 4. Univariate Analysis - Age
# ---------------------------------------------------------

plt.figure(figsize=(8, 5))
sns.histplot(cleaned_df["age"], bins=30, kde=True)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(CHART_DIR / "age_histogram.png")
plt.close()

# Age box plot
plt.figure(figsize=(8, 5))
sns.boxplot(x=cleaned_df["age"])
plt.title("Age Box Plot")
plt.xlabel("Age")
plt.tight_layout()
plt.savefig(CHART_DIR / "age_boxplot.png")
plt.close()

# ---------------------------------------------------------
# 5. Univariate Analysis - Fare
# ---------------------------------------------------------

plt.figure(figsize=(8, 5))
sns.histplot(cleaned_df["fare"], bins=30, kde=True)
plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(CHART_DIR / "fare_histogram.png")
plt.close()

# Fare box plot
plt.figure(figsize=(8, 5))
sns.boxplot(x=cleaned_df["fare"])
plt.title("Fare Box Plot")
plt.xlabel("Fare")
plt.tight_layout()
plt.savefig(CHART_DIR / "fare_boxplot.png")
plt.close()

# ---------------------------------------------------------
# 6. IQR Outlier Counts
# ---------------------------------------------------------

def iqr_outlier_count(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = ((series < lower_bound) | (series > upper_bound)).sum()

    return outliers, lower_bound, upper_bound


age_outliers, age_lower, age_upper = iqr_outlier_count(
    cleaned_df["age"]
)

fare_outliers, fare_lower, fare_upper = iqr_outlier_count(
    cleaned_df["fare"]
)

print("\n===== IQR OUTLIERS =====")
print(f"Age outliers: {age_outliers}")
print(f"Age lower bound: {age_lower:.2f}")
print(f"Age upper bound: {age_upper:.2f}")

print(f"\nFare outliers: {fare_outliers}")
print(f"Fare lower bound: {fare_lower:.2f}")
print(f"Fare upper bound: {fare_upper:.2f}")

# ---------------------------------------------------------
# 7. Fare Mean / Median / Mode
# ---------------------------------------------------------

fare_mean = cleaned_df["fare"].mean()
fare_median = cleaned_df["fare"].median()
fare_mode = cleaned_df["fare"].mode()[0]

print("\n===== FARE STATISTICS =====")
print(f"Fare Mean: {fare_mean:.2f}")
print(f"Fare Median: {fare_median:.2f}")
print(f"Fare Mode: {fare_mode:.2f}")

# Skew direction
if fare_mean > fare_median:
    skew_direction = "Right-skewed"
elif fare_mean < fare_median:
    skew_direction = "Left-skewed"
else:
    skew_direction = "Approximately symmetric"

print(f"Fare Distribution: {skew_direction}")

# ---------------------------------------------------------
# 8. Bivariate Analysis
# ---------------------------------------------------------

# Survival rate by sex
survival_by_sex = cleaned_df.groupby("sex")["survived"].mean()

print("\n===== SURVIVAL RATE BY SEX =====")
print(survival_by_sex)

plt.figure(figsize=(8, 5))
survival_by_sex.plot(kind="bar")
plt.title("Survival Rate by Sex")
plt.xlabel("Sex")
plt.ylabel("Survival Rate")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(CHART_DIR / "survival_by_sex.png")
plt.close()

# Survival rate by passenger class
survival_by_pclass = cleaned_df.groupby("pclass")["survived"].mean()

print("\n===== SURVIVAL RATE BY PCLASS =====")
print(survival_by_pclass)

plt.figure(figsize=(8, 5))
survival_by_pclass.plot(kind="bar")
plt.title("Survival Rate by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(CHART_DIR / "survival_by_pclass.png")
plt.close()

# Survival rate by sex + pclass
survival_by_sex_pclass = (
    cleaned_df
    .groupby(["sex", "pclass"])["survived"]
    .mean()
    .reset_index()
)

print("\n===== SURVIVAL RATE BY SEX + PCLASS =====")
print(survival_by_sex_pclass)

plt.figure(figsize=(9, 5))
sns.barplot(
    data=survival_by_sex_pclass,
    x="pclass",
    y="survived",
    hue="sex"
)
plt.title("Survival Rate by Sex and Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(CHART_DIR / "survival_by_sex_pclass.png")
plt.close()

# ---------------------------------------------------------
# 9. Boolean Masking Examples
# ---------------------------------------------------------

female_survivors = cleaned_df[
    (cleaned_df["sex"] == "female") &
    (cleaned_df["survived"] == 1)
]

third_class_survivors = cleaned_df[
    (cleaned_df["pclass"] == 3) &
    (cleaned_df["survived"] == 1)
]

print("\n===== BOOLEAN MASKING =====")
print(f"Female survivors: {len(female_survivors)}")
print(f"Third-class survivors: {len(third_class_survivors)}")

# ---------------------------------------------------------
# 10. Correlation Matrix - EXACT SIX COLUMNS
# ---------------------------------------------------------

corr_columns = [
    "survived",
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare"
]

corr_matrix = cleaned_df[corr_columns].corr()

print("\n===== CORRELATION MATRIX =====")
print(corr_matrix)

plt.figure(figsize=(9, 7))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    square=True
)

plt.title("Titanic Correlation Matrix")
plt.tight_layout()
plt.savefig(CHART_DIR / "correlation_heatmap.png")
plt.close()

# ---------------------------------------------------------
# 11. Top Two Absolute Off-Diagonal Correlations
# ---------------------------------------------------------

corr_pairs = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)

corr_pairs = corr_pairs.stack()

top_two = corr_pairs.abs().sort_values(ascending=False).head(2)

print("\n===== TOP TWO ABSOLUTE CORRELATIONS =====")

for pair, value in top_two.items():
    actual_value = corr_matrix.loc[pair[0], pair[1]]
    print(
        f"{pair[0]} vs {pair[1]}: "
        f"{actual_value:.3f}"
    )

# ---------------------------------------------------------
# 12. Standardization Check - Exploratory Only
# ---------------------------------------------------------

print("\n===== STANDARDIZATION CHECK =====")

for column in ["age", "fare"]:

    mean_before = cleaned_df[column].mean()
    std_before = cleaned_df[column].std()

    z_score = (
        cleaned_df[column] - mean_before
    ) / std_before

    mean_after = z_score.mean()
    std_after = z_score.std()

    print(f"\n{column.upper()}")
    print(f"Before standardization - Mean: {mean_before:.4f}")
    print(f"Before standardization - Std: {std_before:.4f}")
    print(f"After standardization - Mean: {mean_after:.4f}")
    print(f"After standardization - Std: {std_after:.4f}")

print("\n===== EDA COMPLETED SUCCESSFULLY =====")
print(f"Charts saved in: {CHART_DIR}")