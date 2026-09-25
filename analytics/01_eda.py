import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# ============================================================
# 1. LOAD TITANIC DATASET ONCE AND SAVE OFFLINE COPY
# ============================================================

DATA_PATH = os.path.join(os.path.dirname(__file__), "titanic.csv")
CHART_DIR = os.path.join(os.path.dirname(__file__), "charts")

os.makedirs(CHART_DIR, exist_ok=True)

# Load the Titanic dataset exactly once from seaborn
df = sns.load_dataset("titanic")

# Immediately save the downloaded dataset as an offline fallback
df.to_csv(DATA_PATH, index=False)

print("=" * 70)
print("TITANIC DATASET LOADED")
print("=" * 70)

print(f"Shape: {df.shape}")

print("\nData types:")
print(df.dtypes)

print("\nDataset info:")
df.info()

print("\nDescriptive statistics:")
print(df.describe(include="all"))


# ============================================================
# 2. MISSING VALUE ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("MISSING VALUE ANALYSIS")
print("=" * 70)

missing_count = df.isnull().sum()
missing_percent = (missing_count / len(df)) * 100

missing_table = pd.DataFrame({
    "missing_count": missing_count,
    "missing_percent": missing_percent.round(2)
})

print(missing_table[missing_table["missing_count"] > 0])


# ============================================================
# 3. MISSING VALUE TREATMENT
# ============================================================

cleaned_df = df.copy()

print("\n" + "=" * 70)
print("MISSING VALUE TREATMENT")
print("=" * 70)

# AGE: 5% to 30% missing -> median imputation
age_missing_pct = cleaned_df["age"].isnull().mean() * 100

if 5 <= age_missing_pct <= 30:
    age_median = cleaned_df["age"].median()
    cleaned_df["age"] = cleaned_df["age"].fillna(age_median)

    print(
        f"age: {age_missing_pct:.2f}% missing -> "
        f"median imputation using {age_median:.2f}"
    )


# EMBARKED: less than 5% missing -> drop affected rows
embarked_missing_pct = cleaned_df["embarked"].isnull().mean() * 100

if embarked_missing_pct < 5:
    before = len(cleaned_df)

    cleaned_df = cleaned_df.dropna(subset=["embarked"])

    after = len(cleaned_df)

    print(
        f"embarked: {embarked_missing_pct:.2f}% missing -> "
        f"dropped {before - after} affected rows"
    )


# EMBARK_TOWN: less than 5% missing -> drop affected rows
embark_town_missing_pct = cleaned_df["embark_town"].isnull().mean() * 100

if embark_town_missing_pct < 5:
    before = len(cleaned_df)

    cleaned_df = cleaned_df.dropna(subset=["embark_town"])

    after = len(cleaned_df)

    print(
        f"embark_town: {embark_town_missing_pct:.2f}% missing -> "
        f"dropped {before - after} affected rows"
    )


# DECK: more than 30% missing -> drop high-missing column
deck_missing_pct = cleaned_df["deck"].isnull().mean() * 100

if deck_missing_pct > 30:
    cleaned_df = cleaned_df.drop(columns=["deck"])

    print(
        f"deck: {deck_missing_pct:.2f}% missing -> "
        "column dropped because missingness is very high"
    )


print("\nRemaining missing values:")
print(cleaned_df.isnull().sum())


# Save cleaned dataset so subsequent modules can work offline
cleaned_df.to_csv(DATA_PATH, index=False)

print(f"\nCleaned dataset saved to: {DATA_PATH}")
print(f"Cleaned shape: {cleaned_df.shape}")


# ============================================================
# 4. AGE HISTOGRAM
# ============================================================

plt.figure(figsize=(8, 5))

sns.histplot(cleaned_df["age"], kde=True)

plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.tight_layout()

plt.savefig(
    os.path.join(CHART_DIR, "age_histogram.png"),
    dpi=150
)

plt.close()


# ============================================================
# 5. AGE BOXPLOT
# ============================================================

plt.figure(figsize=(8, 5))

sns.boxplot(x=cleaned_df["age"])

plt.title("Age Boxplot")
plt.xlabel("Age")
plt.tight_layout()

plt.savefig(
    os.path.join(CHART_DIR, "age_boxplot.png"),
    dpi=150
)

plt.close()


# ============================================================
# 6. FARE HISTOGRAM
# ============================================================

plt.figure(figsize=(8, 5))

sns.histplot(cleaned_df["fare"], kde=True)

plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.ylabel("Count")
plt.tight_layout()

plt.savefig(
    os.path.join(CHART_DIR, "fare_histogram.png"),
    dpi=150
)

plt.close()


# ============================================================
# 7. FARE BOXPLOT
# ============================================================

plt.figure(figsize=(8, 5))

sns.boxplot(x=cleaned_df["fare"])

plt.title("Fare Boxplot")
plt.xlabel("Fare")
plt.tight_layout()

plt.savefig(
    os.path.join(CHART_DIR, "fare_boxplot.png"),
    dpi=150
)

plt.close()


# ============================================================
# 8. IQR OUTLIER ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("IQR OUTLIER ANALYSIS")
print("=" * 70)


def iqr_outlier_count(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = series[
        (series < lower_bound) |
        (series > upper_bound)
    ]

    return {
        "Q1": q1,
        "Q3": q3,
        "IQR": iqr,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "outlier_count": len(outliers)
    }


age_outliers = iqr_outlier_count(cleaned_df["age"])
fare_outliers = iqr_outlier_count(cleaned_df["fare"])

print("\nAge:")
print(age_outliers)

print("\nFare:")
print(fare_outliers)


# ============================================================
# 9. FARE MEAN / MEDIAN / MODE / SKEWNESS
# ============================================================

print("\n" + "=" * 70)
print("FARE STATISTICS AND SKEWNESS")
print("=" * 70)

fare_mean = cleaned_df["fare"].mean()
fare_median = cleaned_df["fare"].median()
fare_mode = cleaned_df["fare"].mode().iloc[0]
fare_skewness = cleaned_df["fare"].skew()

print(f"Mean   : {fare_mean:.4f}")
print(f"Median : {fare_median:.4f}")
print(f"Mode   : {fare_mode:.4f}")
print(f"Skewness: {fare_skewness:.4f}")

if fare_mean > fare_median > fare_mode:
    skew_conclusion = "Fare is positively/right skewed because mean > median > mode."
elif fare_mean < fare_median < fare_mode:
    skew_conclusion = "Fare is negatively/left skewed because mean < median < mode."
elif fare_mean > fare_median:
    skew_conclusion = "Fare shows positive/right skewness because mean is greater than median."
elif fare_mean < fare_median:
    skew_conclusion = "Fare shows negative/left skewness because mean is lower than median."
else:
    skew_conclusion = "Mean and median are approximately similar, suggesting limited skewness."

print("\nConclusion:")
print(skew_conclusion)


# ============================================================
# 10. SURVIVAL BY SEX USING BOOLEAN MASKING
# ============================================================

print("\n" + "=" * 70)
print("SURVIVAL BY SEX - BOOLEAN MASKING")
print("=" * 70)

sex_results = []

for sex in cleaned_df["sex"].dropna().unique():

    mask = cleaned_df["sex"] == sex

    group = cleaned_df.loc[mask]

    survival_rate = group["survived"].mean()

    sex_results.append({
        "sex": sex,
        "passengers": len(group),
        "survival_rate": survival_rate
    })

sex_survival = pd.DataFrame(sex_results)

print(sex_survival)


# Chart
plt.figure(figsize=(8, 5))

sns.barplot(
    data=sex_survival,
    x="sex",
    y="survival_rate"
)

plt.title("Survival Rate by Sex")
plt.ylabel("Survival Rate")
plt.xlabel("Sex")
plt.ylim(0, 1)

plt.tight_layout()

plt.savefig(
    os.path.join(CHART_DIR, "survival_by_sex.png"),
    dpi=150
)

plt.close()


# ============================================================
# 11. SURVIVAL BY PCLASS USING BOOLEAN MASKING
# ============================================================

print("\n" + "=" * 70)
print("SURVIVAL BY PCLASS - BOOLEAN MASKING")
print("=" * 70)

pclass_results = []

for pclass in sorted(cleaned_df["pclass"].dropna().unique()):

    mask = cleaned_df["pclass"] == pclass

    group = cleaned_df.loc[mask]

    survival_rate = group["survived"].mean()

    pclass_results.append({
        "pclass": pclass,
        "passengers": len(group),
        "survival_rate": survival_rate
    })

pclass_survival = pd.DataFrame(pclass_results)

print(pclass_survival)


# Chart
plt.figure(figsize=(8, 5))

sns.barplot(
    data=pclass_survival,
    x="pclass",
    y="survival_rate"
)

plt.title("Survival Rate by Passenger Class")
plt.ylabel("Survival Rate")
plt.xlabel("Passenger Class")
plt.ylim(0, 1)

plt.tight_layout()

plt.savefig(
    os.path.join(CHART_DIR, "survival_by_pclass.png"),
    dpi=150
)

plt.close()


# ============================================================
# 12. SURVIVAL BY SEX + PCLASS USING BOOLEAN MASKING
# ============================================================

print("\n" + "=" * 70)
print("SURVIVAL BY SEX + PCLASS - BOOLEAN MASKING")
print("=" * 70)

sex_pclass_results = []

for sex in sorted(cleaned_df["sex"].dropna().unique()):

    for pclass in sorted(cleaned_df["pclass"].dropna().unique()):

        mask = (
            (cleaned_df["sex"] == sex) &
            (cleaned_df["pclass"] == pclass)
        )

        group = cleaned_df.loc[mask]

        if len(group) > 0:

            survival_rate = group["survived"].mean()

            sex_pclass_results.append({
                "sex": sex,
                "pclass": pclass,
                "passengers": len(group),
                "survival_rate": survival_rate
            })

sex_pclass_survival = pd.DataFrame(sex_pclass_results)

print(sex_pclass_survival)


# Chart
plt.figure(figsize=(9, 5))

sns.barplot(
    data=sex_pclass_survival,
    x="pclass",
    y="survival_rate",
    hue="sex"
)

plt.title("Survival Rate by Sex and Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")
plt.ylim(0, 1)

plt.tight_layout()

plt.savefig(
    os.path.join(CHART_DIR, "survival_by_sex_pclass.png"),
    dpi=150
)

plt.close()


# ============================================================
# 13. EXACT SIX-COLUMN CORRELATION MATRIX
# ============================================================

print("\n" + "=" * 70)
print("CORRELATION ANALYSIS")
print("=" * 70)

correlation_columns = [
    "survived",
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare"
]

corr = cleaned_df[correlation_columns].corr()

print("\nCorrelation matrix:")
print(corr)


# Heatmap
plt.figure(figsize=(9, 7))

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Titanic Correlation Heatmap")
plt.tight_layout()

plt.savefig(
    os.path.join(CHART_DIR, "correlation_heatmap.png"),
    dpi=150
)

plt.close()


# ============================================================
# 14. TWO STRONGEST ABSOLUTE OFF-DIAGONAL CORRELATIONS
# ============================================================

corr_abs = corr.abs().copy()

for i in range(len(corr_abs)):
    corr_abs.iloc[i, i] = np.nan
    
top_pairs = (
    corr_abs
    .stack()
    .sort_values(ascending=False)
)

unique_pairs = []

for (col1, col2), value in top_pairs.items():

    pair = frozenset([col1, col2])

    if pair not in [
        frozenset([a, b])
        for a, b, _ in unique_pairs
    ]:

        unique_pairs.append(
            (col1, col2, corr.loc[col1, col2])
        )

    if len(unique_pairs) == 2:
        break


print("\nTwo strongest absolute off-diagonal correlations:")

for col1, col2, value in unique_pairs:

    print(
        f"{col1} <-> {col2}: "
        f"{value:.4f}"
    )


# ============================================================
# 15. Z-SCORE STANDARDIZATION
# ============================================================

print("\n" + "=" * 70)
print("Z-SCORE STANDARDIZATION")
print("=" * 70)

standardization_df = cleaned_df[
    ["age", "fare"]
].copy()

before_mean = standardization_df.mean()
before_std = standardization_df.std()

standardized_df = (
    standardization_df -
    standardization_df.mean()
) / standardization_df.std()

after_mean = standardized_df.mean()
after_std = standardized_df.std()

print("\nBefore standardization:")
print(
    pd.DataFrame({
        "mean": before_mean,
        "std": before_std
    })
)

print("\nAfter standardization:")
print(
    pd.DataFrame({
        "mean": after_mean,
        "std": after_std
    })
)


# ============================================================
# 16. MULTIVARIATE CHART 1
# ============================================================

plt.figure(figsize=(9, 6))

sns.barplot(
    data=sex_pclass_survival,
    x="pclass",
    y="survival_rate",
    hue="sex"
)

plt.title("Multivariate Chart 1: Survival by Sex and Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")
plt.ylim(0, 1)

plt.tight_layout()

plt.savefig(
    os.path.join(CHART_DIR, "multivariate_1_sex_pclass.png"),
    dpi=150
)

plt.close()


# ============================================================
# 17. MULTIVARIATE CHART 2
# ============================================================

plt.figure(figsize=(9, 6))

sns.scatterplot(
    data=cleaned_df,
    x="age",
    y="fare",
    hue="survived",
    style="sex",
    alpha=0.7
)

plt.title("Multivariate Chart 2: Age vs Fare by Survival and Sex")
plt.xlabel("Age")
plt.ylabel("Fare")

plt.tight_layout()

plt.savefig(
    os.path.join(CHART_DIR, "multivariate_2_age_fare.png"),
    dpi=150
)

plt.close()


# ============================================================
# 18. MULTIVARIATE CHART 3
# ============================================================

age_bins = [0, 12, 18, 30, 45, 60, 100]
age_labels = [
    "0-12",
    "13-18",
    "19-30",
    "31-45",
    "46-60",
    "61+"
]

age_group_df = cleaned_df.copy()

age_group_df["age_group"] = pd.cut(
    age_group_df["age"],
    bins=age_bins,
    labels=age_labels,
    include_lowest=True
)

age_survival = (
    age_group_df
    .groupby(
        ["age_group", "sex"],
        observed=False
    )["survived"]
    .mean()
    .reset_index()
)

plt.figure(figsize=(10, 6))

sns.lineplot(
    data=age_survival,
    x="age_group",
    y="survived",
    hue="sex",
    marker="o"
)

plt.title("Multivariate Chart 3: Survival by Age Group and Sex")
plt.xlabel("Age Group")
plt.ylabel("Survival Rate")
plt.ylim(0, 1)

plt.tight_layout()

plt.savefig(
    os.path.join(CHART_DIR, "multivariate_3_age_sex.png"),
    dpi=150
)

plt.close()


# ============================================================
# 19. MULTIVARIATE CHART 4
# ============================================================

plt.figure(figsize=(10, 6))

sns.boxplot(
    data=cleaned_df,
    x="pclass",
    y="fare",
    hue="survived"
)

plt.title("Multivariate Chart 4: Fare by Class and Survival")
plt.xlabel("Passenger Class")
plt.ylabel("Fare")

plt.tight_layout()

plt.savefig(
    os.path.join(CHART_DIR, "multivariate_4_fare_class_survival.png"),
    dpi=150
)

plt.close()


# ============================================================
# 20. WRITTEN MULTIVARIATE INTERPRETATIONS
# ============================================================

interpretations = """

MULTIVARIATE CHART INTERPRETATIONS
===================================

1. Survival by Sex and Passenger Class:
Survival differs substantially across both sex and passenger class.
Female passengers generally show higher survival rates than male passengers within the same class.
Passenger class also separates survival outcomes, with first-class passengers generally showing higher survival than lower classes.

2. Age vs Fare by Survival and Sex:
The scatter plot shows the relationship between passenger age, fare and survival while also separating observations by sex.
Higher fares are concentrated among certain passenger groups, while survival is distributed differently across age and fare levels.
This indicates that fare, age and sex together provide useful information for understanding survival patterns.

3. Survival by Age Group and Sex:
Survival rates vary across age groups and differ between male and female passengers.
The comparison shows that survival is not determined by age alone because the pattern changes between the two sexes.
Age group therefore provides additional context when survival is analyzed together with sex.

4. Fare by Class and Survival:
Fare distributions differ strongly across passenger classes.
First-class passengers generally paid higher fares, and fare values also vary between survivors and non-survivors.
This demonstrates the combined relationship between socioeconomic class, fare and survival.
"""

interpretation_path = os.path.join(
    os.path.dirname(__file__),
    "multivariate_interpretations.txt"
)

with open(
    interpretation_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(interpretations)


# ============================================================
# 21. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("EDA COMPLETED SUCCESSFULLY")
print("=" * 70)

print(f"Original dataset shape : {df.shape}")
print(f"Cleaned dataset shape  : {cleaned_df.shape}")

print("\nCharts saved in:")
print(CHART_DIR)

print("\nMultivariate interpretations saved in:")
print(interpretation_path)

print("\nRequired correlation columns:")
print(correlation_columns)

print("\nEDA pipeline completed.")