# Module 2 — Titanic Analytics & Machine Learning

## Overview

This module performs Exploratory Data Analysis (EDA), data cleaning, classification model training, model comparison, and deployment of a Titanic Survival Prediction application.

## Dataset

The Titanic dataset contains 891 passenger records and 15 columns.

Dataset file:

```text
analytics/titanic.csv
```

The target variable is:

```text
survived
```

where:

* `0` = Did Not Survive
* `1` = Survived

## EDA

The following analysis was performed:

* Dataset shape and column inspection
* Data type inspection
* Missing-value analysis
* Descriptive statistics
* Survival distribution analysis
* Survival analysis by gender
* Survival analysis by passenger class
* Age distribution
* Fare distribution
* Correlation heatmap

## Data Cleaning

The following cleaning decisions were applied:

* Missing `age` values were filled using the median age.
* Missing `fare` values were filled using the median fare.
* Missing `embarked` values were filled using the most frequent value.
* The `deck` column was removed because it contained a large number of missing values.

The cleaned dataset is saved as:

```text
analytics/outputs/titanic_cleaned.csv
```

## Machine Learning

Three classification models were trained:

1. Logistic Regression
2. Decision Tree
3. Random Forest

Features used for prediction:

```text
pclass
sex
age
sibsp
parch
fare
embarked
```

Categorical variables were encoded using One-Hot Encoding.

The data was divided into training and testing sets us
