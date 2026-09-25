# Module 2 — Titanic Analytics & Machine Learning

## Objective

This module performs exploratory data analysis, preprocessing, classification, imbalance handling, hyperparameter tuning, regression, model persistence, and prediction using the Titanic dataset.

## Dataset Loading

The Titanic dataset is loaded once using:

```python
sns.load_dataset("titanic")
```

Immediately after loading, the raw dataset is saved as:

```text
analytics/titanic.csv
```

All subsequent analysis and modeling uses the saved CSV file to improve reproducibility and avoid repeated network loading.

Dataset size:

* Rows: 891
* Columns: 15

## Exploratory Data Analysis

The EDA workflow includes:

* Dataset shape, information, and descriptive statistics
* Missing-value percentages for affected columns
* Missing-value treatment based on the percentage of missing data
* Age and fare histograms
* Age and fare boxplots
* IQR-based outlier counts
* Fare mean, median, mode, and skewness
* Survival analysis by sex
* Survival analysis by passenger class
* Survival analysis by sex and passenger class
* Correlation heatmap
* Z-score standardization of age and fare

### Missing-Value Decisions

The following rules were applied:

* Less than 5% missing → rows were dropped.
* 5%–30% missing → median/mode imputation was applied.
* More than 30% missing → the column was dropped.

The `deck` column had a high percentage of missing values and was removed because retaining it would introduce substantial missing-data uncertainty.

## Correlation Analysis

The required six numerical variables were used:

```text
survived
pclass
age
sibsp
parch
fare
```

The strongest absolute correlations were:

| Variable Pair | Correlation |
| ------------- | ----------: |
| pclass ↔ fare |     -0.5482 |
| age ↔ pclass  |     -0.3365 |

The negative `pclass`–`fare` correlation indicates that passenger class and fare are meaningfully associated in this dataset. The negative `age`–`pclass` correlation is weaker but is the second strongest absolute relationship among the required variables.

## Multivariate Analysis

Four multivariate visualizations were generated:

1. Survival rate by sex and passenger class
2. Age versus fare by survival and sex
3. Survival rate by age group and sex
4. Fare distribution by passenger class and survival

Written interpretations are saved in:

```text
analytics/multivariate_interpretations.txt
```

Charts are saved under:

```text
analytics/charts/
```

## Classification

The target variable is:

```text
survived
```

Features include:

```text
pclass, sex, age, sibsp, parch, fare, embarked
```

A stratified train/test split was used so that the survival-class proportions remain approximately consistent between training and test sets.

### Preprocessing

Preprocessing was fit only on the training data using a `ColumnTransformer` and pipeline:

* Numeric missing values → median imputation
* Categorical missing values → most-frequent imputation
* Numeric features → standardization
* Categorical features → one-hot encoding

This prevents test-set information from leaking into model training.

## Classification Models

Three classifiers were trained using the same train/test split:

* Logistic Regression
* Decision Tree
* Random Forest

Evaluation metrics include:

* Confusion matrix
* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC

The trained pipelines are saved in:

```text
models/
```

## Class Imbalance

Three approaches were compared:

1. Baseline Random Forest
2. Random Forest with `class_weight="balanced"`
3. Random Forest with SMOTE

SMOTE was applied only to the training data.

The comparison is saved as:

```text
analytics/outputs/imbalance_comparison.csv
```

## Random Forest GridSearchCV

GridSearchCV was used to tune:

* `n_estimators`
* `max_depth`
* `max_features`

The Random Forest estimator used:

```text
oob_score=True
```

### GridSearch Results

| Metric         |  Value |
| -------------- | -----: |
| Best CV F1     | 0.7449 |
| OOB Score      | 0.8073 |
| Test Accuracy  | 0.8034 |
| Test Precision | 0.7619 |
| Test Recall    | 0.7059 |
| Test F1        | 0.7328 |
| Test ROC-AUC   | 0.8237 |

The tuned Random Forest pipeline is saved as:

```text
models/best_random_forest_pipeline.pkl
```

Detailed results are saved in:

```text
analytics/outputs/gridsearch_results.csv
```

## Fare Regression

A multivariate linear regression model was trained to predict passenger fare.

Evaluation metrics:

| Metric      |   Value |
| ----------- | ------: |
| MAE         | 21.0986 |
| RMSE        | 41.7021 |
| R²          |  0.3482 |
| Adjusted R² |  0.3091 |

The model explains approximately 34.8% of the variance in fare on the test data.

The regression residual plot is saved as:

```text
analytics/outputs/regression_residual_plot.png
```

The fitted regression pipeline is saved as:

```text
models/fare_regression_pipeline.pkl
```

Regression metrics are saved as:

```text
analytics/outputs/regression_results.csv
```

## Model Comparison

The classification models are evaluated using classification metrics, while the fare regression model is evaluated separately using regression metrics.

The tuned Random Forest achieved:

* Test Accuracy: 0.8034
* Precision: 0.7619
* Recall: 0.7059
* F1-score: 0.7328
* ROC-AUC: 0.8237

For fare prediction, the regression model achieved:

* MAE: 21.0986
* RMSE: 41.7021
* R²: 0.3482
* Adjusted R²: 0.3091

### Final Model Recommendation

The tuned Random Forest provides a useful balance of precision, recall, F1-score, and ROC-AUC for Titanic survival classification. Its test ROC-AUC of 0.8237 indicates useful discrimination between survival classes on the held-out test set. The OOB score of 0.8073 also provides an additional estimate of Random Forest generalization performance. The fare regression model has an R² of 0.3482, indicating that the selected passenger attributes explain part, but not most, of the observed fare variation.

## Model Persistence and Reload

The complete fitted Random Forest pipeline is saved using `joblib`.

It can be reloaded and used with raw passenger input without manually repeating preprocessing.

Example:

```bash
python analytics/09_model_reload.py
```

The reload workflow demonstrates:

* Loading the complete fitted pipeline
* Providing raw input
* Applying the saved preprocessing automatically
* Generating a survival prediction
* Returning prediction probabilities

## How to Run

From the project root:

```bash
python analytics/01_eda.py
python analytics/02_modeling.py
python analytics/06_imbalance.py
python analytics/07_gridsearch.py
python analytics/08_regression.py
python analytics/09_model_reload.py
```

## Output Files

Important generated outputs include:

```text
analytics/
├── charts/
├── outputs/
│   ├── imbalance_comparison.csv
│   ├── gridsearch_results.csv
│   ├── regression_results.csv
│   ├── regression_residual_plot.png
│   └── ...
├── multivariate_interpretations.txt
└── titanic.csv

models/
├── logistic_pipeline.pkl
├── decision_tree_pipeline.pkl
├── random_forest_pipeline.pkl
├── best_random_forest_pipeline.pkl
└── fare_regression_pipeline.pkl
```
