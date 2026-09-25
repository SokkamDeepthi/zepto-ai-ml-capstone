# Zepto AI/ML Capstone Project

An end-to-end Artificial Intelligence and Machine Learning capstone project covering:

1. **Data Pipeline & SQL Analytics**
2. **Titanic Exploratory Data Analysis & Machine Learning**
3. **Policy-Based Support Assistant using RAG, ChromaDB, LangGraph and FastAPI**

The complete project is implemented in Python and maintained in a public GitHub repository.

---

## Project Structure

```text
zepto-ai-ml-capstone/
│
├── README.md
├── requirements.txt
│
├── data_pipeline/
│   ├── scrape_books.py
│   ├── database.py
│   ├── queries.py
│   ├── books_cleaned.csv
│   ├── books.db
│   ├── query_outputs.txt
│   └── README.md
│
├── analytics/
│   ├── titanic.csv
│   ├── 01_eda.py
│   ├── 02_modeling.py
│   ├── 06_imbalance.py
│   ├── 07_gridsearch.py
│   ├── 08_regression.py
│   ├── 09_model_reload.py
│   ├── charts/
│   └── outputs/
│
├── models/
│   ├── best_random_forest_pipeline.pkl
│   └── fare_regression_pipeline.pkl
│
└── support_assistant/
    ├── knowledge_base.py
    ├── graph.py
    ├── main.py
    ├── Dockerfile
    ├── README.md
    ├── doc_01_delivery_policy.txt
    ├── doc_02_return_policy.txt
    ├── doc_03_refund_policy.txt
    ├── doc_04_membership_policy.txt
    ├── doc_05_tracking_policy.txt
    ├── doc_06_cancellation_policy.txt
    ├── doc_07_gift_card_policy.txt
    ├── doc_08_support_hours.txt
    └── chroma_db/
```

---

# 1. Environment Setup

Python 3.11/3.12 and a virtual environment are recommended.

### Create virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
```

### Activate virtual environment

```powershell
.\.venv\Scripts\Activate.ps1
```

### Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The project uses one consolidated `requirements.txt` for the data pipeline, analytics, machine-learning and support-assistant components.

---

# 2. Module 1 — Data Pipeline & SQL Analytics

## Objective

Scrape the first five catalogue pages from Books to Scrape, clean the data, convert GBP prices to INR using a fixed exchange rate, store the data in a normalized SQLite database, and demonstrate SQL querying and pandas SQL/JOIN equivalence.

### Technologies

* Python
* Requests
* BeautifulSoup
* Pandas
* SQLite
* SQL

### Dataset

Source:

```text
https://books.toscrape.com/
```

The scraper processes the first five catalogue pages and collects book-level information including:

* title
* price in GBP
* star rating
* availability
* category

The cleaned dataset contains at least 60 books.

### Cleaning Decisions

The following fields are created:

* `price_gbp` — numeric floating-point value
* `rating` — integer from 1 to 5
* `in_stock` — Boolean value
* `price_inr` — converted price

A fixed conversion rate is used:

```text
1 GBP = ₹105.50
```

Therefore:

```text
price_inr = price_gbp × 105.50
```

### Run Module 1

From the project root:

```powershell
python data_pipeline/scrape_books.py
```

Create the SQLite database:

```powershell
python data_pipeline/database.py
```

Run the required SQL queries and pandas JOIN comparison:

```powershell
python data_pipeline/queries.py
```

Outputs are stored in:

```text
data_pipeline/query_outputs.txt
```

---

# 3. Module 2 — Titanic EDA & Machine Learning

## Objective

Perform exploratory data analysis on the Titanic dataset and build, evaluate and compare multiple machine-learning models.

### Dataset Loading

The Titanic dataset is loaded using Seaborn and saved locally as:

```text
analytics/titanic.csv
```

The saved CSV allows subsequent analysis to run without requiring another network download.

### EDA

The analysis includes:

* dataset shape
* data types
* descriptive statistics
* missing-value percentages
* missing-value treatment
* age and fare distributions
* boxplots
* IQR-based outlier analysis
* fare mean, median and mode
* skewness interpretation
* survival analysis by sex
* survival analysis by passenger class
* survival analysis by sex and passenger class
* correlation analysis
* correlation heatmap
* z-score standardization
* multivariate visualizations

The correlation analysis uses exactly these six variables:

```text
survived
pclass
age
sibsp
parch
fare
```

### Classification Models

Three classification models are evaluated using the same stratified train/test split:

1. Logistic Regression
2. Decision Tree
3. Random Forest

Evaluation metrics include:

* Confusion Matrix
* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC

The decision tree visualization includes feature names and class names.

### Imbalanced Classification

The following approaches are compared:

* baseline model
* class-weight-balanced model
* SMOTE-based model

SMOTE is applied only to the training data to avoid test-set leakage.

### Hyperparameter Tuning

Random Forest is tuned using `GridSearchCV` over:

```text
n_estimators
max_depth
max_features
```

The Random Forest estimator uses:

```text
oob_score=True
```

The best parameters, cross-validation score and OOB score are reported.

### Fare Regression

A regression model is used to predict passenger fare.

The following metrics are calculated:

* MAE
* RMSE
* R²
* Adjusted R²

A residual plot is also generated and used to discuss heteroscedasticity.

### Model Persistence

The best trained pipeline is saved using Joblib and reloaded for prediction on raw input data.

---

## Run Module 2

Run EDA:

```powershell
python analytics/01_eda.py
```

Run classification models:

```powershell
python analytics/02_modeling.py
```

Run imbalance comparison:

```powershell
python analytics/06_imbalance.py
```

Run Random Forest GridSearch:

```powershell
python analytics/07_gridsearch.py
```

Run fare regression:

```powershell
python analytics/08_regression.py
```

Test model reload and prediction:

```powershell
python analytics/09_model_reload.py
```

Generated charts are stored in:

```text
analytics/charts/
```

Generated evaluation results are stored in:

```text
analytics/outputs/
```

---

# 4. Module 3 — Policy Support Assistant

## Objective

Build a local Retrieval-Augmented Generation (RAG) support assistant using eight policy documents.

### Technologies

* Sentence Transformers
* `all-MiniLM-L6-v2`
* ChromaDB
* LangGraph
* Pydantic
* FastAPI
* Uvicorn

No paid API is required for the graded mock-LLM workflow.

---

## Knowledge Base

The assistant uses eight policy documents:

1. Delivery Policy
2. Return Policy
3. Refund Policy
4. Membership Policy
5. Tracking Policy
6. Cancellation Policy
7. Gift Card Policy
8. Support Hours

The documents are embedded locally using:

```text
all-MiniLM-L6-v2
```

The embeddings are stored in ChromaDB using cosine similarity.

---

## RAG Architecture

```text
Policy Documents
       │
       ▼
Document Ingestion
       │
       ▼
Sentence Transformer
(all-MiniLM-L6-v2)
       │
       ▼
ChromaDB
       │
       ▼
User Query
       │
       ▼
Intent Classification
       │
       ├───────────────┐
       ▼               ▼
Policy Query       General Query
       │               │
       ▼               ▼
Top-3 Retrieval    Direct Answer
       │
       ▼
Structured Response
       │
       ▼
FastAPI /ask
```

### LangGraph Nodes

The graph contains three nodes:

```text
classify_intent
retrieve_and_answer
direct_answer
```

A conditional edge routes policy-related queries to retrieval and general queries to the direct-answer path.

---

## Structured Prompt

The support assistant prompt contains:

* Role
* Context
* Task
* Format
* Length constraint
* Negative constraint
* Few-shot example

The response schema contains:

```text
answer
sources
confidence
```

The mock workflow is deterministic and does not require a paid LLM API.

---

# 5. Run the Support Assistant

From the project root:

### Build the knowledge base

```powershell
python support_assistant/knowledge_base.py
```

### Start FastAPI

```powershell
uvicorn support_assistant.main:app --port 8001
```

The API is available at:

```text
http://127.0.0.1:8001
```

Swagger documentation:

```text
http://127.0.0.1:8001/docs
```

---

## API Endpoint

### POST

```text
/ask
```

Request:

```json
{
  "query": "What is the delivery policy?"
}
```

Example response:

```json
{
```
