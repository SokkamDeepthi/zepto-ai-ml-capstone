# Titanic Survival Prediction — AI/ML Capstone Project

## Project Overview

This project builds an end-to-end Machine Learning pipeline to predict whether a Titanic passenger would survive based on passenger information.

The project covers data loading, exploratory analysis, preprocessing, model training, model evaluation, model saving, prediction, and Streamlit deployment.

## Objectives

* Analyze the Titanic passenger dataset.
* Perform data preprocessing and exploratory analysis.
* Train a Random Forest classification model.
* Evaluate model performance.
* Save the trained model for reuse.
* Make predictions for new passenger data.
* Build an interactive Streamlit application.

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Joblib
* Streamlit
* BeautifulSoup
* Requests

## Dataset

The project uses a Titanic passenger dataset containing passenger information such as:

* Passenger Class
* Gender
* Age
* Fare
* Survival Status

Dataset file:

`analytics/titanic.csv`

## Project Structure

```text
zepto-ai-ml-capstone/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data_pipeline/
│   └── README.md
│
├── analytics/
│   ├── titanic.csv
│   ├── 02_modeling.py
│   ├── 03_evaluation.py
│   ├── 04_save_model.py
│   ├── 05_predict.py
│   └── outputs/
│       ├── confusion_matrix.png
│       └── feature_importance.png
│
└── models/
    ├── titanic_model.pkl
    └── sex_encoder.pkl
```

## Machine Learning Workflow

### 1. Data Loading

The Titanic dataset is loaded using Pandas.

### 2. Data Preprocessing

Selected features:

* Pclass
* Sex
* Age
* Fare

Missing age values are handled using the median age.

The gender column is converted into numerical values using LabelEncoder.

### 3. Model Training

A Random Forest Classifier is used for prediction.

The dataset is divided into:

* 80% training data
* 20% testing data

### 4. Model Evaluation

The trained model is evaluated using accuracy and classification metrics.

Test accuracy achieved:

**80.45%**

The project also generates:

* Confusion Matrix
* Feature Importance Chart

### 5. Model Saving

The trained model is saved using Joblib:

`models/titanic_model.pkl`

The gender encoder is saved as:

`models/sex_encoder.pkl`

### 6. Prediction

The saved model is loaded and tested with new passenger information.

Example input:

```text
Passenger Class: 3
Gender: female
Age: 25
Fare: 20
```

The model returns a predicted survival class and prediction probability.

### 7. Streamlit Application

An interactive Streamlit web application allows users to enter passenger information and receive a survival prediction.

Run the application using:

```bash
python -m streamlit run app.py
```

The application runs locally at:

```text
http://localhost:8501
```

## Results

The Random Forest model achieved approximately:

**Accuracy: 80.45%**

The saved model can generate predictions for new passenger records through the Python prediction script and Streamlit application.

## Future Improvements

* Add more Titanic features such as SibSp, Parch, and Embarked.
* Perform advanced feature engineering.
* Compare multiple classification algorithms.
* Perform hyperparameter tuning.
* Perform cross-validation.
* Deploy the Streamlit application online.

## Conclusion

This project demonstrates a complete Machine Learning workflow, from dataset preparation and preprocessing to model training, evaluation, model persistence, prediction, and interactive application deployment.
