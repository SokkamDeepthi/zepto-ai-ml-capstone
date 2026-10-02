from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent / "models" / "titanic_model.pkl"


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="centered",
)


# --------------------------------------------------
# Load model
# --------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("🚢 Titanic Survival Prediction")

st.write(
    "Enter passenger details below to predict whether "
    "the passenger would have survived."
)


# --------------------------------------------------
# User inputs
# --------------------------------------------------

pclass = st.selectbox(
    "Passenger Class",
    options=[1, 2, 3],
    index=2,
)

sex = st.selectbox(
    "Sex",
    options=["male", "female"],
)

age = st.number_input(
    "Age",
    min_value=0.0,
    max_value=100.0,
    value=30.0,
    step=1.0,
)

sibsp = st.number_input(
    "Number of Siblings/Spouses Aboard",
    min_value=0,
    max_value=10,
    value=0,
    step=1,
)

parch = st.number_input(
    "Number of Parents/Children Aboard",
    min_value=0,
    max_value=10,
    value=0,
    step=1,
)

fare = st.number_input(
    "Ticket Fare",
    min_value=0.0,
    max_value=600.0,
    value=32.0,
    step=1.0,
)

embarked = st.selectbox(
    "Port of Embarkation",
    options=["S", "C", "Q"],
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button("Predict Survival", type="primary"):

    input_data = pd.DataFrame(
        {
            "pclass": [pclass],
            "sex": [sex],
            "age": [age],
            "sibsp": [sibsp],
            "parch": [parch],
            "fare": [fare],
            "embarked": [embarked],
        }
    )

    prediction = model.predict(input_data)[0]

    # Probability
    probabilities = model.predict_proba(input_data)[0]

    survived_probability = probabilities[1] * 100
    not_survived_probability = probabilities[0] * 100

    st.subheader("Prediction")

    if prediction == 1:
        st.success("✅ Prediction: Survived")
    else:
        st.error("❌ Prediction: Did Not Survive")

    st.write(
        f"**Survived probability:** "
        f"{survived_probability:.1f}%"
    )

    st.write(
        f"**Did Not Survive probability:** "
        f"{not_survived_probability:.1f}%"
    )


# --------------------------------------------------
# Model information
# --------------------------------------------------

st.divider()

st.caption(
    "Model: Best-performing classification pipeline "
    "trained on the Titanic dataset."
)