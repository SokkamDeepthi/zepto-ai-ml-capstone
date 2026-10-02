import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Project folder
BASE_DIR = Path(__file__).resolve().parent

# Load model and encoder
MODEL_PATH = BASE_DIR / "models" / "titanic_model.pkl"
ENCODER_PATH = BASE_DIR / "models" / "sex_encoder.pkl"

model = joblib.load(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)

# Page settings
st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="centered"
)

st.title("🚢 Titanic Survival Prediction")
st.write("Enter passenger details to predict survival.")

# User inputs
pclass = st.selectbox(
    "Passenger Class",
    [1, 2, 3]
)

sex = st.selectbox(
    "Gender",
    ["female", "male"]
)

age = st.number_input(
    "Age",
    min_value=0.0,
    max_value=100.0,
    value=25.0
)

fare = st.number_input(
    "Fare",
    min_value=0.0,
    value=20.0
)

# Prediction button
if st.button("Predict Survival"):

    passenger = pd.DataFrame({
        "pclass": [pclass],
        "sex": [sex],
        "age": [age],
        "fare": [fare]
    })

    # Encode gender
    passenger["sex"] = encoder.transform(passenger["sex"])

    # Prediction
    prediction = model.predict(passenger)[0]
    probability = model.predict_proba(passenger)[0]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success("✅ Passenger is predicted to Survive")
    else:
        st.error("❌ Passenger is predicted not to Survive")

    st.write(
        f"**Survival Probability:** {probability[1] * 100:.2f}%"
    )

    st.write(
        f"**Non-Survival Probability:** {probability[0] * 100:.2f}%"
    )