import streamlit as st
import pandas as pd
import joblib

# Load model + encoder
model = joblib.load("crop_logistic_model.pkl")
le = joblib.load("label_encoder.pkl")

st.title("Crop Recommendation System (Logistic Regression)")

n = st.number_input("Nitrogen (n)", 0, 300, 50)
p = st.number_input("Phosphorus (p)", 0, 300, 40)
k = st.number_input("Potassium (k)", 0, 300, 40)
temperature = st.number_input("Temperature (°C)", 0.0, 60.0, 25.0)
humidity = st.number_input("Humidity (%)", 0.0, 100.0, 80.0)
ph = st.number_input("Soil pH", 0.0, 14.0, 6.5)
rainfall = st.number_input("Rainfall (mm)", 0.0, 500.0, 200.0)

if st.button("Predict Crop"):
    sample = pd.DataFrame([{
        "n": n,
        "p": p,
        "k": k,
        "temperature": temperature,
        "humidity": humidity,
        "ph": ph,
        "rainfall": rainfall
    }])

    pred_encoded = model.predict(sample)[0]
    pred_label = le.inverse_transform([pred_encoded])[0]

    st.success(f"Recommended Crop: **{pred_label}**")
