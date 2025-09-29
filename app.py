# app.py - Streamlit Ozone Predictor with error handling

import streamlit as st
import pandas as pd
import joblib
import os

# Path to the trained Random Forest model
model_path = "rf_ozone_model.joblib"


# Load model with error handling
if os.path.exists(model_path):
    try:
        model = joblib.load(model_path)
        model_loaded = True
    except Exception as e:
        st.error(f"Error loading model: {e}")
        model_loaded = False
else:
    st.error(f"Model file not found at {model_path}")
    model_loaded = False

# Only continue if model is loaded successfully
if model_loaded:
    # Set Streamlit page config
    st.set_page_config(page_title="Ozone Predictor", layout="centered")

    # Title and description
    st.title("Ozone Level Predictor (AirQuality Dataset)")
    st.markdown("Enter feature values to predict Ozone concentration (ppb).")

    # Sidebar inputs
    st.sidebar.header("Input Features")
    solar = st.sidebar.number_input("Solar.R (solar radiation)", min_value=0.0, value=200.0, step=1.0)
    wind = st.sidebar.number_input("Wind (mph)", min_value=0.0, value=7.0, step=0.1)
    temp = st.sidebar.number_input("Temperature (F)", min_value=0.0, value=77.0, step=0.1)
    month = st.sidebar.selectbox("Month", options=[5,6,7,8,9], index=2, format_func=lambda x: f"{x}")

    # Prepare input DataFrame
    inp = pd.DataFrame({
        'Solar.R': [solar],
        'Wind': [wind],
        'Temp': [temp],
        'Month': [month]
    })

    st.write("### Input Values")
    st.table(inp.T)

    # Prediction button
    if st.button("Predict Ozone"):
        try:
            pred = model.predict(inp)[0]
            st.success(f"Predicted Ozone (ppb): {pred:.2f}")

            # Simple interpretation
            if pred < 50:
                st.info("Air quality: Good (low ozone)")
            elif pred < 100:
                st.warning("Air quality: Moderate (elevated ozone)")
            else:
                st.error("Air quality: Poor (high ozone)")
        except Exception as e:
            st.error(f"Error making prediction: {e}")
