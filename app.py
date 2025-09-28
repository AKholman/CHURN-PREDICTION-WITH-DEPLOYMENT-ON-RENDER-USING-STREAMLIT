# app.py
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import json

@st.cache_resource  # caches loaded object across reruns
def load_model():
    pipeline = joblib.load("artifacts/model_pipeline.joblib")
    with open("artifacts/model_metadata.json") as f:
        meta = json.load(f)
    return pipeline, meta

pipeline, meta = load_model()
expected_features = meta['features']

st.title("Churn Prediction (XGBoost)")

# Example: simple form for a single prediction
with st.form("predict_form"):
    # Numeric features
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=50.0)
    total_charges = st.number_input("Total Charges", min_value=0.0, value=600.0)
    tenure_days = st.number_input("Tenure Days", min_value=0, max_value=5000, value=365)
    senior_citizen = st.selectbox("Senior Citizen", [0, 1])

    # Categorical features
    type_input = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    online_security = st.selectbox("Online Security", ["Yes", "No"])
    online_backup = st.selectbox("Online Backup", ["Yes", "No"])
    device_protection = st.selectbox("Device Protection", ["Yes", "No"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No"])

    submitted = st.form_submit_button("Predict")

if submitted:
    input_dict = {
        "type": type_input,
        "paperless_billing": paperless_billing,
        "payment_method": payment_method,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "tenure_days": tenure_days,
        "senior_citizen": senior_citizen,
        "partner": partner,
        "dependents": dependents,
        "internet_service": internet_service,
        "online_security": online_security,
        "online_backup": online_backup,
        "device_protection": device_protection,
        "tech_support": tech_support,
        "streaming_tv": streaming_tv,
        "streaming_movies": streaming_movies,
        "multiple_lines": multiple_lines
    }

    df = pd.DataFrame([input_dict])

    # Defaults for missing columns (safe values)
    defaults = {
        "type": "Month-to-month",
        "paperless_billing": "Yes",
        "payment_method": "Electronic check",
        "partner": "No",
        "dependents": "No",
        "internet_service": "DSL",
        "online_security": "No",
        "online_backup": "No",
        "device_protection": "No",
        "tech_support": "No",
        "streaming_tv": "No",
        "streaming_movies": "No",
        "multiple_lines": "No",
        "monthly_charges": 0.0,
        "total_charges": 0.0,
        "tenure_days": 0,
        "senior_citizen": 0
    }

    for col in expected_features:
        if col not in df.columns:
            df[col] = defaults[col]

    # Ensure column order and types
    df = df[expected_features]
    categorical_cols = [
        "type", "paperless_billing", "payment_method", "partner", "dependents",
        "internet_service", "online_security", "online_backup",
        "device_protection", "tech_support", "streaming_tv",
        "streaming_movies", "multiple_lines"
    ]
    df[categorical_cols] = df[categorical_cols].astype(str)

    st.write("DF before prediction:", df)
    st.write("DF dtypes:", df.dtypes)

    # Prediction
    proba = pipeline.predict_proba(df)[:, 1][0]
    st.metric("Churn probability", f"{proba:.2%}")
    st.write("Predicted churn:", "Yes" if proba >= 0.5 else "No")