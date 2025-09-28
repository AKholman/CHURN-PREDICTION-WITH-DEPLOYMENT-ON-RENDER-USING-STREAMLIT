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

st.title("Churn prediction (XGBoost)")

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
    
    submitted = st.form_submit_button("Predict")

if submitted:
    input_dict = {
        "type": type_input,
        "paperless_billing": paperless_billing,
        "payment_method": payment_method,
        "internet_service": internet_service,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "tenure_days": tenure_days,
        "senior_citizen": senior_citizen
    }

    df = pd.DataFrame([input_dict])

    for col in expected_features:
        if col not in df.columns:
            if col in ["type", "paperless_billing", "payment_method", "internet_service",
                       "partner", "dependents", "online_security", "online_backup"]:
                df[col] = "Unknown"
            else:
                df[col] = 0

    df = df[expected_features]

    # Force categoricals to str
    categorical_cols = ["type", "paperless_billing", "payment_method", "internet_service",
        "partner", "dependents", "online_security", "online_backup"
    ]
    df[categorical_cols] = df[categorical_cols].astype(str)

    st.write("DF before prediction:", df)
    st.write("DF dtypes:", df.dtypes)

    proba = pipeline.predict_proba(df)[:, 1][0]
    st.metric("Churn probability", f"{proba:.2%}")
    st.write("Predicted churn:", "Yes" if proba >= 0.5 else "No")
