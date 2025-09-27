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
    
    # numeric features
    tenure = st.number_input("tenure_months", min_value=0, max_value=200, value=12)
    monthly_charges = st.number_input("monthly_charges", min_value=0.0, value=50.0)
    total_charges = st.number_input("total_charges", min_value=0.0, value=600.0)
    tenure_days = st.number_input("tenure_days", min_value=0, max_value=5000, value=365)

    # categorical/binary features
    contract_type = st.selectbox("Contract type", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.radio("Paperless billing", ["Yes", "No"])
    payment_method = st.selectbox("Payment method", 
                                  ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
    senior_citizen = st.radio("Senior citizen", ["Yes", "No"])
    partner = st.radio("Has partner", ["Yes", "No"])
    internet_service = st.selectbox("Internet service", ["DSL", "Fiber optic", "None"])
    tech_support = st.selectbox("Tech support", ["Yes", "No", "None"])
    
    submitted = st.form_submit_button("Predict")

if submitted:
    input_dict = {
        "type": contract_type,
        "paperless_billing": paperless_billing,
        "payment_method": payment_method,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "tenure_days": tenure_days,
        "senior_citizen": senior_citizen,
        "partner": partner,
        "internet_service": internet_service,
        "tech_support": tech_support,
        # fill the rest with default values or expose them in UI
    }
    df = pd.DataFrame([input_dict])

    # Ensure correct column order and add missing columns as NaN
    df = df.reindex(columns=expected_features, fill_value=np.nan)

    # Predict
    proba = pipeline.predict_proba(df)[:, 1][0] # probability of class 1 (churn)
    st.metric("Churn probability", f"{proba:.2%}")
    st.write("Predicted churn:", "Yes" if proba >= 0.5 else "No")

    # optional: thresholded prediction
    threshold = 0.5
    st.write("Predicted churn:", "Yes" if proba >= threshold else "No")
