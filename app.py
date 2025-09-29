# app.py
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import json

@st.cache_resource
def load_model():
    pipeline = joblib.load("artifacts/model_pipeline.joblib")
    with open("artifacts/model_metadata.json") as f:
        meta = json.load(f)
    return pipeline, meta

pipeline, meta = load_model()
expected_features = meta['features']

st.title("💡 Churn Prediction Dashboard")

# ---- Tabs ----
tab1, tab2, tab3 = st.tabs(["Input", "Prediction", "Feature Info"])

with tab1:
    st.header("Customer Information")
    # Use columns for layout
    with st.form("predict_form"):
        col1, col2 = st.columns(2)

        with col1:
            monthly_charges = st.slider("Monthly Charges", 0.0, 200.0, 50.0)
            total_charges = st.slider("Total Charges", 0.0, 5000.0, 600.0)
            tenure_days = st.slider("Tenure Days", 0, 5000, 365)
            senior_citizen = st.radio("Senior Citizen", ["No", "Yes"])

            type_input = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
            paperless_billing = st.radio("Paperless Billing", ["Yes", "No"])
            payment_method = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
            ])
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

        with col2:
            partner = st.radio("Partner", ["Yes", "No"])
            dependents = st.radio("Dependents", ["Yes", "No"])
            online_security = st.radio("Online Security", ["Yes", "No"])
            online_backup = st.radio("Online Backup", ["Yes", "No"])
            device_protection = st.radio("Device Protection", ["Yes", "No"])
            tech_support = st.radio("Tech Support", ["Yes", "No"])
            streaming_tv = st.radio("Streaming TV", ["Yes", "No"])
            streaming_movies = st.radio("Streaming Movies", ["Yes", "No"])
            multiple_lines = st.radio("Multiple Lines", ["Yes", "No"])

        submitted = st.form_submit_button("Predict")

with tab2:
    if submitted:
        input_dict = {
            "type": type_input,
            "paperless_billing": paperless_billing,
            "payment_method": payment_method,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "tenure_days": tenure_days,
            "senior_citizen": 1 if senior_citizen=="Yes" else 0,
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

        # Defaults
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

        df = df[expected_features]

        categorical_cols = [
            "type", "paperless_billing", "payment_method", "partner", "dependents",
            "internet_service", "online_security", "online_backup",
            "device_protection", "tech_support", "streaming_tv",
            "streaming_movies", "multiple_lines"
        ]
        df[categorical_cols] = df[categorical_cols].astype(str)

        proba = pipeline.predict_proba(df)[:, 1][0]

        # Color-coded metric
        if proba >= 0.6:
            st.error(f"Churn Probability: {proba:.2%} ⚠️")
        elif proba >= 0.3:
            st.warning(f"Churn Probability: {proba:.2%} ⚠️")
        else:
            st.success(f"Churn Probability: {proba:.2%} ✅")

        st.progress(int(proba*100))

        st.write("Predicted churn:", "Yes" if proba >= 0.5 else "No")

with tab3:
    st.header("Feature Information")
    st.write("The app uses the following features for prediction:")
    st.write(expected_features)