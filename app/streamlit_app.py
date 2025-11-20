import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ----------------------------------
# Load model + features
# ----------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "churn_model.pkl")
FEATURE_PATH = os.path.join(BASE_DIR, "models", "feature_names.pkl")

model = joblib.load(MODEL_PATH)
feature_names = joblib.load(FEATURE_PATH)

st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

st.title("📉 Customer Churn Prediction App")
st.markdown("Enter customer details below to predict the likelihood of churn.")

# ----------------------------------
# USER INPUT FORM (clean UI)
# ----------------------------------

st.header("Enter Customer Details")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", ["Yes", "No"])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])

with col2:
    tenure = st.slider("Tenure (Months)", 0, 72, 1)
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    payment = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"]
    )
    monthly = st.slider("Monthly Charges (€)", 0, 150, 50)
    total = st.slider("Total Charges (€)", 0, 10000, 500)


# ----------------------------------
# Convert user data → model features
# ----------------------------------

user_input = {
    "gender": gender,
    "SeniorCitizen": 1 if senior == "Yes" else 0,
    "Partner": 1 if partner == "Yes" else 0,
    "Dependents": 1 if dependents == "Yes" else 0,
    "tenure": tenure,
    "PhoneService": 1 if phone_service == "Yes" else 0,
    "PaperlessBilling": 1 if paperless == "Yes" else 0,
    "MonthlyCharges": monthly,
    "TotalCharges": total,
    "Contract": contract,
    "InternetService": internet,
    "PaymentMethod": payment
}

# Convert to DataFrame
input_df = pd.DataFrame([user_input])

# One-hot encode to match training data
input_encoded = pd.get_dummies(input_df)

# Add missing feature columns
for col in feature_names:
    if col not in input_encoded.columns:
        input_encoded[col] = 0

# Ensure correct column order
input_encoded = input_encoded[feature_names]

# ----------------------------------
# Predict Churn
# ----------------------------------
if st.button("Predict Churn"):
    prediction = model.predict(input_encoded)[0]
    prob = model.predict_proba(input_encoded)[0][1]

    st.subheader("🔮 Prediction Result")

    if prediction == 1:
        st.error(f"⚠️ Customer is **likely to churn**. (Probability: {prob:.2f})")
    else:
        st.success(f"✅ Customer is **likely to stay**. (Probability: {prob:.2f})")

# ----------------------------------
# Add Visual Insights
# ----------------------------------

st.header("📊 Customer Churn Insights")

# Load original dataset
csv_path = os.path.join(BASE_DIR, "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df = pd.read_csv(csv_path)

df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

import plotly.express as px

colA, colB = st.columns(2)

with colA:
    fig = px.histogram(df, x="Churn", title="Churn Distribution", color="Churn")
    st.plotly_chart(fig, use_container_width=True)

with colB:
    fig = px.box(df, x="Churn", y="tenure", title="Tenure vs Churn")
    st.plotly_chart(fig, use_container_width=True)
