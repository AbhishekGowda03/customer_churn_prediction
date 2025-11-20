import streamlit as st
import joblib
import pandas as pd
import os

st.title("Customer Churn Prediction App")

#model = joblib.load('../models/churn_model.pkl')\

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "churn_model.pkl")
FEATURE_NAMES_PATH = os.path.join(BASE_DIR, "models", "feature_names.pkl")

# Load artifacts
model = joblib.load(MODEL_PATH)
feature_names = joblib.load(FEATURE_NAMES_PATH)

def predict(input_data):
    df = pd.DataFrame([input_data], columns=feature_names)
    prediction = model.predict(df)[0]
    return "Customer WILL Churn" if prediction == 1 else "Customer WILL NOT Churn"

st.header("Enter Customer Details")

user_input = {}
for feature in feature_names:
    user_input[feature] = st.number_input(feature, min_value=0.0, max_value=200.0, value=1.0)

if st.button("Predict"):
    st.success(predict(user_input))
