# 📉 Customer Churn Prediction (Machine Learning & Streamlit Deployment)

This project implements a machine learning solution to predict customer churn using the Telco Customer dataset. The goal is to identify customers who are likely to discontinue service based on behavioral and service-related features.

---

## 🚀 Project Features

✔ Data preprocessing & feature engineering  
✔ Random Forest model for churn prediction (~80% accuracy)  
✔ Interactive Streamlit web UI for real-time churn prediction  
✔ Visual insights & churn analytics using Plotly  
✔ End-to-end ML pipeline from dataset → model → UI → deployment  
✔ Public hosted version for demonstration

---

## 🧠 Problem Overview

Customer churn critically impacts subscription-based businesses.  
Retaining customers is more cost-effective than acquiring new ones.

📍 Objective:  
Predict whether a customer will churn using machine learning and historical telecom service usage data.

---

## 🗂 Dataset

Telco Customer Churn dataset  
Contains features such as:

- Contract Type
- Tenure
- Internet Service
- Payment Method
- Monthly Charges
- Total Charges
- Partner / Dependents
- Senior Citizen status

Target column: `Churn` (Yes/No)

---

## 🏗 Project Architecture

Dataset → Preprocessing → Model Training → Evaluation → Streamlit UI → Deployment


---

## 🛠 Tech Stack

**Languages & Libraries**  
- Python  
- Pandas  
- NumPy  
- Scikit-Learn  
- Plotly  
- Streamlit  
- Joblib

---

## 🧪 Model Training

The model was trained using RandomForestClassifier:

```python
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
