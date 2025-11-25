import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from data_preprocessing import preprocess_features

# Load CSV
df = pd.read_csv("../data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# TARGET (processed separately)
y = df["Churn"].map({"Yes": 1, "No": 0})

# FEATURES (only features are preprocessed)
X = df.drop("Churn", axis=1)
X = preprocess_features(X)

# Train split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model + features
joblib.dump(model, "../models/churn_model.pkl")
joblib.dump(list(X.columns), "../models/feature_names.pkl")
