# ...existing code...
import os
import pandas as pd
import joblib
import numpy as np

from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

# Set up base directory relative to this script
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_DIR = os.path.join(BASE_DIR, 'model')
DATA_DIR = os.path.join(BASE_DIR, 'data')

os.makedirs(MODEL_DIR, exist_ok=True)

# Locate dataset (data/ preferred, fallback to model/)
dataset_path = os.path.join(DATA_DIR, 'Realistic_Large-Scale_Network_Traffic_Dataset.csv')
if not os.path.exists(dataset_path):
    alt = os.path.join(MODEL_DIR, 'Realistic_Large-Scale_Network_Traffic_Dataset.csv')
    if os.path.exists(alt):
        dataset_path = alt
    else:
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}\nAlso checked: {alt}\n"
            "Place the CSV in the project's data/ or model/ directory or update the filename in ml_model.py."
        )

# Load dataset
df_large = pd.read_csv(dataset_path)

if 'label' not in df_large.columns:
    raise RuntimeError("Expected 'label' column in dataset for supervised evaluation/training.")

# Split features and target
X = df_large.drop(columns=['label']).copy()
y = pd.to_numeric(df_large['label'], errors='coerce').fillna(0).astype(int)

# Initialize and fit label encoders for categorical/high-cardinality features if present
label_encoders = {}
for col in ['ip.src', 'ip.dst', 'http.request.uri']:
    if col in X.columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

# Persist label encoders
joblib.dump(label_encoders, os.path.join(MODEL_DIR, 'label_encoders.pkl'))

# Save trained column order (use a plain list)
trained_columns = list(X.columns)
joblib.dump(trained_columns, os.path.join(MODEL_DIR, 'trained_columns.pkl'))

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y))>1 else None)

# Handle missing values with SimpleImputer (most frequent)
imputer = SimpleImputer(strategy='most_frequent')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)
joblib.dump(imputer, os.path.join(MODEL_DIR, 'imputer.pkl'))

# Scale features and persist scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)
joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))

# Fit IsolationForest on scaled training data
clf = IsolationForest(n_estimators=100, max_samples='auto', contamination=0.05, random_state=42)
clf.fit(X_train_scaled)

# Persist trained model
joblib.dump(clf, os.path.join(MODEL_DIR, 'isolation_forest_model.pkl'))

# Predict on the scaled test set (use scaled data)
y_pred_raw = clf.predict(X_test_scaled)    # -1 = anomaly, 1 = normal
y_pred = np.where(y_pred_raw == -1, 1, 0)  # map to 1=anomaly, 0=normal (matches typical label meaning)

# Ensure y_test is integer numpy array for metrics
y_test_arr = np.asarray(y_test).astype(int)

# Evaluate
accuracy = accuracy_score(y_test_arr, y_pred)
print(f"Model Accuracy: {accuracy:.5f}")
print("Classification Report:")
print(classification_report(y_test_arr, y_pred, zero_division=0))

# ...existing code...