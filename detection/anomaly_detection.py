# ...existing code...
import os
import re
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")

def _find(candidates):
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None

# candidate artifact names (include trained_columns.pkl fallback)
cols_candidates = [
    os.path.join(MODEL_DIR, "trained_columns.pkl"),
    os.path.join(MODEL_DIR, "columns_used.pkl"),
    os.path.join(HERE, "trained_columns.pkl"),
    os.path.join(HERE, "columns_used.pkl"),
]
model_candidates = [
    os.path.join(MODEL_DIR, "isolation_forest_model.pkl"),
    os.path.join(HERE, "isolation_forest_model.pkl"),
]
scaler_candidates = [os.path.join(MODEL_DIR, "scaler.pkl"), os.path.join(HERE, "scaler.pkl")]
imputer_candidates = [os.path.join(MODEL_DIR, "imputer.pkl"), os.path.join(HERE, "imputer.pkl")]
encoders_candidates = [os.path.join(MODEL_DIR, "label_encoders.pkl"), os.path.join(HERE, "label_encoders.pkl")]

cols_file = _find(cols_candidates)
model_file = _find(model_candidates)
scaler_file = _find(scaler_candidates)
imputer_file = _find(imputer_candidates)
encoders_file = _find(encoders_candidates)

if not cols_file or not model_file:
    missing = []
    if not cols_file:
        missing.append("trained_columns.pkl / columns_used.pkl")
    if not model_file:
        missing.append("isolation_forest_model.pkl")
    raise FileNotFoundError(f"Missing artifacts: {', '.join(missing)}. Look in {MODEL_DIR} or detection/")

# load artifacts
trained_columns = list(joblib.load(cols_file))  # ensure list
model = joblib.load(model_file)
scaler = joblib.load(scaler_file) if scaler_file else None
imputer = joblib.load(imputer_file) if imputer_file else None
label_encoders = joblib.load(encoders_file) if encoders_file else {}

# locate CSV
csv_candidates = [
    os.path.join(DATA_DIR, "live_traffic.csv"),
    os.path.join(PROJECT_ROOT, "data", "live_traffic.csv"),
]
csv_path = _find(csv_candidates)
if not csv_path:
    raise FileNotFoundError(f"live_traffic.csv not found in {DATA_DIR}")

df = pd.read_csv(csv_path, dtype=str)  # read raw

# --- Minimal preprocessing: normalize port fields so numeric coercion works ---
def _first_port(val):
    if val is None:
        return np.nan
    s = str(val).strip()
    if s == "" or s.lower() == "nan":
        return np.nan
    # extract first integer occurrence (handles "61727,443" and quoted forms)
    m = re.search(r"(\d+)", s)
    return float(m.group(1)) if m else np.nan

for pcol in ("tcp.port", "udp.port"):
    if pcol in df.columns:
        df[pcol] = df[pcol].apply(_first_port)

# Build feature matrix in the exact order used during training
X = pd.DataFrame(index=df.index, columns=trained_columns, dtype=float)

for col in trained_columns:
    if col in df.columns:
        series = df[col].astype("object").copy()
        # if encoder exists for this column, map using encoder classes_ -> numeric index
        if col in label_encoders and label_encoders[col] is not None:
            le = label_encoders[col]
            # build mapping to avoid transform() errors on unseen labels
            mapping = {v: i for i, v in enumerate(getattr(le, "classes_", []))}
            X[col] = series.map(mapping).astype(float).fillna(np.nan)
        else:
            # numeric coercion for numeric features
            X[col] = pd.to_numeric(series, errors="coerce")
    else:
        X[col] = np.nan

# Impute (use saved imputer if available)
if imputer is not None:
    X_imputed = pd.DataFrame(imputer.transform(X), columns=trained_columns, index=X.index)
else:
    X_imputed = X.fillna(0)
    print("Warning: imputer.pkl not found — filled missing values with 0 (may increase false positives).")

# Scale using saved scaler — do NOT fit a new scaler
if scaler is not None:
    # scaler may have been fitted on numpy arrays; accept the UserWarning about feature names
    X_scaled = scaler.transform(X_imputed)
else:
    X_scaled = X_imputed.values
    print("Warning: scaler.pkl not found — not scaling input (model expected scaled features). This may cause many anomalies.")

# Predict
pred = model.predict(X_scaled)  # IsolationForest: -1 = anomaly, 1 = normal
df["anomaly"] = pred

# Write anomalies.csv for downstream components
out_path = os.path.join(DATA_DIR, "anomalies.csv")
anomalies = df[df["anomaly"] == -1]
anomalies.to_csv(out_path, index=False)

# optional debug output: save lowest-score rows if model supports decision_function
try:
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X_scaled)
        debug_out = anomalies.copy()
        if not debug_out.empty:
            # attach score column where available
            debug_out = debug_out.assign(score=scores[anomalies.index])
            debug_out.to_csv(os.path.join(DATA_DIR, "anomalies_debug.csv"), index=False)
except Exception:
    pass

print(f"Processed {len(df)} rows — anomalies: {len(anomalies)} written to {out_path}")
# ...existing code...