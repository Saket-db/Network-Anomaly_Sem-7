import os
import re
import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report
)

HERE = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

# --- REPLACED: discover separate training and test CSVs ---
training_candidates = [
    os.path.join(DATA_DIR, "Realistic_Large-Scale_Network_Traffic_Dataset.csv"),
    os.path.join(DATA_DIR, "labeled_traffic.csv"),
    os.path.join(PROJECT_ROOT, "data", "dataset.csv"),
    os.path.join(PROJECT_ROOT, "data", "train.csv"),
]
test_candidates = [
    os.path.join(DATA_DIR, "anomalies_debug.csv"),
    os.path.join(DATA_DIR, "anomalies.csv"),
    os.path.join(PROJECT_ROOT, "data", "test.csv"),
]

train_path = None
for p in training_candidates:
    if p and os.path.exists(p):
        train_path = p
        break

test_path = None
for p in test_candidates:
    if p and os.path.exists(p):
        test_path = p
        break

# --- CHANGED: if training CSV missing, fall back to test CSV (with warning) instead of failing ---
if not train_path:
    if test_path:
        # fallback behavior: use the test CSV as training data (NOT RECOMMENDED for real experiments)
        train_path = test_path
        warn = ("Warning: training CSV not found. Falling back to test CSV "
                f"'{os.path.basename(test_path)}' for training and evaluation. "
                "This produces a self-evaluated model and is only intended for convenience.")
        try:
            from log_module.anomaly_logging import log_anomaly as _lf
            _lf(warn)
        except Exception:
            print(warn)
    else:
        raise FileNotFoundError("Training CSV not found. Place Realistic_Large-Scale_Network_Traffic_Dataset.csv (or equivalent) into data/")

if not test_path:
    # If no explicit test CSV, try to use the training CSV as the test set (best-effort)
    if train_path:
        test_path = train_path
        warn2 = ("Warning: test CSV not found. Using the training CSV for evaluation (self-evaluation).")
        try:
            from log_module.anomaly_logging import log_anomaly as _lf2
            _lf2(warn2)
        except Exception:
            print(warn2)
    else:
        raise FileNotFoundError("Test CSV not found. Place anomalies_debug.csv (or anomalies.csv) into data/")

# load datasets
df_train = pd.read_csv(train_path, dtype=str)
df_test = pd.read_csv(test_path, dtype=str)

# If train/test do not have 'label' but have 'anomaly', derive as before
def _derive_label_column(df):
    if "label" not in df.columns and "anomaly" in df.columns:
        def _to_label(x):
            if pd.isnull(x):
                return np.nan
            s = str(x).strip().lower()
            if s in ("1", "true", "t", "yes", "y"):
                return 1
            if s in ("0", "false", "f", "no", "n"):
                return 0
            try:
                v = float(s)
                return 1 if v == 1 else 0
            except Exception:
                return 0
        df["label"] = df["anomaly"].apply(_to_label).astype(int)
    return df

df_train = _derive_label_column(df_train)
df_test = _derive_label_column(df_test)

# Minimal preprocessing helper (ports)
def _first_port(val):
    if pd.isnull(val):
        return np.nan
    s = str(val).strip()
    if s == "" or s.lower() == "nan":
        return np.nan
    m = re.search(r"(\d+)", s)
    return float(m.group(1)) if m else np.nan

for pcol in ("tcp.port", "udp.port"):
    if pcol in df_train.columns:
        df_train[pcol] = df_train[pcol].apply(_first_port)
    if pcol in df_test.columns:
        df_test[pcol] = df_test[pcol].apply(_first_port)

# Exclude leaky/model-derived columns from features
drop_cols = {"label", "frame.time_epoch", "ip.src", "ip.dst", "http.request.uri", "http.response.code", "anomaly"}
_leak_keywords = ("iso_", "rf_", "ensemble", "score", "pred", "prob", "anomaly")

def _is_leaky(colname):
    low = str(colname).lower()
    if colname in drop_cols:
        return True
    for kw in _leak_keywords:
        if kw in low:
            return True
    return False

# Feature columns = train columns excluding leak fields; ensure test uses same set (intersection)
train_feature_cols = [c for c in df_train.columns if not _is_leaky(c)]
# keep only those that exist in test as well (avoid missing features)
feature_cols = [c for c in train_feature_cols if c in df_test.columns]

# Partition numeric vs categorical (based on training set)
numeric_cols = []
categorical_cols = []
for c in feature_cols:
    coerced = pd.to_numeric(df_train[c].dropna().astype(str).str.strip(), errors="coerce")
    if coerced.notna().sum() > 0:
        numeric_cols.append(c)
    else:
        categorical_cols.append(c)

# Fit label encoders on training categorical columns
label_encoders = {}
for c in categorical_cols:
    vals = df_train[c].fillna("<<MISSING>>").astype(str)
    le = LabelEncoder()
    try:
        le.fit(vals)
        label_encoders[c] = le
    except Exception:
        label_encoders[c] = None

# Build numeric matrices for train and test using the same encoders/order
def build_matrix(df, cols, numeric_cols, categorical_cols, encoders):
    X = pd.DataFrame(index=df.index, columns=cols, dtype=float)
    for c in numeric_cols:
        if c in df.columns:
            X[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            X[c] = np.nan
    for c in categorical_cols:
        if c in df.columns:
            if encoders.get(c) is not None:
                mapping = {v: i for i, v in enumerate(getattr(encoders[c], "classes_", []))}
                X[c] = df[c].astype(object).map(mapping).astype(float)
            else:
                vals = df[c].fillna("<<MISSING>>").astype(str)
                uniques = vals.unique().tolist()
                mapping = {v: i for i, v in enumerate(uniques)}
                X[c] = vals.map(mapping).astype(float)
        else:
            X[c] = np.nan
    return X

X_train = build_matrix(df_train, feature_cols, numeric_cols, categorical_cols, label_encoders)
X_test = build_matrix(df_test, feature_cols, numeric_cols, categorical_cols, label_encoders)

# Drop columns with no observed values in training (and from test)
_all_na_cols = [c for c in X_train.columns if X_train[c].notna().sum() == 0]
if _all_na_cols:
    for col in _all_na_cols:
        try:
            from log_module.anomaly_logging import log_anomaly as _log_file
            _log_file(f"Dropping all-NaN feature column from training: {col}")
        except Exception:
            print(f"Dropping all-NaN feature column from training: {col}")
    X_train.drop(columns=_all_na_cols, inplace=True)
    X_test.drop(columns=_all_na_cols, inplace=True)

# Target vector (from training and test)
if "label" not in df_train.columns:
    raise ValueError("Training data must contain a 'label' column.")
y_train = df_train["label"].astype(int)
y_test = df_test["label"].astype(int) if "label" in df_test.columns else None

# Imputer and scaler fitted on training
imputer = SimpleImputer(strategy="median")
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# Train IsolationForest on normal-only from training
contamination = float(np.mean(y_train)) if len(np.unique(y_train)) > 1 else 0.05
contamination = min(max(contamination, 0.005), 0.2)
iso = IsolationForest(n_estimators=100, random_state=42)
if (y_train == 0).sum() > 5:
    iso.fit(X_train_scaled[y_train == 0])
else:
    iso.fit(X_train_scaled)

# Calibrate iso threshold from training
raw_train = -iso.decision_function(X_train_scaled)
iso_threshold = float(np.nanpercentile(raw_train, 100.0 * (1.0 - contamination)))
joblib.dump(iso_threshold, os.path.join(MODEL_DIR, "iso_threshold.pkl"))

# Train RandomForest on training labels (if possible)
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf_trained = True
try:
    rf.fit(X_train_scaled, y_train)
except Exception:
    rf_trained = False

# --- Evaluate on test set (anomalies_debug.csv) ---
# IsolationForest predictions (use calibrated threshold)
iso_raw_test = -iso.decision_function(X_test_scaled)
# normalized iso_prob for ensemble
minv, maxv = float(np.nanmin(iso_raw_test)), float(np.nanmax(iso_raw_test))
span = maxv - minv
iso_prob_test = (iso_raw_test - minv) / (span + 1e-12) if span > 0 else np.zeros_like(iso_raw_test)
iso_pred_test = (iso_raw_test >= iso_threshold).astype(int)  # 1 = anomaly

# RandomForest predictions (if trained)
rf_pred_test = np.zeros(len(X_test_scaled), dtype=int)
rf_prob_test = np.full(len(X_test_scaled), np.nan, dtype=float)
if rf_trained:
    rf_pred_test = rf.predict(X_test_scaled).astype(int)
    if hasattr(rf, "predict_proba"):
        classes = list(getattr(rf, "classes_", []))
        if 1 in classes:
            idx = classes.index(1)
            rf_prob_test = rf.predict_proba(X_test_scaled)[:, idx]
        else:
            rf_prob_test = (rf_pred_test == 1).astype(float)
else:
    # leave rf_prob_test as NaN — ensemble will ignore RF contribution
    pass

# Ensemble scoring (same weighting logic)
w_iso = 0.5
w_rf = 0.5
ensemble_score = np.empty(len(X_test_scaled), dtype=float)
for i in range(len(X_test_scaled)):
    parts = []
    weights = []
    parts.append(w_iso * float(iso_prob_test[i]))
    weights.append(w_iso)
    if rf_trained and not np.isnan(rf_prob_test[i]):
        parts.append(w_rf * float(rf_prob_test[i]))
        weights.append(w_rf)
    ensemble_score[i] = sum(parts) / (sum(weights) if sum(weights) > 0 else 1.0)
ensemble_pred = (ensemble_score >= 0.5).astype(int)

# Compute metrics and classification reports
def metrics_and_report(y_true, y_pred, prob=None):
    out = {}
    out["accuracy"] = accuracy_score(y_true, y_pred)
    out["precision"] = precision_score(y_true, y_pred, zero_division=0)
    out["recall"] = recall_score(y_true, y_pred, zero_division=0)
    out["f1"] = f1_score(y_true, y_pred, zero_division=0)
    out["roc_auc"] = roc_auc_score(y_true, prob) if (prob is not None and len(np.unique(y_true)) > 1) else None
    out["report"] = classification_report(y_true, y_pred, zero_division=0)
    return out

if y_test is None:
    raise ValueError("Test data must contain labels (derived from anomaly column or labeled).")

iso_res = metrics_and_report(y_test, iso_pred_test, prob=iso_prob_test)
rf_res = metrics_and_report(y_test, rf_pred_test if rf_trained else np.zeros_like(y_test), prob=rf_prob_test if rf_trained else None)
ens_res = metrics_and_report(y_test, ensemble_pred, prob=ensemble_score)

# Logging/printing per your request (accuracy + classification report)
try:
    from log_module.anomaly_logging import log_anomaly as _log_file
except Exception:
    _log_file = None

def _log(msg):
    if _log_file:
        try:
            _log_file(msg)
        except Exception:
            pass
    else:
        print(msg)

_log(f"Training dataset: {os.path.basename(train_path)}")
_log(f"Evaluation dataset: {os.path.basename(test_path)}")
_log("=== Ensemble (test) ===")
_log(f"Accuracy: {ens_res['accuracy']:.5f}")
_log("Classification Report:\n" + ens_res["report"])

# Persist artifacts (same filenames as detection expects)
joblib.dump(list(X_train.columns), os.path.join(MODEL_DIR, "trained_columns.pkl"))
joblib.dump(iso, os.path.join(MODEL_DIR, "isolation_forest_model.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
joblib.dump(imputer, os.path.join(MODEL_DIR, "imputer.pkl"))
joblib.dump(label_encoders, os.path.join(MODEL_DIR, "label_encoders.pkl"))
joblib.dump(rf if rf_trained else None, os.path.join(MODEL_DIR, "random_forest_model.pkl"))

_log(f"Saved artifacts to {MODEL_DIR}")
_log(f"Ensemble F1: {ens_res.get('f1')}")