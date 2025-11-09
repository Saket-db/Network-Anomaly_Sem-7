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

# --- new: random forest model candidates ---
rf_candidates = [os.path.join(MODEL_DIR, "random_forest_model.pkl"), os.path.join(HERE, "random_forest_model.pkl")]

cols_file = _find(cols_candidates)
model_file = _find(model_candidates)
scaler_file = _find(scaler_candidates)
imputer_file = _find(imputer_candidates)
encoders_file = _find(encoders_candidates)
rf_file = _find(rf_candidates)  # may be None

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
rf_model = joblib.load(rf_file) if rf_file else None  # may remain None

# load calibrated ISO threshold if present
iso_threshold = None
iso_thr_file = os.path.join(MODEL_DIR, "iso_threshold.pkl")
try:
    if os.path.exists(iso_thr_file):
        iso_threshold = float(joblib.load(iso_thr_file))
except Exception:
    iso_threshold = None

# try to use the project's logging helper; fall back to no-op function
_log_to_file = None
try:
	from log_module.anomaly_logging import log_anomaly as _log_func
	_log_to_file = _log_func
except Exception:
	pass

def _safe_log(msg):
	"""Log to file only if logger available, never print from here"""
	if _log_to_file:
		try:
			_log_to_file(msg)
		except Exception:
			pass

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
    # scaler may have been fitted on numpy arrays; pass a numpy array to avoid
    # "X has feature names, but StandardScaler was fitted without feature names" warning.
    X_scaled = scaler.transform(X_imputed.values)
else:
    X_scaled = X_imputed.values
    print("Warning: scaler.pkl not found — not scaling input (model expected scaled features). This may cause many anomalies.")

# Predict with IsolationForest
pred = model.predict(X_scaled)  # IsolationForest: -1 = anomaly, 1 = normal
df["anomaly"] = pred

# ----------------- Begin replacement / augmentation -----------------
# Replace the simple pred -> df["anomaly"] logic with fused ensemble logic.

# IsolationForest raw prediction and optional score
iso_pred_raw = pred  # -1 anomaly, 1 normal
df["iso_anomaly"] = (iso_pred_raw == -1)

# compute an IsolationForest "anomaly probability" (0..1, higher -> more anomalous)
iso_prob = None
if hasattr(model, "decision_function"):
    try:
        # decision_function: higher -> more normal typically, so invert to make higher -> more anomalous
        iso_raw_score = -model.decision_function(X_scaled)
        # normalize to 0..1 for ranking/ensemble
        minv = float(np.nanmin(iso_raw_score))
        maxv = float(np.nanmax(iso_raw_score))
        span = maxv - minv
        if span <= 0:
            iso_prob = np.zeros_like(iso_raw_score, dtype=float)
        else:
            iso_prob = (iso_raw_score - minv) / (span + 1e-12)
        df["iso_score"] = iso_raw_score
        df["iso_prob"] = iso_prob
        # Decide anomaly using calibrated threshold when available; otherwise fall back to model.predict
        if iso_threshold is not None:
            df["iso_anomaly"] = df["iso_score"] >= float(iso_threshold)
        else:
            df["iso_anomaly"] = (iso_pred_raw == -1)
    except Exception:
        iso_prob = (iso_pred_raw == -1).astype(float)
        df["iso_prob"] = iso_prob
        df["iso_anomaly"] = (iso_pred_raw == -1)
else:
    iso_prob = (iso_pred_raw == -1).astype(float)
    df["iso_prob"] = iso_prob
    df["iso_anomaly"] = (iso_pred_raw == -1)

# RandomForest-derived probability / vote (if available)
df["rf_pred"] = ""
df["rf_prob"] = np.nan
rf_count = 0

if rf_model is not None:
    try:
        # Get RF predictions - class 1 = anomaly, class 0 = normal (as trained)
        rf_pred_raw = rf_model.predict(X_scaled)
        df["rf_pred"] = rf_pred_raw.astype(str)
        
        # Count anomalies: RF predicts class 1 for anomalies
        rf_count = int((rf_pred_raw == 1).sum())
        
        # Get probabilities if available
        if hasattr(rf_model, "predict_proba"):
            proba = rf_model.predict_proba(X_scaled)
            classes = list(getattr(rf_model, "classes_", []))
            
            # Find probability column for class 1 (anomaly)
            if 1 in classes:
                idx = classes.index(1)
                df["rf_prob"] = proba[:, idx]
            else:
                # Fallback: if no class 1, use prediction as probability
                df["rf_prob"] = (rf_pred_raw == 1).astype(float)
        else:
            # No predict_proba, use hard predictions as 0/1 probabilities
            df["rf_prob"] = (rf_pred_raw == 1).astype(float)
            
    except Exception as e:
        print(f"Warning: RandomForest prediction failed: {e}")
        rf_count = 0

# Ensemble fusion (weighted average of iso_prob and rf_prob where available)
# Configurable weights and threshold
w_iso = 0.5
w_rf = 0.5
ensemble_score = np.array([np.nan] * len(df), dtype=float)

# compute per-row ensemble score respecting missing rf_prob and only include RF if rf_count > 0
for i in range(len(df)):
	parts = []
	weights = []
	# IsolationForest always contributes
	ip = float(df["iso_prob"].iat[i]) if not pd.isna(df["iso_prob"].iat[i]) else 0.0
	parts.append(w_iso * ip)
	weights.append(w_iso)
	# RF contributes only when rf_prob is available (not NaN) AND rf_count > 0
	if rf_count > 0 and not pd.isna(df["rf_prob"].iat[i]):
		try:
			rp = float(df["rf_prob"].iat[i])
			parts.append(w_rf * rp)
			weights.append(w_rf)
		except Exception:
			# skip malformed rf_prob
			pass
	# normalize by sum of used weights to keep score in 0..1
	weight_sum = sum(weights) if sum(weights) > 0 else 1.0
	ensemble_score[i] = sum(parts) / weight_sum

df["ensemble_score"] = ensemble_score

# decide anomaly from ensemble_score with threshold (default 0.5)
ensemble_threshold = 0.5
df["ensemble_anomaly"] = df["ensemble_score"] >= ensemble_threshold

# for backward compatibility, keep the original isolation prediction column
df["anomaly"] = iso_pred_raw

# --- NEW: compute and log summary counts ONCE ---
try:
	total_rows = int(len(df))
	iso_count = int(df["iso_anomaly"].sum()) if "iso_anomaly" in df.columns else int((pred == -1).sum())
	ensemble_count = int(df["ensemble_anomaly"].sum())

	# Print to console ONCE
	print(f"Processed data: {total_rows} rows")
	print(f"Anomalies detected by IsolationForest: {iso_count}")
	# print(f"Anomalies Classified by the Random Forest: {rf_count}")
	print(f"Anomalies detected by Ensemble model: {ensemble_count}")
	
	# Log to file separately (no printing from log function)
	_safe_log(f"Processed data: {total_rows} rows")
	_safe_log(f"Anomalies detected by IsolationForest: {iso_count}")
	_safe_log(f"Anomalies Classified by the Random Forest: {rf_count}")
	_safe_log(f"Anomalies detected by Ensemble model: {ensemble_count}")
		
except Exception as _e:
	print(f"Warning: failed to compute/log summary counts: {_e}")

# Write anomalies.csv for downstream components (use ensemble decision by default)
out_path = os.path.join(DATA_DIR, "anomalies.csv")
anomalies = df[df["ensemble_anomaly"] == True].copy()
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

# Also write a combined debug CSV (includes RF predictions and ensemble info)
try:
    df.to_csv(os.path.join(DATA_DIR, "detection_full_output.csv"), index=False)
except Exception:
    pass

print(f"Processed {len(df)} rows — ensemble anomalies: {len(anomalies)} written to {out_path}")