<!-- filepath: d:\Social Media\NetworkAnomaly_Detect\README.md -->
# NetworkAnomaly_Detect — README

Overview
--------
This repo captures live network traffic, runs an IsolationForest anomaly detector, and exposes a small Streamlit UI. Typical workflow:
1. Train/generate model artifacts (run the ML training script).
2. Capture live traffic into data/live_traffic.csv.
3. Run anomaly detection to produce data/anomalies.csv (or use the Streamlit app to orchestrate these steps).

Prerequisites
-------------
- Python 3.8+
- tshark (Wireshark) installed and on Windows the full path to tshark.exe must be correct.
- Recommended Python packages: pandas, numpy, scikit-learn, joblib, streamlit
  Install quickly:
  pip install pandas numpy scikit-learn joblib streamlit

Key files & locations
---------------------
- model/ml_model.py — trains model and saves artifacts to model/:
  - model/isolation_forest_model.pkl
  - model/scaler.pkl
  - model/imputer.pkl
  - model/label_encoders.pkl
  - model/trained_columns.pkl
  - model/random_forest_model.pkl  # Optional: supervised RF trained on the same labeled CSV when >=2 classes
- capture/live_capture.py — runs tshark and writes data/live_traffic.csv
- detection/anomaly_detection.py — loads artifacts and writes data/anomalies.csv
- app.py — Streamlit UI to start/stop capture and run detection
- logs/anomaly_detection.log — detection logging output

Quick start (recommended order)
-------------------------------
1) Prepare training dataset
   - Place Realistic_Large-Scale_Network_Traffic_Dataset.csv into data/ or model/.

2) Train model (must run before detection)
   - From project root:
     python model/ml_model.py
   - This creates the trained artifacts used by anomaly_detection.py.

3) Capture live traffic
   - Edit capture/live_capture.py:
     - Set tshark_path (default: `C:\Program Files\Wireshark\tshark.exe`).
     - Set the interface index (`-i` argument) appropriate for your system.
   - Start capture (one of):
     - Run manually (short demo run included in __main__):  
       python capture/live_capture.py
     - Or use the Streamlit app (next step) to start capture in background.

4) Run detection
   - Ensure data/live_traffic.csv exists (created by capture).
   - Run:
     python detection/anomaly_detection.py
   - Output: data/anomalies.csv (and optional anomalies_debug.csv). Check logs in logs/.

5) Streamlit UI (optional)
   - Start the UI:
     streamlit run app.py
   - Use buttons to start/stop capture and run detection. The app uses the same python interpreter.

Notes & troubleshooting
-----------------------
- ml_model.py expects a labeled CSV with a `label` column. If not present, training aborts.
- If detection raises missing artifact errors, re-run model/ml_model.py to recreate model/*.pkl.
- If anomalies are unexpectedly many or few, check:
  - The scaler/imputer/label encoders were created by training and loaded during detection.
  - live_traffic.csv column names match training columns (trained_columns.pkl).
- On Windows, stopping tshark may require CTRL-BREAK; live_capture.py attempts to handle this. If a subprocess remains, kill it from Task Manager.
- Permissions: ensure the script can write to data/ and logs/ directories.
- If you want a reproducible environment, create a requirements.txt:
  pandas
  numpy
  scikit-learn
  joblib
  streamlit

Contact / Next steps
--------------------
- Adjust feature list or model parameters in model/ml_model.py for better results.
- Add a requirements.txt and a small Dockerfile if you need containerized runs.
