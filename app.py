import streamlit as st
import subprocess
import os
import sys
import pandas as pd
from pathlib import Path

# ...existing code...
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CAPTURE_SCRIPT = os.path.join(BASE_DIR, "capture", "live_capture.py")
DETECT_SCRIPT = os.path.join(BASE_DIR, "detection", "anomaly_detection.py")
DATA_DIR = os.path.join(BASE_DIR, "data")
ANOMALIES_CSV = os.path.join(DATA_DIR, "anomalies.csv")
LOG_FILE = os.path.join(BASE_DIR, "logs", "anomaly_detection.log")

# State management for subprocess
if "capture_process" not in st.session_state:
    st.session_state.capture_process = None

def start_capture():
    if st.session_state.capture_process is None:
        try:
            proc = subprocess.Popen(
                [sys.executable, CAPTURE_SCRIPT],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=BASE_DIR
            )
            st.session_state.capture_process = proc
            st.success("Packet capture started.")
        except Exception as e:
            st.error(f"Failed to start capture: {e}")
    else:
        st.warning("Packet capture is already running.")

def stop_capture():
    proc = st.session_state.capture_process
    if proc is not None:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        st.session_state.capture_process = None
        st.success("Packet capture stopped.")
    else:
        st.warning("No active packet capture to stop.")

def detect_anomalies():
    try:
        # run using the current python executable, capture stdout/stderr
        result = subprocess.run([sys.executable, DETECT_SCRIPT],
                                capture_output=True, text=True, cwd=BASE_DIR, timeout=300)
        st.subheader("Anomaly Detection Output")
        if result.stdout:
            st.code(result.stdout)
        if result.stderr:
            st.error(result.stderr)

        # load anomalies CSV if present
        if os.path.exists(ANOMALIES_CSV) and os.path.getsize(ANOMALIES_CSV) > 0:
            df = pd.read_csv(ANOMALIES_CSV, dtype=str)
            if not df.empty:
                st.success(f"Anomalies found: {len(df)}")
                st.dataframe(df)
                csv_bytes = df.to_csv(index=False).encode("utf-8")
                st.download_button("Download anomalies.csv", data=csv_bytes,
                                   file_name="anomalies.csv", mime="text/csv")
            else:
                st.info("No anomalies detected (file empty).")
        else:
            st.info("No anomalies detected (anomalies.csv missing).")
    except subprocess.TimeoutExpired:
        st.error("Anomaly detection timed out.")
    except Exception as e:
        st.error(f"An error occurred during anomaly detection: {e}")

# log helper (use log_module if available, otherwise read log file)
def read_logs():
    try:
        from log_module.anomaly_logging import get_logs
        return get_logs()
    except Exception:
        if os.path.exists(LOG_FILE):
            try:
                return open(LOG_FILE, encoding="utf-8", errors="ignore").read()
            except Exception as e:
                return f"Failed to read log file: {e}"
        return "(no logs available)"

st.title("Network Traffic Capture and Anomaly Detection")

col1, col2 = st.columns([1, 2])

with col1:
    if st.button("Start Capture"):
        start_capture()
    if st.button("Stop Capture"):
        stop_capture()
    if st.button("Detect Anomalies"):
        detect_anomalies()

with col2:
    st.subheader("Anomalies / Logs")
    if os.path.exists(ANOMALIES_CSV):
        try:
            df = pd.read_csv(ANOMALIES_CSV, dtype=str)
            st.metric("Anomalies", len(df))
            st.dataframe(df.head(200))
        except Exception as e:
            st.error(f"Failed to read anomalies.csv: {e}")
    else:
        st.info("No anomalies.csv present. Run detection to generate it.")

    # if st.button("Show Logs"):
    #     logs = read_logs()
    #     st.text_area("Anomaly Detection Logs", logs, height=300)

st.caption("Notes: scripts are executed with the same Python interpreter as Streamlit. Ensure virtualenv is activated before launching Streamlit.")
# ...existing code...