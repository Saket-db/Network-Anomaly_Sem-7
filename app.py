import streamlit as st
import subprocess
import os
import time
import pandas as pd

# Get the absolute path to the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CAPTURE_SCRIPT = os.path.join(BASE_DIR, "capture", "live_capture.py")
DETECT_SCRIPT = os.path.join(BASE_DIR, "detection", "anomaly_detection.py")

# State management for subprocess
if 'capture_process' not in st.session_state:
    st.session_state.capture_process = None

# Function to start packet capture
def start_capture():
    if st.session_state.capture_process is None:
        try:
            st.session_state.capture_process = subprocess.Popen(
                ["python", CAPTURE_SCRIPT], stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
            )
            st.success("Packet capture started.")
        except Exception as e:
            st.error(f"Failed to start capture: {e}")
    else:
        st.warning("Packet capture is already running.")

# Function to stop packet capture
def stop_capture():
    if st.session_state.capture_process is not None:
        try:
            st.session_state.capture_process.terminate()
            st.session_state.capture_process.wait()
            st.session_state.capture_process = None
            st.success("Packet capture stopped.")
        except Exception as e:
            st.error(f"Error stopping capture: {e}")
    else:
        st.warning("No active packet capture to stop.")

def detect_anomalies():
    try:
        result = subprocess.run(["python", DETECT_SCRIPT], capture_output=True, text=True)
        st.text("Anomaly Detection Output:")
        output = result.stdout
        #st.code(output)
        # Try to load and display anomalies.csv if it exists
        ANOMALIES_CSV = 'D:/Social Media/NetworkAnomaly_Detect/data/anomalies.csv'
        ANOMALIES_CSV = os.path.abspath(ANOMALIES_CSV)
        if os.path.exists(ANOMALIES_CSV) and os.path.getsize(ANOMALIES_CSV) > 0:
            df = pd.read_csv(ANOMALIES_CSV)
            if not df.empty:
                st.success("Anomaly detected!")
                st.dataframe(df)
            else:
                st.info("No anomalies detected.")
        else:
            st.info("No anomalies detected.")
    except Exception as e:
        st.error(f"An error occurred during anomaly detection: {e}")

import io
from log_module.anomaly_logging import get_logs

st.title("Network Traffic Capture and Anomaly Detection")

if st.button("Start Capture"):
    start_capture()

if st.button("Stop Capture"):
    stop_capture()

if st.button("Detect Anomalies"):
    detect_anomalies()

if st.button("Show Logs"):
    logs = get_logs()
    st.text_area("Anomaly Detection Logs", logs, height=300) 