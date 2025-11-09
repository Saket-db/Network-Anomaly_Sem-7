import logging
import os

LOG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'logs'))
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

LOG_FILE = os.path.join(LOG_DIR, "anomaly_detection.log")
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format="%(asctime)s %(message)s")


def log_anomaly(message):
    # Write to logfile only; do NOT print here to avoid duplicate terminal lines.
    logging.info(message)


def get_logs():
    with open(LOG_FILE, "r") as file:
        return file.read()
