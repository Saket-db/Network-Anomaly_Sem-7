from prometheus_client import start_http_server, Summary, Counter, Gauge
import random
import time

REQUEST_TIME = Summary('request_processing_seconds', 'Time spent processing request')
REQUEST_COUNTER = Counter('http_requests_total', 'Total HTTP Requests')
GAUGE = Gauge('live_anomalies', 'Current number of live anomalies')

def process_request(t):
    REQUEST_COUNTER.inc()  # Increment the counter
    with REQUEST_TIME.time():  # Timer
        time.sleep(t)

if __name__ == '__main__':
    start_http_server(8000)  # Expose metrics on port 8000
    while True:
        process_request(random.random())
        GAUGE.set(random.randint(0, 100))  # Example gauge update
