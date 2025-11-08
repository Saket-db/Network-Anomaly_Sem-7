import smtplib
from email.mime.text import MIMEText

def send_alert(anomalies):
    '''msg = MIMEText(f"Anomaly detected:\n{anomalies}")
    msg['Subject'] = "Network Anomaly Alert"
    msg['From'] = "your_email@example.com"
    msg['To'] = "admin@example.com"

    with smtplib.SMTP("smtp.yourprovider.com", 587) as server:
        server.login("your_email@example.com", "password")
        server.sendmail("your_email@example.com", "admin@example.com", msg.as_string())'''
    print("anomaly")
