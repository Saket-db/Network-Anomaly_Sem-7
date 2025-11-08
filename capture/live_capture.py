import subprocess
import os
import signal
import platform
import time

# Global variable to hold the subprocess reference
capture_process = None
fl = False  # Flag to indicate if capturing is active

def start_capture():
    global capture_process, fl

    # Ensure the 'data' directory exists
    data_dir = r"D:\Social Media\NetworkAnomaly_Detect\data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # File path for the output CSV
    file_path = os.path.join(data_dir, "live_traffic.csv")

    # Full path to tshark.exe (adjust if different)
    tshark_path = r"C:\Program Files\Wireshark\tshark.exe"

    # Define the tshark command to capture the necessary fields
    tshark_command = [
        tshark_path, "-i", "4", "-T", "fields",
        "-e", "frame.time_epoch",
        "-e", "ip.src",
        "-e", "ip.dst",
        "-e", "frame.len",
        "-e", "tcp.port",
        "-e", "udp.port",
        "-e", "ip.proto",
        "-e", "frame.time_delta",
        "-e", "tcp.flags",
        "-e", "ip.len",
        "-e", "icmp.type",
        "-e", "tcp.stream",
        "-e", "tcp.seq",
        "-e", "tcp.ack",
        "-e", "ip.flags.df",
        "-e", "tcp.window_size",
        "-e", "ip.ttl",
        "-e", "tcp.analysis.flags",
        "-e", "http.request.uri",
        "-e", "http.response.code",
        "-E", "header=y",
        "-E", "separator=,",
        "-E", "quote=d",
    ]

    try:
        # Start the tshark command as a subprocess
        if platform.system() == "Windows":
            capture_process = subprocess.Popen(
                tshark_command,
                stdout=open(file_path, 'w'),
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                text=True
            )
        else:
            capture_process = subprocess.Popen(
                tshark_command,
                stdout=open(file_path, 'w'),
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid,
                text=True
            )
        fl = True
        print("Traffic capture started.")
    except Exception as e:
        print(f"Error during traffic capture: {e}")

def stop_capture():
    global capture_process, fl
    if capture_process is not None and fl:
        try:
            if platform.system() == "Windows":
                capture_process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                os.killpg(os.getpgid(capture_process.pid), signal.SIGTERM)

            # Print any stderr output from tshark
            stderr_output = capture_process.stderr.read()
            if stderr_output:
                print("TShark stderr:", stderr_output)

            capture_process.wait()
            print("Traffic capture stopped.")
        except Exception as e:
            print(f"Error stopping capture: {e}")
        finally:
            capture_process = None
            fl = False
    else:
        print("No active capture process to stop.")

if __name__ == "__main__":
    try:
        start_capture()
        time.sleep(15)  # Capture packets for 15 seconds
        stop_capture()
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received. Stopping capture...")
        stop_capture()