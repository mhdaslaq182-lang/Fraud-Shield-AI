# =============================================
#   Start Script — Runs both Flask + Streamlit
#   start.py
# =============================================
import subprocess
import sys
import os
import time

print("="*55)
print("  AI FRAUD DETECTION SYSTEM")
print("  Sri Lanka Edition - CG02 Group 07")
print("  Starting all services...")
print("="*55)

# Start Streamlit in background
print("\n  Starting Streamlit dashboard...")
streamlit = subprocess.Popen([
    sys.executable, "-m", "streamlit", "run", "app.py",
    "--server.port", "8501",
    "--server.address", "0.0.0.0",
    "--server.enableCORS", "false",
    "--server.headless", "true"
])
print("  Streamlit started on port 8501")

time.sleep(3)

# Start Flask
print("\n  Starting Flask website...")
os.chdir("flask_app")
print("  Flask starting on port 5000")
print("="*55)
print("  Open: http://localhost:5000")
print("  Dashboard: http://localhost:8501")
print("="*55)

flask = subprocess.Popen([
    sys.executable, "app.py"
])

# Wait for both
try:
    flask.wait()
except KeyboardInterrupt:
    print("\n  Shutting down...")
    streamlit.terminate()
    flask.terminate()
