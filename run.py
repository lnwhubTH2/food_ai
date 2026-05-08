import subprocess
import sys

print("🚀 Starting FastAPI on port 8000...")
# ใช้ sys.executable (-m) เพื่อรับประกันว่าหา uvicorn เจอแน่นอน
fastapi = subprocess.Popen([sys.executable, "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"])

print("🎨 Starting Streamlit on port 7860...")
# ใช้ sys.executable (-m) เพื่อรับประกันว่าหา streamlit เจอแน่นอน
streamlit = subprocess.Popen([
    sys.executable, "-m", "streamlit", "run", "web_app.py", 
    "--server.port", "7860", 
    "--server.address", "0.0.0.0",
    "--server.enableXsrfProtection", "false",
    "--server.enableCORS", "false"
])

# รอให้ทั้งสองระบบทำงานไปเรื่อยๆ โดยไม่ดับ
fastapi.wait()
streamlit.wait()