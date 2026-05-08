import subprocess

print("🚀 Starting FastAPI on port 8000...")
fastapi = subprocess.Popen(["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"])

print("🎨 Starting Streamlit on port 7860...")
streamlit = subprocess.Popen([
    "streamlit", "run", "web_app.py", 
    "--server.port", "7860", 
    "--server.address", "0.0.0.0",
    "--server.enableXsrfProtection", "false",
    "--server.enableCORS", "false"
])

# รอให้ทั้งสองระบบทำงานไปเรื่อยๆ โดยไม่ดับ
fastapi.wait()
streamlit.wait()