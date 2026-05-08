import os
import sys
import time
import subprocess

# ใช้ absolute path เพื่อกัน working-dir แปลกๆ ตอนรันใน Docker
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_APP = os.path.join(BASE_DIR, "web_app.py")

# ตรวจสอบไฟล์สำคัญก่อนเริ่ม จะได้รู้ทันทีถ้าไฟล์หาย
if not os.path.exists(WEB_APP):
    print(f"❌ Fatal: web_app.py not found at {WEB_APP}")
    print("   ตรวจสอบว่า .dockerignore ไม่ได้ exclude ไฟล์นี้")
    sys.exit(1)

print("🚀 Starting FastAPI on port 8000...")
fastapi = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"],
    cwd=BASE_DIR,
)

print("🎨 Starting Streamlit on port 7860...")
streamlit = subprocess.Popen(
    [
        sys.executable, "-m", "streamlit", "run", WEB_APP,
        "--server.port", "7860",
        "--server.address", "0.0.0.0",
        "--server.enableXsrfProtection", "false",
        "--server.enableCORS", "false",
        "--server.headless", "true",
    ],
    cwd=BASE_DIR,
)

# ถ้า process ตัวใดตัวหนึ่งตาย -> kill อีกตัว แล้ว exit
# (กัน Hugging Face Space ค้างอยู่ในสถานะ half-broken)
try:
    while True:
        if fastapi.poll() is not None:
            print(f"❌ FastAPI exited with code {fastapi.returncode}")
            streamlit.terminate()
            sys.exit(fastapi.returncode or 1)
        if streamlit.poll() is not None:
            print(f"❌ Streamlit exited with code {streamlit.returncode}")
            fastapi.terminate()
            sys.exit(streamlit.returncode or 1)
        time.sleep(2)
except KeyboardInterrupt:
    fastapi.terminate()
    streamlit.terminate()
