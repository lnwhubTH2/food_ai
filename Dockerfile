# ใช้ python slim เพื่อให้ image เล็ก
FROM python:3.10-slim

# ติดตั้ง system deps สำหรับ opencv (ที่ ultralytics ต้องใช้)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces ต้องการ user ที่ไม่ใช่ root
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"
ENV HOME=/home/user

WORKDIR /app

# Copy requirements ก่อน เพื่อใช้ Docker layer cache
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy โค้ดและไฟล์โมเดล
COPY --chown=user:user . .

# เปิด port 7860 (Hugging Face Spaces default)
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]