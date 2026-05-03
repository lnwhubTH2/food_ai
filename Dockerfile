# 1. เลือก Base Image เป็น Python 3.10 (เวอร์ชันเล็กกะทัดรัด)
FROM python:3.10-slim

# 2. ตั้งค่าโฟลเดอร์ทำงานภายใน Docker
WORKDIR /app

# 3. ติดตั้ง System Dependencies ที่ YOLO/OpenCV จำเป็นต้องใช้ (สำคัญมาก! ขาดไป YOLO รันไม่ขึ้น)
# 3. ติดตั้ง System Dependencies (อัปเดตชื่อแพ็กเกจให้เข้ากับ Debian เวอร์ชันใหม่)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4. ก๊อปปี้ไฟล์ requirements.txt เข้าไปก่อน
COPY requirements.txt .

# 5. สั่งติดตั้งไลบรารี Python ทั้งหมด
RUN pip install --no-cache-dir -r requirements.txt

# 6. ก๊อปปี้ไฟล์โค้ดและโมเดลทั้งหมดของเราเข้าไปใน Docker
COPY . .

# 7. เปิดพอร์ต 8000 ให้ภายนอกเชื่อมต่อเข้ามาได้
EXPOSE 8000

# 8. คำสั่งรัน FastAPI ทันทีที่ Container เริ่มทำงาน
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]