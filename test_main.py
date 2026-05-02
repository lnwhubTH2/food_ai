import pytest
from fastapi.testclient import TestClient
from main import app
import io
from PIL import Image

# สร้าง TestClient สำหรับจำลองการยิง API
client = TestClient(app)

# ==========================================
# Test Case 1: ส่งรูปภาพปกติ (ต้องผ่านและคืนค่า JSON ที่ถูกต้อง)
# ==========================================
def test_predict_success():
    # สร้างรูปภาพจำลองขนาด 224x224 (สีแดง)
    img = Image.new("RGB", (224, 224), color="red")
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="JPEG")
    img_bytes = img_byte_arr.getvalue()

    # จำลองการแนบไฟล์ยิง API
    response = client.post(
        "/predict",
        files={"file": ("test_image.jpg", img_bytes, "image/jpeg")}
    )
    
    # ตรวจสอบว่า API ตอบกลับเป็น 200 OK
    assert response.status_code == 200
    
    # ตรวจสอบว่า JSON ที่ตอบกลับมามีฟิลด์ครบถ้วนตาม Pydantic Model
    data = response.json()
    assert data["status"] == "success"
    assert "menu_name" in data
    assert "confidence" in data
    assert "nutrition" in data

# ==========================================
# Test Case 2: ส่งไฟล์ที่ไม่ใช่รูปภาพ เช่น PDF (ต้องเด้ง 400)
# ==========================================
def test_invalid_file_type():
    # จำลองไฟล์ text/pdf
    fake_pdf_content = b"%PDF-1.4 dummy content"
    
    response = client.post(
        "/predict",
        files={"file": ("document.pdf", fake_pdf_content, "application/pdf")}
    )
    
    # ต้องตอบกลับเป็น 400 Bad Request
    assert response.status_code == 400
    assert "Invalid file type" in response.json()["detail"]

# ==========================================
# Test Case 3: ส่งไฟล์รูปภาพที่พัง (Corrupted File)
# ==========================================
def test_corrupted_image():
    # แนบไฟล์บอกว่าเป็น JPEG แต่ไส้ในเป็นข้อความมั่วๆ
    corrupted_content = b"This is not a real image data"
    
    response = client.post(
        "/predict",
        files={"file": ("corrupted.jpg", corrupted_content, "image/jpeg")}
    )
    
    # ต้องเด้ง 400 Bad Request (ดักด้วย Image.verify())
    assert response.status_code == 400

# ==========================================
# Test Case 4: ส่งไฟล์ขนาดใหญ่เกิน 5MB (ต้องเด้ง 400)
# ==========================================
def test_file_too_large():
    # สร้างไฟล์ขยะขนาด 6MB (ใหญ่กว่าที่ MAX_FILE_SIZE กำหนด)
    large_content = b"0" * (6 * 1024 * 1024)
    
    response = client.post(
        "/predict",
        files={"file": ("large_image.jpg", large_content, "image/jpeg")}
    )
    
    assert response.status_code == 400
    assert "File too large" in response.json()["detail"]