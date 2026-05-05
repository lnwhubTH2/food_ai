"""
Unit tests for Food Classification API
Run: pytest test_main.py -v

Note: ตั้ง TESTING=1 ก่อน import app เพื่อใช้ mock pipeline
       ทำให้ test ไม่ต้องโหลดโมเดลจริง (เร็วกว่า + ไม่ต้องมีไฟล์ .onnx/.pt บน CI)
"""
import os
# ⚠️ ต้องตั้งก่อน import main!
os.environ["TESTING"] = "1"

import io
from PIL import Image
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def generate_dummy_image(fmt: str = "PNG", size=(512, 512), color="blue"):
    """สร้างรูปภาพจำลองในหน่วยความจำ"""
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


# ==========================================
# Health endpoints
# ==========================================
def test_root_endpoint():
    """GET / ต้องตอบกลับ 200 พร้อม JSON มี status"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "message" in data


def test_health_endpoint():
    """GET /health ต้องตอบกลับ 200"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


# ==========================================
# /predict — Success case (ใช้ mock)
# ==========================================
def test_predict_endpoint_success():
    """ส่งรูป PNG ที่ถูกต้อง → 200 พร้อม schema ครบ"""
    img_bytes = generate_dummy_image("PNG")
    response = client.post(
        "/predict",
        files={"file": ("test.png", img_bytes, "image/png")}
    )
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    data = response.json()
    assert data["status"] == "success"
    assert "menu_name" in data
    assert "confidence" in data
    assert "inference_time_ms" in data
    assert "nutrition" in data
    nut = data["nutrition"]
    for key in ["calories", "protein", "carbs", "fat"]:
        assert key in nut
    assert 0 <= data["confidence"] <= 1


def test_predict_endpoint_jpeg():
    """ส่งรูป JPEG → 200"""
    img_bytes = generate_dummy_image("JPEG")
    response = client.post(
        "/predict",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")}
    )
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"


# ==========================================
# /predict — Error cases
# ==========================================
def test_predict_invalid_file_type():
    """ส่งไฟล์ text → 400"""
    invalid = b"This is not an image."
    response = client.post(
        "/predict",
        files={"file": ("test.txt", invalid, "text/plain")}
    )
    assert response.status_code == 400
    assert "Invalid file type" in response.json()["detail"]


def test_predict_corrupted_image():
    """ส่ง bytes ที่บอกว่าเป็น PNG แต่จริงๆ เสีย → 400"""
    corrupt = b"\x89PNG\r\n\x1a\nNOT_A_REAL_PNG"
    response = client.post(
        "/predict",
        files={"file": ("corrupt.png", corrupt, "image/png")}
    )
    assert response.status_code == 400


def test_predict_empty_file():
    """ส่งไฟล์ว่าง → 400"""
    response = client.post(
        "/predict",
        files={"file": ("empty.png", b"", "image/png")}
    )
    assert response.status_code == 400


def test_predict_no_file():
    """ไม่ส่งไฟล์เลย → 422 (Pydantic validation)"""
    response = client.post("/predict")
    assert response.status_code == 422