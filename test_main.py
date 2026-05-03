import io
from PIL import Image
from fastapi.testclient import TestClient
from main import app  # ดึง API จากไฟล์ main.py ของเรามา

# สร้าง TestClient เพื่อจำลองการทำงานของแอปพลิเคชัน
client = TestClient(app)

def generate_dummy_image():
    """ฟังก์ชันสร้างรูปภาพจำลองขนาด 512x512 เพื่อใช้ทดสอบ"""
    img = Image.new("RGB", (512, 512), color="blue")
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="PNG")
    return img_byte_arr.getvalue()

def test_predict_endpoint_success():
    """ทดสอบกรณีส่งรูปภาพที่ถูกต้องเข้าไป (ควรได้ Status 200)"""
    dummy_img = generate_dummy_image()
    
    # จำลองการส่งไฟล์รูปผ่าน POST request
    response = client.post(
        "/predict",
        files={"file": ("test_dummy.png", dummy_img, "image/png")}
    )
    
    # 1. เช็คว่าเซิร์ฟเวอร์ตอบกลับว่าสำเร็จ (200 OK)
    assert response.status_code == 200
    
    # 2. เช็คว่าข้อมูลที่ส่งกลับมาเป็นโครงสร้าง JSON ที่ถูกต้องตาม Pydantic
    data = response.json()
    assert data["status"] == "success"
    assert "menu_name" in data
    assert "confidence" in data
    assert "nutrition" in data
    
    # 3. เช็คว่าใน nutrition มีฟิลด์ครบ
    nutrition = data["nutrition"]
    assert "calories" in nutrition
    assert "protein" in nutrition
    assert "carbs" in nutrition
    assert "fat" in nutrition

def test_predict_endpoint_invalid_file():
    """ทดสอบกรณีส่งไฟล์ที่ไม่ใช่รูปภาพเข้าไป (ควรโดนดักและได้ Status 400)"""
    invalid_content = b"This is a text file, not an image."
    
    response = client.post(
        "/predict",
        files={"file": ("test.txt", invalid_content, "text/plain")}
    )
    
    # โค้ดใน main.py ของเราน่าจะดักไว้และตอบกลับเป็น 400 Bad Request
    assert response.status_code == 400