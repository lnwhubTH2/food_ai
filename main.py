import io
import os
import json
import asyncio
import numpy as np
import onnxruntime as ort
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from pydantic import BaseModel
from concurrent.futures import ProcessPoolExecutor
from ultralytics import YOLO

# ==========================================
# 1. Pydantic Models (ตรวจสอบรูปแบบ Data)
# ==========================================
class NutritionData(BaseModel):
    calories: int
    protein: float
    carbs: float
    fat: float

class PredictResponse(BaseModel):
    status: str
    menu_name: str
    confidence: float
    nutrition: NutritionData

# ==========================================
# 2. โหลดโมเดล (Global Level)
# ==========================================
# โหลด YOLO (มันจะโหลดไฟล์ yolo11n.pt มาให้อัตโนมัติถ้ายังไม่มี)
yolo_model = YOLO("yolo11n.pt") 

# โหลดโมเดล ViT แบบ ONNX ที่คุณแปลงไว้ (ใส่ชื่อไฟล์ของคุณให้ถูกนะครับ)
vit_session = ort.InferenceSession("best.onnx")
# โหลดไฟล์ Classes ที่เราเพิ่งสร้าง
with open("classes.json", "r", encoding="utf-8") as f:
    classes_list = json.load(f)

# โหลด Nutrition DB (ใส่เงื่อนไขดักไว้เผื่อยังไม่มีไฟล์)
if os.path.exists("nutrition.json"):
    with open("nutrition.json", "r", encoding="utf-8") as f:
        nutrition_db = json.load(f)
else:
    nutrition_db = {}

# ==========================================
# 3. ฟังก์ชันเตรียมรูปให้ ViT (Preprocessing)
# ==========================================
def preprocess_for_vit(img: Image.Image):
    img = img.resize((512, 512)) 
    # ... โค้ดส่วนอื่นปล่อยไว้เหมือนเดิมครับ ...
    
    # แปลงเป็น float32 ให้เรียบร้อยตั้งแต่แรก
    img_data = np.array(img, dtype=np.float32) / 255.0 
    
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    img_data = (img_data - mean) / std
    img_data = np.transpose(img_data, (2, 0, 1))
    img_data = np.expand_dims(img_data, axis=0)
    
    # เช็คอีกทีให้ชัวร์ก่อนส่งออก
    return img_data.astype(np.float32)
# ==========================================
# 4. ตัวรัน AI Pipeline (YOLO -> ViT)
# ==========================================
def run_ai_pipeline(image_bytes: bytes) -> dict:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # 4.1 YOLO หาตำแหน่ง
    results = yolo_model(img)
    target_img = img
    if len(results[0].boxes) > 0:
        box = results[0].boxes[0].xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = map(int, box)
        target_img = img.crop((x1, y1, x2, y2))

    # 4.2 ViT จำแนกเมนู
    input_data = preprocess_for_vit(target_img)
    input_name = vit_session.get_inputs()[0].name
    logits = vit_session.run(None, {input_name: input_data})[0]
    
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / exp_logits.sum()
    class_idx = int(np.argmax(probs))
    confidence = float(probs[0, class_idx])
    
    try:
        class_name = classes_list[str(class_idx)]
    except (IndexError, KeyError):
        class_name = "Unknown"

    # ดึงข้อมูลจากฐานข้อมูล (ถ้าไม่เจอให้เป็น dict ว่างๆ)
    raw_nut = nutrition_db.get(class_name, {})
    
    # ดึงค่าทีละตัว ถ้าตัวไหนไม่มีในไฟล์ json ให้ใส่ 0 แทน
    nut_data = {
        "calories": raw_nut.get("calories", 0),
        "protein": raw_nut.get("protein", 0.0),
        "carbs": raw_nut.get("carbs", 0.0),
        "fat": raw_nut.get("fat", 0.0)
    }

    return {
        "menu_name": class_name,
        "confidence": confidence,
        "nutrition": nut_data
    }

# ==========================================
# 5. ตั้งค่า FastAPI & ProcessPoolExecutor
# ==========================================
app = FastAPI(title="Food Classification API", description="YOLO11 + ViT Pipeline")
executor = ProcessPoolExecutor(max_workers=2)
MAX_FILE_SIZE = 5 * 1024 * 1024

# ==========================================
# 6. API Endpoint
# ==========================================
@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="File too large. Maximum size is 5MB.")

    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid file type. Only JPEG and PNG are supported.")

    try:
        img = Image.open(io.BytesIO(contents))
        img.verify()
    except UnidentifiedImageError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Corrupted image file. Cannot read the image.")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid image format: {str(e)}")

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(executor, run_ai_pipeline, contents)

    return PredictResponse(
        status="success",
        menu_name=result["menu_name"],
        confidence=result["confidence"],
        nutrition=result["nutrition"]
    )
if __name__ == "__main__":
    import uvicorn
    # บังคับรันที่พอร์ต 7860
    uvicorn.run(app, host="0.0.0.0", port=7860)