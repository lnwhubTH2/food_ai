import io
import os
import json
import time
import asyncio
import numpy as np
import onnxruntime as ort
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from concurrent.futures import ProcessPoolExecutor
from ultralytics import YOLO

# ==========================================
# 1. Pydantic Models
# ==========================================
class NutritionData(BaseModel):
    calories: int
    protein: float
    carbs: float
    fat: float

class PredictResponse(BaseModel):
    status: str
    menu_name: str
    name_th: str
    confidence: float
    inference_time_ms: float
    nutrition: NutritionData

class HealthResponse(BaseModel):
    status: str
    message: str
    model_loaded: bool

# ==========================================
# 2. โหลดโมเดล (Global Level)
# ==========================================
yolo_model = YOLO("yolo11n.pt")
vit_session = ort.InferenceSession("best.onnx")

with open("classes.json", "r", encoding="utf-8") as f:
    classes_list = json.load(f)

if os.path.exists("nutrition.json"):
    with open("nutrition.json", "r", encoding="utf-8") as f:
        nutrition_db = json.load(f)
else:
    nutrition_db = {}

# ==========================================
# 3. Preprocessing
# ==========================================
def preprocess_for_vit(img: Image.Image):
    img = img.resize((224, 224))
    img_data = np.array(img, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_data = (img_data - mean) / std
    img_data = np.transpose(img_data, (2, 0, 1))
    img_data = np.expand_dims(img_data, axis=0)
    return img_data.astype(np.float32)

# ==========================================
# 4. AI Pipeline (YOLO -> ViT)
# ==========================================
def run_ai_pipeline(image_bytes: bytes) -> dict:
    start = time.time()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # YOLO crop
    results = yolo_model(img, verbose=False)
    target_img = img
    if len(results[0].boxes) > 0:
        box = results[0].boxes[0].xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = map(int, box)
        target_img = img.crop((x1, y1, x2, y2))

    # ViT classify
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

    # ✅ FIXED: รองรับทั้ง key "cal" และ "calories"
    raw_nut = nutrition_db.get(class_name, {})
    nut_data = {
        "calories": int(raw_nut.get("calories", raw_nut.get("cal", 0))),
        "protein": float(raw_nut.get("protein", 0.0)),
        "carbs": float(raw_nut.get("carbs", 0.0)),
        "fat": float(raw_nut.get("fat", 0.0))
    }
    name_th = raw_nut.get("name_th", class_name)

    elapsed_ms = (time.time() - start) * 1000

    return {
        "menu_name": class_name,
        "name_th": name_th,
        "confidence": confidence,
        "inference_time_ms": round(elapsed_ms, 2),
        "nutrition": nut_data
    }

# ==========================================
# 5. FastAPI Setup
# ==========================================
app = FastAPI(
    title="Food Classification API",
    description="High-Throughput YOLO11 + ViT (ONNX Quantized) Pipeline",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

executor = ProcessPoolExecutor(max_workers=2)
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/jpg", "image/webp"}

# ==========================================
# 6. Endpoints
# ==========================================
@app.get("/", response_model=HealthResponse)
async def root():
    return HealthResponse(
        status="ok",
        message="Food Classification API is running 🍔",
        model_loaded=os.path.exists("best.onnx")
    )

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="healthy",
        message="Service operational",
        model_loaded=os.path.exists("best.onnx")
    )

@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    # 1) Validate content type
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type '{file.content_type}'. Allowed: JPEG, PNG, WEBP"
        )

    # 2) Read & validate size
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File too large ({len(contents)/1024/1024:.2f} MB). Max 5 MB."
        )
    if len(contents) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty file uploaded."
        )

    # 3) Validate image format
    try:
        img = Image.open(io.BytesIO(contents))
        img.verify()
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Corrupted image file. Cannot decode."
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid image format: {str(e)}"
        )

    # 4) Run inference in process pool
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(executor, run_ai_pipeline, contents)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )

    return PredictResponse(status="success", **result)


if __name__ == "__main__":
    import uvicorn
    # Port 7860 = Hugging Face Spaces default
    uvicorn.run(app, host="0.0.0.0", port=7860)