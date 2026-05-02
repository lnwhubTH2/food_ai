import requests
from PIL import Image
from transformers import pipeline
from ultralytics import YOLO

print("⏳ กำลังโหลดระบบ AI Pipeline...")

# 1. โหลดโมเดล YOLO11 (Object Detection)
# เราใช้ yolo11n.pt (Nano) เพื่อความรวดเร็วในการทำ PoC ครั้งแรก (ระบบจะโหลดไฟล์มาให้อัตโนมัติ)
print("-> โหลดโมเดล YOLO11...")
yolo_model = YOLO("yolo11n.pt") 

# 2. โหลดโมเดล ViT (Image Classification)
# ใช้ตัวที่ Fine-tuned สำหรับ Food-101 มาแล้วตามสถาปัตยกรรมของคุณ
print("-> โหลดโมเดล Vision Transformer...")
vit_classifier = pipeline("image-classification", model="chriamue/vit-base-patch16-224-food-101")

def analyze_full_image(image_path):
    print(f"\n🔍 เริ่มกระบวนการวิเคราะห์ภาพ: {image_path}")
    
    # เปิดรูปภาพต้นฉบับด้วย PIL
    original_image = Image.open(image_path).convert("RGB")
    
    # --- STEP 1: ตรวจจับวัตถุด้วย YOLO11 ---
    # ให้ YOLO หาว่ามี "อะไร" อยู่ในภาพบ้าง และอยู่ที่พิกัดไหน
    yolo_results = yolo_model(original_image)
    
    # ดึงข้อมูลกรอบสี่เหลี่ยม (Bounding Boxes) ทั้งหมดที่เจอ
    boxes = yolo_results[0].boxes
    
    if len(boxes) == 0:
        print("⚠️ YOLO ไม่พบวัตถุที่เป็นอาหารหรือจานในภาพนี้ครับ")
        return

    print(f"✅ YOLO ตรวจพบวัตถุทั้งหมด {len(boxes)} จุดในภาพเดียว (Multi-object Analysis)")
    print("-" * 40)

    # วนลูปจัดการทีละกรอบ (ทีละจาน)
    for index, box in enumerate(boxes):
        # ดึงพิกัด x1, y1, x2, y2 ของกรอบนั้นๆ
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        
        # --- STEP 2: ตัดภาพ (Crop) ---
        # ตัดภาพเฉพาะส่วนที่ YOLO ตีกรอบไว้ เพื่อส่งให้ ViT ดูชัดๆ
        cropped_food = original_image.crop((x1, y1, x2, y2))
        
        # --- STEP 3: ทายชื่ออาหารด้วย ViT ---
        vit_results = vit_classifier(cropped_food)
        predicted_name = vit_results[0]['label']
        confidence = vit_results[0]['score']
        
        print(f"🎯 วัตถุชิ้นที่ {index + 1}:")
        print(f"   - ViT ทายว่า: {predicted_name} (มั่นใจ {confidence:.2%})")
        
        # --- STEP 4: ดึงข้อมูลโภชนาการจาก FastAPI ---
        try:
            api_url = f"http://127.0.0.1:8000/food/{predicted_name}"
            response = requests.get(api_url)
            
            if response.status_code == 200:
                data = response.json()
                print(f"   - 📊 โภชนาการ: {data['calories']} kcal | โปรตีน {data['protein']}g | ไขมัน {data['fat']}g")
                print(f"   - 💡 ชื่อในระบบ: {data['dish_name']}")
            else:
                print(f"   - ⚠️ ไม่มีข้อมูลโภชนาการของ '{predicted_name}' ในฐานข้อมูล")
        except Exception as e:
            print(f"   - ❌ ไม่สามารถติดต่อ API ได้ (เปิด uvicorn ไว้หรือเปล่าครับ?)")
        
        print("-" * 40)

if __name__ == "__main__":
    # ระบุพาทของรูปภาพที่มีอาหารหลายๆ อย่างในภาพเดียว เพื่อทดสอบความสตรองของระบบ
    analyze_full_image("images/test_food_1.jpg")