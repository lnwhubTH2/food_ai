import requests
from transformers import pipeline
from PIL import Image

# 1. โหลดโมเดล ViT ของ Google (ตามที่คุณระบุ)
print("⏳ กำลังโหลด Google ViT Model...")
# เราใช้ pipeline 'image-classification' เพื่อให้มันทำหน้าที่จำแนกภาพ
classifier = pipeline("image-classification", model="google/vit-base-patch16-224")

def get_food_analysis(image_path):
    # --- ส่วนของ AI (The Vision) ---
    print(f"🔍 วิเคราะห์รูปภาพ: {image_path}...")
    img = Image.open(image_path).convert("RGB")
    results = classifier(img)
    
    # ดึงผลลัพธ์ที่มั่นใจที่สุด
    top_prediction = results[0]['label']
    confidence = results[0]['score']
    
    # เคล็ดลับ: ชื่อจาก Google ViT อาจจะเป็นชื่อกว้างๆ เช่น 'cup', 'plate' 
    # เราจะเอาชื่อนี้ไปค้นใน Database ของเรา
    print(f"🤖 AI มองเห็นเป็น: {top_prediction} ({confidence:.2%})")

    # --- ส่วนของ API (The Brain) ---
    print(f"📡 กำลังดึงข้อมูลจาก Database ผ่าน FastAPI...")
    try:
        # ส่งชื่อที่ AI ทายได้ ไปที่ API ของเรา
        api_url = f"http://127.0.0.1:8000/food/{top_prediction}"
        response = requests.get(api_url)
        
        if response.status_code == 200:
            data = response.json()
            print("\n" + "✨" + "="*30 + "✨")
            print(f"🍴 รายการอาหาร: {data['dish_name']}")
            print(f"🔥 พลังงาน: {data['calories']} kcal")
            print(f"🧪 สารอาหาร: P:{data['protein']}g | F:{data['fat']}g | C:{data['carbs']}g")
            print(f"📍 วิธีปรุง: {data['cooking_method']}")
            print("✨" + "="*30 + "✨")
        else:
            print(f"ℹ️ AI ทายว่า '{top_prediction}' แต่เมนูนี้ยังไม่มีใน Database ของเรา")
            
    except Exception as e:
        print(f"❌ เชื่อมต่อ API ไม่ได้: {e} (อย่าลืมเปิด uvicorn ในอีกหน้าต่างนะ!)")

if __name__ == "__main__":
    get_food_analysis("images/test_food_1.jpg")