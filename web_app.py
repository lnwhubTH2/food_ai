import streamlit as st
from ultralytics import YOLO
from PIL import Image
import json

# ==========================================
# 1. โหลดโมเดลสมอง AI (ONNX)
# ==========================================
@st.cache_resource
def load_model():
    return YOLO('best.onnx', task='classify')

model = load_model()

# ==========================================
# 2. โหลดฐานข้อมูลโภชนาการ (จากไฟล์ JSON)
# ==========================================
@st.cache_data
def load_nutrition_data():
    try:
        with open('nutrition.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return {} # ถ้าหาไฟล์ไม่เจอ ให้คืนค่าว่างเปล่าไปก่อน

nutrition_db = load_nutrition_data()

# ==========================================
# 3. ส่วนหน้าเว็บ (Streamlit UI)
# ==========================================
st.set_page_config(page_title="AI Food Tracker", page_icon="🍔")

st.title("🍔 AI สแกนอาหาร & คำนวณแคลอรี่")
st.write("อัปโหลดรูปอาหารของคุณ เพื่อให้ AI ช่วยบอกว่าคือเมนูอะไร และมีกี่แคลอรี่!")

uploaded_file = st.file_uploader("เลือกรูปภาพอาหาร...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='รูปภาพที่คุณอัปโหลด', use_container_width=True)
    st.write("🔍 AI กำลังวิเคราะห์...")

    # ส่งรูปให้ YOLO (ONNX) ทำนาย
    results = model(image)
    
    top_result = results[0]
    class_index = top_result.probs.top1
    class_name = top_result.names[class_index]
    confidence = top_result.probs.top1conf.item() * 100

    # ==========================================
    # 4. ส่วนแสดงผลลัพธ์
    # ==========================================
    st.markdown("---")
    st.subheader(f"🎯 AI ทายว่านี่คือ: **{class_name}** (มั่นใจ {confidence:.1f}%)")

    # ค้นหาข้อมูลใน JSON
    if class_name in nutrition_db:
        info = nutrition_db[class_name]
        st.success(f"🇹🇭 ชื่อไทย/ชื่อเรียก: {info['name_th']}")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🔥 พลังงาน", f"{info['cal']} kcal")
        col2.metric("🥩 โปรตีน", f"{info['protein']} g")
        col3.metric("🍚 คาร์โบไฮเดรต", f"{info['carbs']} g")
        col4.metric("🥑 ไขมัน", f"{info['fat']} g")
    else:
        st.warning("⚠️ รู้จักเมนูนี้ แต่ยังไม่มีข้อมูลโภชนาการในระบบครับ")