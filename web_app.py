import streamlit as st
from PIL import Image
import requests

# ==========================================
# 1. ตั้งค่าหน้าเว็บ & API
# ==========================================
st.set_page_config(page_title="AI Food Tracker", page_icon="🍔", layout="centered")

# ชี้ไปที่ FastAPI ที่รันอยู่เบื้องหลังใน Docker เดียวกัน
API_URL = "http://localhost:8000" 

st.title("🍔 AI สแกนอาหาร & คำนวณแคลอรี่")
st.write("อัปโหลดรูปอาหารของคุณ เพื่อให้ AI ช่วยบอกว่าคือเมนูอะไร และมีกี่แคลอรี่!")

# ==========================================
# 2. เช็คสถานะ API เบื้องหลัง
# ==========================================
@st.cache_resource(ttl=60)
def check_api():
    try:
        # เช็คไปที่หน้าแรกของ FastAPI ว่าตอบกลับมาไหม
        res = requests.get(f"{API_URL}/", timeout=5)
        return res.status_code == 200
    except:
        return False

if not check_api():
    st.warning("⏳ ระบบ AI หลังบ้านกำลังสตาร์ท... โปรดรอสักครู่แล้วรีเฟรชหน้าเว็บ")

# ==========================================
# 3. อัปโหลดรูปและแสดงผล
# ==========================================
uploaded_file = st.file_uploader("เลือกรูปภาพอาหาร...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='รูปภาพที่คุณอัปโหลด', use_container_width=True)
    st.write("🔍 AI กำลังวิเคราะห์...")

    try:
        # ส่งไฟล์ไปที่ API
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "image/jpeg")}
        response = requests.post(f"{API_URL}/predict", files=files, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            
            # 🌟 จัดการตัวเลขความมั่นใจ (แก้ปัญหา 1.0%)
            raw_confidence = result.get("confidence", 0)
            confidence = raw_confidence * 100 if raw_confidence <= 1.0 else raw_confidence
            
            # ดึงชื่อเมนู (รองรับทั้ง key: menu_name และ class_name)
            class_name = result.get("menu_name", result.get("class_name", "Unknown"))
            nutrition = result.get("nutrition", {})
            
            # 🎯 แสดงผลลัพธ์
            st.markdown("---")
            st.subheader(f"🎯 AI ทายว่านี่คือ: **{class_name}** (มั่นใจ {confidence:.1f}%)")
            
            if confidence < 50:
                st.warning("⚠️ ระวัง! ความมั่นใจต่ำกว่า 50% อาจไม่ถูกต้องครับ ลองอัปโหลดรูปที่ชัดเจนกว่านี้")
            
            if nutrition:
                st.success(f"✅ เมนู: {class_name}")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("🔥 พลังงาน", f"{nutrition.get('calories', 0)} kcal")
                col2.metric("🥩 โปรตีน", f"{nutrition.get('protein', 0):.1f} g")
                col3.metric("🍚 คาร์โบไฮเดรต", f"{nutrition.get('carbs', 0):.1f} g")
                col4.metric("🥑 ไขมัน", f"{nutrition.get('fat', 0):.1f} g")
            else:
                st.info("💡 รู้จักเมนูนี้ แต่ยังไม่มีข้อมูลโภชนาการในระบบครับ")
                
        else:
            st.error(f"❌ API Error ({response.status_code}): ไม่สามารถวิเคราะห์ได้")

    except Exception as e:
        st.error(f"❌ ไม่สามารถติดต่อระบบ AI ได้: {str(e)}")