import pandas as pd
import requests
import os
import time

# --- ตั้งค่า ---
csv_file = "MM-Food-100K_Data.csv"
output_dir = "food_dataset/images"
output_csv = "food_dataset/train_labels.csv"
num_images_to_download = 1000  # จำนวนรูปที่ต้องการโหลดทดสอบก่อน

# สร้างโฟลเดอร์หลักสำหรับเทรน ถ้ายังไม่มี
os.makedirs(output_dir, exist_ok=True)

# โหลดข้อมูลใบเฉลย
print("กำลังอ่านไฟล์ CSV...")
df = pd.read_csv(csv_file)
df_subset = df.head(num_images_to_download).copy()

new_records = []

print(f"🚀 กำลังเริ่มดาวน์โหลดภาพ {num_images_to_download} รูป (อาจใช้เวลาสักครู่)...")

for index, row in df_subset.iterrows():
    img_url = row['image_url']
    # ตั้งชื่อไฟล์ใหม่ให้รันตามเลข 4 หลัก เช่น img_0000.jpg
    file_name = f"img_{index:04d}.jpg"
    save_path = os.path.join(output_dir, file_name)
    
    try:
        # สั่งโหลดภาพ (รอสูงสุด 5 วินาทีต่อภาพ)
        response = requests.get(img_url, timeout=5)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
            
            # เก็บข้อมูลชื่อไฟล์ภาพที่อยู่บนเครื่องเรา คู่กับข้อมูลอาหารเดิม
            row_data = row.to_dict()
            row_data['local_image_path'] = file_name
            new_records.append(row_data)
            
            # ปรินต์อัปเดตทุกๆ 100 รูป
            if (index + 1) % 100 == 0:
                print(f"✅ โหลดสำเร็จแล้ว {index + 1} รูป...")
        else:
            print(f"⚠️ รูปที่ {index} โหลดไม่ได้ (ลิงก์อาจตาย)")
            
    except Exception as e:
        print(f"⚠️ รูปที่ {index} โหลดไม่ได้ (Error)")
        
    # พักเบรก 0.1 วินาที เพื่อไม่ให้เซิร์ฟเวอร์ปลายทางบล็อกเรา
    time.sleep(0.1)

# เซฟไฟล์ CSV อันใหม่ที่ล้างข้อมูลและจับคู่เรียบร้อยแล้ว
new_df = pd.DataFrame(new_records)
new_df.to_csv(output_csv, index=False)

print("\n" + "="*40)
print(f"🎉 ดาวน์โหลดเสร็จสิ้น! ได้ภาพที่สมบูรณ์ {len(new_df)} รูป")
print(f"📁 รูปเก็บไว้ที่โฟลเดอร์: {output_dir}")
print(f"📝 ใบเฉลยอันใหม่เก็บไว้ที่: {output_csv}")
print("="*40)