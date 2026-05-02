import splitfolders

# แก้ตรง r"ที่อยู่โฟลเดอร์จริงของคุณ" 
splitfolders.ratio(r"E:\food_ai\archive\images", 
                   output=r"E:\food_ai\dataset_ready", 
                   seed=1337, ratio=(0.8, 0.2))

print("✅ แบ่งไฟล์เสร็จเรียบร้อยแล้ว!")