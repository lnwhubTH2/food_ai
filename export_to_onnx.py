import torch
from transformers import ViTForImageClassification

print("⏳ กำลังดาวน์โหลดโมเดล ViT จาก Hugging Face...")
# 1. ระบุชื่อโมเดล (ถ้าคุณมีโมเดลที่เทรนเองบน Hugging Face ให้เปลี่ยนชื่อตรงนี้ครับ)
model_name = "google/vit-base-patch16-224" 
model = ViTForImageClassification.from_pretrained(model_name)
model.eval()

# 2. สร้างรูปภาพจำลอง (Dummy Input) เพื่อให้ระบบรู้ว่า Input หน้าตาเป็นยังไง
# รูปแบบ: (Batch Size=1, Channels=3, Height=224, Width=224)
dummy_input = torch.randn(1, 3, 224, 224)

# 3. แปลงร่างเป็น ONNX
onnx_file_path = "vit_model.onnx"
print(f"⚙️ กำลังแปลงโมเดลเป็นไฟล์ {onnx_file_path} ...")

torch.onnx.export(
    model,                      # โมเดลต้นฉบับ
    dummy_input,                # ข้อมูลจำลอง
    onnx_file_path,             # ชื่อไฟล์ที่จะเซฟ
    export_params=True,         # เก็บ Weights ไว้ข้างในไฟล์เลย
    opset_version=14,           # เวอร์ชันของ ONNX (แนะนำ 14 ขึ้นไป)
    do_constant_folding=True,   # ปรับจูนให้ทำงานเร็วขึ้น
    input_names=['pixel_values'], # ตั้งชื่อท่อรับข้อมูล
    output_names=['logits'],      # ตั้งชื่อท่อส่งข้อมูล
    dynamic_axes={              # ทำให้รองรับการส่งรูปเข้ามาหลายรูปพร้อมกันได้
        'pixel_values': {0: 'batch_size'}, 
        'logits': {0: 'batch_size'}
    }
)

print(f"✅ แปลงไฟล์สำเร็จ! คุณได้ไฟล์ {onnx_file_path} แล้วครับ 🎉")