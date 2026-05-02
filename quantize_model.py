import os
import time
from onnxruntime.quantization import quantize_dynamic, QuantType

# 1. กำหนดชื่อไฟล์ต้นฉบับ และไฟล์ใหม่ที่จะสร้าง
model_fp32 = 'vit_model.onnx'             # ไฟล์ ONNX ตัวเดิมของคุณ
model_quant = 'vit_model_quantized.onnx'  # ไฟล์ ONNX ตัวใหม่ที่จะถูกบีบอัด

print(f"⏳ กำลังบีบอัดโมเดล {model_fp32} ...")
start_time = time.time()

# 2. คำสั่งทำ Dynamic Quantization (ลดสเกลเป็น 8-bit)
quantize_dynamic(
    model_input=model_fp32,
    model_output=model_quant,
    weight_type=QuantType.QUInt8
)

end_time = time.time()
print(f"✅ Quantization สำเร็จ! ใช้เวลาไป {end_time - start_time:.2f} วินาที")

# 3. สรุปผลเปรียบเทียบขนาดไฟล์ (สำหรับเอาไปใส่ในรายงาน PDF)
size_fp32 = os.path.getsize(model_fp32) / (1024 * 1024)
size_quant = os.path.getsize(model_quant) / (1024 * 1024)

print("-" * 30)
print(f"📊 สรุปผลขนาดไฟล์ (Model Size):")
print(f"• ไฟล์ต้นฉบับ (FP32) : {size_fp32:.2f} MB")
print(f"• ไฟล์บีบอัด (INT8)   : {size_quant:.2f} MB")
print(f"• ขนาดลดลงไป          : {(1 - (size_quant/size_fp32)) * 100:.2f}%")
print("-" * 30)