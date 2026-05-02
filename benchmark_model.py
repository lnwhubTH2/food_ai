import torch
from transformers import ViTForImageClassification
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
import os
import time
import numpy as np

model_name = "google/vit-base-patch16-224"
model = ViTForImageClassification.from_pretrained(model_name)
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)
onnx_file_path = "vit_model.onnx"
quant_file_path = "vit_model_quantized.onnx"

# 1. Export ONNX
torch.onnx.export(
    model, dummy_input, onnx_file_path,
    export_params=True, opset_version=14,
    do_constant_folding=True,
    input_names=['pixel_values'], output_names=['logits'],
    dynamic_axes={'pixel_values': {0: 'batch_size'}, 'logits': {0: 'batch_size'}}
)

# 2. Quantize
quantize_dynamic(
    model_input=onnx_file_path,
    model_output=quant_file_path,
    weight_type=QuantType.QUInt8
)

# 3. Sizes
onnx_size = os.path.getsize(onnx_file_path) / (1024*1024)
quant_size = os.path.getsize(quant_file_path) / (1024*1024)

# 4. Latency
# PyTorch
with torch.no_grad():
    for _ in range(5): model(dummy_input) # Warmup
    start = time.time()
    for _ in range(50): model(dummy_input)
    pt_latency = (time.time() - start) / 50 * 1000 # ms

# ONNX FP32
ort_sess_fp32 = ort.InferenceSession(onnx_file_path)
dummy_input_np = dummy_input.numpy()
input_name = ort_sess_fp32.get_inputs()[0].name
for _ in range(5): ort_sess_fp32.run(None, {input_name: dummy_input_np}) # Warmup
start = time.time()
for _ in range(50): ort_sess_fp32.run(None, {input_name: dummy_input_np})
onnx_latency = (time.time() - start) / 50 * 1000 # ms

# ONNX Quantized
ort_sess_quant = ort.InferenceSession(quant_file_path)
for _ in range(5): ort_sess_quant.run(None, {input_name: dummy_input_np}) # Warmup
start = time.time()
for _ in range(50): ort_sess_quant.run(None, {input_name: dummy_input_np})
quant_latency = (time.time() - start) / 50 * 1000 # ms

print(f"Sizes: ONNX FP32={onnx_size:.2f}MB, ONNX Quantized={quant_size:.2f}MB")
print(f"Latency: PyTorch={pt_latency:.2f}ms, ONNX FP32={onnx_latency:.2f}ms, ONNX Quantized={quant_latency:.2f}ms")