---
title: Food AI Model
emoji: 🍔
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# 🍕 Food Vision API — High-Throughput Image Classification

[![CI/CD Pipeline](https://github.com/joyjoysad/food_ai/actions/workflows/deploy.yml/badge.svg)](https://github.com/joyjoysad/food_ai/actions)
[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/joyjoysad/food_ai)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX-Runtime-blue?logo=onnx)](https://onnxruntime.ai/)

High-throughput Image Classification Service for Thai and International food.
This API uses an optimized dual-model pipeline (**YOLOv11 detection + INT8-quantized ViT classifier**) delivering fast inference with detailed nutritional information.

> Implemented as part of the **MLOps Challenge** assignment (270 food categories).

---

## 🌟 MLOps Lifecycle Implemented

1. **Model Optimization** — PyTorch → ONNX → Dynamic Quantization (INT8). Size −75%, Latency −73% (vs PyTorch baseline).
2. **Production-Ready API** — FastAPI `async` + `ProcessPoolExecutor` for CPU-bound inference. Defense-in-depth validation (Pydantic + MIME + size + corrupted image check).
3. **CI/CD Automation** — GitHub Actions runs `pytest` on every push, then auto-deploys to Hugging Face Spaces on green tests.
4. **Performance Testing** — Apache JMeter test plans for both Local (Docker) and Cloud environments.

---

## 📁 Repository Structure

```text
food_ai/
├── main.py                      # FastAPI application
├── test_main.py                 # Pytest unit tests
├── classes.json                 # 270 class labels
├── nutrition.json               # Nutrition database
├── best.onnx                    # INT8-quantized ViT (Git LFS)
├── yolo11n.pt                   # YOLO model (Git LFS)
│
├── Dockerfile                   # Container image (port 7860)
├── docker-compose.yml           # Local orchestration
├── requirements.txt             # Python dependencies
│
├── .github/workflows/
│   └── deploy.yml               # CI/CD pipeline
│
├── jmeter/
│   ├── load_test.jmx            # JMeter test plan
│   └── README.md                # How to run load tests
│
├── postman/
│   └── Food_AI_API.postman_collection.json
│
├── Project_Report.pdf           # Full project report
└── README.md
```

---

## 🚀 Quickstart

### 1. Run Locally (Python)

```bash
git clone https://github.com/joyjoysad/food_ai.git
cd food_ai

pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 7860
```

Open `http://localhost:7860/docs` for interactive Swagger UI.

### 2. Run via Docker

```bash
docker-compose up --build
# → API on http://localhost:7860
```

### 3. Run Tests

```bash
pytest test_main.py -v
```

---

## 📡 API Reference

| Method | Path        | Description                          |
| :----- | :---------- | :----------------------------------- |
| `GET`  | `/`         | Service health & welcome             |
| `GET`  | `/health`   | Health probe (Docker / K8s)          |
| `POST` | `/predict`  | Classify a food image                |
| `GET`  | `/docs`     | Swagger UI (auto-generated)          |

### 🎯 cURL — Production (Hugging Face)

```bash
curl -X POST 'https://joyjoysad-food-ai.hf.space/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@sample_burger.jpg;type=image/jpeg'
```

### 🎯 cURL — Local

```bash
curl -X POST 'http://localhost:7860/predict' \
  -F 'file=@sample_burger.jpg;type=image/jpeg'
```

### Sample Response

```json
{
  "status": "success",
  "menu_name": "hamburger",
  "name_th": "hamburger",
  "confidence": 0.9854,
  "inference_time_ms": 65.2,
  "nutrition": {
    "calories": 450,
    "protein": 20.0,
    "carbs": 40.0,
    "fat": 25.0
  }
}
```

---

## 🛡️ Production Error Handling

| Scenario                          | HTTP Code | Response Detail                                           |
| :-------------------------------- | :-------- | :-------------------------------------------------------- |
| Unsupported file type (.txt/.pdf) | `400`     | `"Invalid file type ... Allowed: JPEG, PNG, WEBP"`        |
| File too large (> 5 MB)           | `400`     | `"File too large (X.XX MB). Max 5 MB."`                   |
| Empty file                        | `400`     | `"Empty file uploaded."`                                  |
| Corrupted image data              | `400`     | `"Corrupted image file. Cannot decode."`                  |
| Missing `file` field              | `422`     | Standard Pydantic validation error                        |
| Inference worker error            | `500`     | `"Prediction failed: <error>"`                            |

---

## ⚙️ CI/CD Pipeline

The workflow at `.github/workflows/deploy.yml`:

1. **Test phase** — checkout (with LFS) → setup Python 3.10 → install deps → run `pytest`.
2. **Deploy phase** — only on `main`, only after tests pass — pushes the repo to Hugging Face Spaces using `HF_TOKEN`.

### Required Secrets (GitHub → Settings → Secrets)
- `HF_TOKEN` — Hugging Face access token (write scope)
- `HF_USERNAME` — your HF username (e.g. `joyjoysad`)
- `HF_SPACE_NAME` — your Space name (e.g. `food_ai`)

---

## 📊 Performance Testing (JMeter)

Test plan at `jmeter/load_test.jmx` is parametric — override via `-J<param>=<value>`.

```bash
jmeter -n -t jmeter/load_test.jmx \
  -Jhost=localhost -Jport=7860 -Jthreads=50 \
  -Jrampup=30 -Jloops=10 -Jimage=sample_food.jpg \
  -l results.jtl -e -o jmeter/dashboard
```

See `jmeter/README.md` for full instructions and the **Project Report PDF** for analysis.

---

## 📦 Optimization Results

| Metric         | PyTorch FP32 | ONNX FP32 | ONNX INT8 (Quantized) |
| :------------- | :----------- | :-------- | :-------------------- |
| Model Size     | 346 MB       | 343 MB    | **86 MB** (−75%)      |
| Latency (CPU)  | 285 ms       | 155 ms    | **78 ms** (3.65× faster) |
| Accuracy Drop  | —            | 0%        | < 1%                  |

---

**Authors**
- ชินวัตร ทองสุวรรณ — 1650900275
- กิตติภพ สายโย — 1650902875

**Submission:** May 9, 2026