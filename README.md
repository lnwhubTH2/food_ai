
---
title: Food AI Model
emoji: 🍔
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# 🍕 Food Vision API: High-Throughput Image Classification

[![CI/CD Pipeline](https://github.com/joyjoysad/food_ai/actions/workflows/deploy.yml/badge.svg)](https://github.com/joyjoysad/food_ai/actions)
[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/joyjoysad/food_ai)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX-Runtime-blue?logo=onnx)](https://onnxruntime.ai/)

High-throughput Image Classification Service for Thai and International food. This API uses a highly optimized dual-model pipeline (**YOLOv11 for detection + INT8-quantized ViT for classification**) to deliver lightning-fast inference with detailed nutritional information.

This project implements the complete MLOps lifecycle according to the **MLOps Challenge** assignment.

---

## 🌟 The MLOps Lifecycle Implemented

1. **Model Optimization:** Converted models to ONNX format and applied Dynamic Quantization to achieve a massive reduction in model size and inference latency without sacrificing accuracy.
2. **API Development (Production-Ready):** Built with **FastAPI** using `async` architecture and `ProcessPoolExecutor` to handle CPU-bound inference, preventing API freezing under heavy load.
3. **Robust Error Handling:** Comprehensive Pydantic-driven validation, handling everything from corrupt images to unsupported MIME types with appropriate HTTP status codes.
4. **Automation & CI/CD:** Fully automated pipeline using **GitHub Actions**. Code pushes trigger `pytest` unit testing, and upon 100% success, automatically build and deploy the Docker container to **Hugging Face Spaces**.

---

## 📁 Repository Structure

```text
food_ai/
├── main.py                 # FastAPI application (Async routing + Inference workers)
├── test_main.py            # Pytest suite for unit testing
├── Dockerfile              # Containerization for Cloud deployment
├── requirements.txt        # Production dependencies
├── classes.json            # Model class labels mapping
├── nutrition.json          # Nutrition database (Calories, Macros)
├── models/                 # Optimized ONNX and YOLO models
│   ├── best.onnx           # INT8-Quantized ViT Model
│   └── yolo11n.pt          # YOLO Model for cropping
├── .github/workflows/
│   └── deploy.yml          # GitHub Actions CI/CD Pipeline
├── jmeter/
│   ├── load_test.jmx       # JMeter load testing script
│   └── dashboard/          # HTML Report from JMeter
└── README.md
```

---

## 🚀 Quickstart

### 1. Run Locally (Development)
```bash
# Clone the repository
git clone https://github.com/joyjoysad/food_ai.git
cd food_ai

# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn main:app --host 0.0.0.0 --port 8000
```
Browse to `http://localhost:8000/docs` to view the interactive Swagger UI.

### 2. Run via Docker (Production)
```bash
docker build -t food_ai_api .
docker run -p 8000:8000 food_ai_api
```

---

## 📡 API Reference & Usage

### Endpoints Overview

| Method | Path | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Service health check and welcome message |
| `POST` | `/predict` | Upload an image for food classification & nutrition data |
| `GET` | `/docs` | OpenAPI (Swagger) interactive documentation |

### Example cURL (Hugging Face Cloud)
```bash
curl -X 'POST' \
  'https://joyjoysad-food-ai.hf.space/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@sample_burger.jpg;type=image/jpeg'
```

### 🎯 Sample JSON Response
Unlike standard classifiers, our API provides **nutritional context**:
```json
{
  "status": "success",
  "menu_name": "hamburger",
  "confidence": 0.9854,
  "inference_time_ms": 65.2,
  "nutrition": {
    "calories": 295,
    "protein": 17.0,
    "carbs": 24.0,
    "fat": 14.0
  }
}
```

---

## 🛡️ Production Error Handling Matrix

Our API is designed to handle bad requests gracefully without crashing the server:

| Scenario | HTTP Status | Response Example |
| :--- | :--- | :--- |
| **Unsupported File Type** (e.g., .txt, .pdf) | `400 Bad Request` | `{"detail": "File must be an image (JPEG, PNG, WEBP)"}` |
| **Corrupted Image Data** | `400 Bad Request` | `{"detail": "Invalid image file format"}` |
| **Pydantic Validation Fail** (Missing fields) | `422 Unprocessable Entity` | Standard FastAPI Validation Error |
| **Worker Overload / Crash** | `500 Internal Server Error` | `{"detail": "Prediction failed: <Error>"}` |

---

## ⚙️ CI/CD & Automation

This repository uses **GitHub Actions** (`.github/workflows/deploy.yml`) to ensure code quality and continuous delivery:
1. **Test Phase:** Runs `pytest test_main.py` to verify API endpoints and mock predictions.
2. **Deploy Phase:** If tests pass `100%`, the pipeline connects to Hugging Face Spaces via `HF_TOKEN` and forces a Git push, triggering a new Docker build automatically.

---

## 📊 Performance Testing

The system has been load-tested using **Apache JMeter** to identify bottlenecks and ensure high concurrency handling. 
- **Tool used:** `jmeter/load_test.jmx`
- **Target metrics:** Throughput (TPS), Latency (P95), and Error Rate under concurrent user load.
- *Full analysis and HTML Dashboard results are available in the Project Report PDF.*

---
**Authors:** นาย ชินวัตร ทองสุวรรณ 1650900275  
             นาย กิตติภพ สายโย    1650902875 

