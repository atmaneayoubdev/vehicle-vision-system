# 🚗 Vehicle Vision System

A production-ready **Computer Vision** system built with **YOLOv8** and **FastAPI**, offering real-time APIs for:

- 🔍 **License Plate Detection**
- 💥 **Vehicle Damage Detection**
- 🚘 **Vehicle Detection**

This project showcases end-to-end deep learning integration — from **training custom models** to **deploying scalable APIs**, making it a powerful example of applying computer vision to real-world automotive use cases.

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-FB9F3C?logo=yolo&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Model%20Inference-EE4C2C?logo=pytorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Image%20Processing-5C3EE8?logo=opencv&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Data%20Handling-013243?logo=numpy&logoColor=white)
![Uvicorn](https://img.shields.io/badge/Uvicorn-ASGI%20Server-111111?logo=uvicorn&logoColor=white)
![Pydantic](https://img.shields.io/badge/Pydantic-Validation-6A1B9A?logo=pydantic&logoColor=white)
![Postman](https://img.shields.io/badge/Postman-Tested%20APIs-FF6C37?logo=postman&logoColor=white)
![Conda](https://img.shields.io/badge/Conda-Environment-44A833?logo=anaconda&logoColor=white)

---

> 💡 **Note:** Pretrained model weights and datasets are **not included** in this repo for intellectual property reasons. The README includes instructions on training and integrating your own models.


## 🚘 Project Overview

**Vehicle Vision System** is a complete computer vision pipeline built to detect, identify, and analyze vehicles in real-world images. This system includes three production-ready APIs powered by custom-trained YOLOv8 models:

- 📛 **License Plate Detection API** – Locates and extracts license plates from vehicle images.
- 🚗 **Vehicle Type Detection API** – Identifies and classifies vehicles (e.g., car, truck, bus).
- 💥 **Vehicle Damage Detection API** – Detects external damage areas like broken bumpers, hoods, or headlights.

Designed for real-time applications like insurance claim processing, parking automation, and vehicle registration systems, this project integrates modern deep learning models with a FastAPI backend for efficient, scalable deployment.

---

### 🔍 Key Features

- ✅ **Three separate endpoints** with modular model handling
- 🧠 **Custom-trained YOLOv8 models** for high-accuracy object detection
- 🖼️ **Base64 image support** for simple frontend integration
- 📦 **Clean codebase structure** ready for production or research extension
- 📊 **Extensible logging system** for tracking inference and failures
- 🧪 **Built-in test scripts** for quick local validation

This project simulates a real-world AI system with model training, optimization, and deployment — ideal for production teams or research applications.

