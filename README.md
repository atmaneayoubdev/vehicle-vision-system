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


## Examples / Demo

This section demonstrates sample inputs and the corresponding outputs for the three main APIs in the Vehicle Vision System: License Plate Detection, Vehicle Type Classification, and Vehicle Damage Detection.

---


### Vehicle Detection



**API Request (Python example):**

    import requests
    import base64

    with open("examples/vehicle_001.jpg", "rb") as img_file:
        b64_image = base64.b64encode(img_file.read()).decode()

    payload = {"image": b64_image, "origin": "demo"}

    response = requests.post("http://localhost:8000/api/v1/vehicle-detection", json=payload)
    print(response.json())

**Sample API Response:**

    {
      "detections": [
        {
          "label": "Vehicle",
          "confidence": 0.97,
          "box": [120, 80, 280, 140]
        }
      ]
    }

**Output Visualization:**

![License Plate Output](examples/vehicle_detection_example.png)

---


### License Plate Detection



**API Request (Python example):**

    import requests
    import base64

    with open("examples/license_plate_01.jpg", "rb") as img_file:
        b64_image = base64.b64encode(img_file.read()).decode()

    payload = {"image": b64_image, "origin": "demo"}

    response = requests.post("http://localhost:8000/api/v1/license-plate-detector", json=payload)
    print(response.json())

**Sample API Response:**

    {
      "detections": [
        {
          "label": "Vehicle Plate",
          "confidence": 0.97,
          "box": [120, 80, 280, 140]
        }
      ]
    }

**Output Visualization:**

![License Plate Input](examples/vehicle_licence_plate_detection_example.png)

---

### Vehicle Damage Detection


**API Request (Python example):**

    import requests
    import base64

    with open("examples/vehicle_damage_01.jpg", "rb") as img_file:
        b64_image = base64.b64encode(img_file.read()).decode()

    payload = {"image": b64_image, "origin": "demo"}

    response = requests.post("http://localhost:8000/api/v1/vehicle-damage-detector", json=payload)
    print(response.json())

**Sample API Response:**

    {
      "detections": [
        {
          "label": "damaged bumper",
          "confidence": 0.88,
          "box": [230, 340, 450, 470]
        },
        {
          "label": "damaged headlight",
          "confidence": 0.82,
          "box": [120, 200, 180, 260]
        }
      ]
    }

**Output Visualization:**
![Vehicle Damage Output](examples/vehicle_damage_detection_example.png)



## Training the Models

This project leverages custom-trained YOLOv8 models for three key computer vision tasks:

- License Plate Detection
- Vehicle Type Classification
- Vehicle Damage Detection

To achieve high accuracy and robustness, the models were trained on carefully curated datasets with precise annotations.

### Dataset Preparation & Labeling

For training, images need to be collected and annotated according to the task:

- **License Plate Detection:** Annotate bounding boxes around vehicle license plates.
- **Vehicle Type Classification:** Label images by vehicle type (e.g., Sedan, SUV, Truck).
- **Vehicle Damage Detection:** Annotate damaged vehicle parts with bounding boxes (e.g., damaged bumper, broken headlight).

**Recommended Tool:** [Roboflow](https://roboflow.com/) is an excellent platform for dataset management, annotation, and export in YOLO format. You can upload images, label them with bounding boxes, and export datasets ready for YOLOv8 training.

### Example: Dataset Structure (YOLO format)

```
/dataset
/images
/train
img001.jpg
img002.jpg
...
/val
img101.jpg
img102.jpg
...
/labels
/train
img001.txt
img002.txt
...
/val
img101.txt
img102.txt
...
```
**Pro tip:**  
Use annotation tools such as [LabelImg](https://github.com/tzutalin/labelImg) or [Roboflow](https://roboflow.com/) to speed up the labeling process and export directly in YOLO format.

If you want to see a full training pipeline example, check out the [Ultralytics YOLOv8 docs](https://docs.ultralytics.com/).


Each `.txt` label file corresponds to an image and contains object annotations formatted as:  
`<class_id> <x_center> <y_center> <width> <height>`,  
where coordinates are normalized (values between 0 and 1) relative to the image dimensions.

### Training Workflow

The models were trained using the [Ultralytics YOLOv8](https://docs.ultralytics.com/) framework, which provides an efficient and flexible pipeline.

Basic steps include:

1. **Label your data:** Annotate images using the tools mentioned above.
2. **Organize dataset:** Structure images and labels following the example above, ensuring a split between training and validation sets.
3. **Train the model:** Use commands like the following to start training:

```bash
yolo task=detect mode=train model=yolov8n.pt data=dataset.yaml epochs=50 imgsz=640
```

---

## Setup & Running Inference

Once you've trained your models and saved them into the `app/models/` directory, you can easily run the system locally or in a production-like containerized environment using Docker.

### 📁 Expected Directory Structure

```bash
vehicle-vision-system/
├── app/
│ ├── main.py
│ ├── api/
│ ├── models/
│ │ ├── license_plate_best.pt
│ │ ├── vehicle_type_best.pt
│ │ └── vehicle_damage_best.pt
│ └── services/
├── examples/
├── Dockerfile
├── requirements.txt
└── README.md
```


Ensure the trained YOLOv8 models are correctly named and placed inside `app/models/`.

---

### 🔧 Local Development (FastAPI + Uvicorn)

You can test the APIs locally using Uvicorn.

**Install dependencies (in a virtualenv or Conda environment):**

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Navigate to the interactive API docs at:
📍 http://localhost:8000/docs


### 🐳 Run with Docker (Recommended)
A Dockerized setup ensures consistency across environments.

1. Build the Docker image
```bash
docker build -t vehicle-vision-api .
```
2. Run the container
```bash
docker run -p 8000:8000 vehicle-vision-api
```

This will expose the API at http://localhost:8000.

To run in the background:

```bash
docker run -d -p 8000:8000 vehicle-vision-api
```



### 🧪 Testing the APIs
After the server is running, test it by sending a base64 image payload to any of the 3 endpoints:

/api/v1/license-plate-detector

/api/v1/vehicle-type-classifier

/api/v1/vehicle-damage-detector

### Example test request (Python):

```python

import requests
import base64

with open("examples/license_plate_01.jpg", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

res = requests.post("http://localhost:8000/api/v1/license-plate-detector", json={"image": b64, "origin": "test"})
print(res.json())
```

### ✅ Endpoints Recap

| Endpoint                            | Method | Description                     |
|------------------------------------|--------|---------------------------------|
| `/api/v1/license-plate-detector`   | POST   | Detects vehicle license plates  |
| `/api/v1/vehicle-type-classifier`  | POST   | Classifies vehicle type         |
| `/api/v1/vehicle-damage-detector`  | POST   | Detects damaged vehicle regions |

> 🧠 **Pro Tip:** If you're running in production, consider mounting volumes for model files and serving behind a reverse proxy like Nginx with HTTPS.


## 🚀 Deployment

To deploy the Vehicle Vision System in a production-ready environment, we recommend using **Docker** along with **Docker Compose**. This ensures reproducibility, easy scaling, and compatibility across different systems.

### 🔧 Folder Structure

Ensure your project has the following structure:

```
vehicle-vision-system/
│
├── app/
│ ├── main.py
│ ├── api/
│ ├── models/ # Pretrained YOLOv8 .pt model files here
│ ├── services/
│ └── core/
│
├── examples/ # Example images for testing
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

### 🐳 Dockerfile

This Dockerfile sets up the FastAPI app with all required dependencies and runs the server using Uvicorn.

```Dockerfile
# Use official lightweight Python image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire app
COPY . .

# Expose the FastAPI port
EXPOSE 8000

# Run the FastAPI app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]


### ⚙️ Step 3: docker-compose.yml

Use `docker-compose` to simplify running the API service, especially useful in a development or deployment environment. This configuration ensures the container builds properly, mounts the local model files, and restarts automatically if needed.

```yaml
version: "3.9"

services:
  vehicle-vision-api:
    build: .
    container_name: vehicle_vision_api
    ports:
      - "8000:8000"
    volumes:
      - ./app/models:/app/app/models  # Mount model files
      - ./examples:/app/examples      # Mount example images (optional)
    restart: always


## 🚀 Usage

Once your container is running, you can interact with the APIs for vehicle license plate detection, type classification, and damage detection.

### 🔧 How to Use

Send a POST request with a base64-encoded image in the following JSON format:

```json
{
  "image": "<base64_encoded_image>",
  "origin": "demo"
}
```

### 🧪 Python Example
```python
import requests
import base64

# Replace with your test image path
with open("examples/vehicle_type_01.jpg", "rb") as img_file:
    b64_image = base64.b64encode(img_file.read()).decode()

payload = {"image": b64_image, "origin": "demo"}
response = requests.post("http://localhost:8000/api/v1/vehicle-type-classifier", json=payload)

print(response.json())
```

### 🧪 curl Example
```bash
curl -X POST http://localhost:8000/api/v1/vehicle-type-classifier \
  -H "Content-Type: application/json" \
  -d '{"image": "<base64_encoded_image>", "origin": "demo"}'  
```

> 💡 **Pro Tip:** For production environments:
> - Mount model directories as volumes instead of copying them into the container.
> - Use a reverse proxy like **Nginx** to serve the FastAPI app behind **HTTPS**.
> - Enable request logging and monitoring (e.g., with **Prometheus**, **Grafana**, or **Sentry**) for observability.
> - Scale using container orchestration tools like **Docker Compose**, **Kubernetes**, or **AWS ECS**.
