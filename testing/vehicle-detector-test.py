import requests
import base64
import os
import cv2
import numpy as np

# === CONFIG ===
API_URL = "http://127.0.0.1:8000/api/v1/vehicle-detector"
image_path = os.path.join("testing", "images", "vehicle_002.jpg")

# === ENCODE IMAGE ===
with open(image_path, "rb") as img_file:
    b64_image = base64.b64encode(img_file.read()).decode("utf-8")

payload = {
    "image": b64_image,
    "origin": "test-script"
}

# === SEND REQUEST ===
response = requests.post(API_URL, json=payload)

# === PARSE AND DRAW ===
if response.status_code == 200:
    print("✅ Detection successful:")
    detections = response.json()["detections"]
    print(detections)

    # Load original image using OpenCV
    image = cv2.imread(image_path)
    height, width = image.shape[:2]

    # Dynamically adjust font scale and thickness
    font_scale = max(0.8, min(width, height) / 1000)
    thickness = max(2, int(min(width, height) / 500))

    for det in detections:
        if det["label"] != "Vehicle":
            continue  # Skip anything that's not a license plate

        confidence = det["confidence"]
        x1, y1, x2, y2 = map(int, det["box"])

        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), thickness)

        # Add label and confidence
        text = f"Vehicle: {confidence:.2f}"
        cv2.putText(image, text, (x1, max(30, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), thickness)

    # Show the image
    cv2.imshow("Vehicle Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:
    print(f"❌ Error {response.status_code}: {response.text}")
