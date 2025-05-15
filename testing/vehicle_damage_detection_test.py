import requests
import base64
import os
import cv2
import numpy as np

# === CONFIG ===
# Endpoint for vehicle damage
API_URL = "http://127.0.0.1:8000/api/v1/vehicle-damage-detector"
# Change the image path if needed
image_path = os.path.join("testing", "images", "damaged_vehicle_002.jpg")

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
    font_scale = max(0.6, min(width, height) / 1000)
    thickness = max(2, int(min(width, height) / 300))

    for det in detections:
        # We can filter for specific damage classes like "damaged door", "damaged bumper" etc.
        if "damaged" not in det["label"].lower():
            continue  # Skip anything that's not a vehicle damage part

        confidence = det["confidence"]
        x1, y1, x2, y2 = map(int, det["box"])

        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), thickness)

        # Add label and confidence
        text = f"{det['label']}: {confidence:.2f}"
        cv2.putText(image, text, (x1, max(30, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), thickness)

    # Show the image
    cv2.imshow("Vehicle Damage Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:
    print(f"❌ Error {response.status_code}: {response.text}")
