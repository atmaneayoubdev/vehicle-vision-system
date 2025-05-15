# app/api/v1/license_plate_detector/utils.py
import numpy as np


def process_image_with_model(image: np.ndarray, model) -> dict:
    """
    Process the image with the given model and return the detections.
    Args:
    image (np.ndarray): The input image as a NumPy array.
    model: The YOLO model used for inference.
    Returns:
    dict: A dictionary containing the detections with labels, confidences, and bounding boxes.
    """

    # Define label mapping
    label_mapping = {0: "Vehicle Plate", 1: "Vehicle"}

    # Run inference
    results = model.predict(image)

    # Extract bounding box coordinates, confidences, and labels
    boxes = results[0].boxes.xyxy
    confs = results[0].boxes.conf
    labels = results[0].boxes.cls

    # Move the results to CPU if they are on GPU
    if boxes.is_cuda:
        boxes = boxes.cpu()
    if confs.is_cuda:
        confs = confs.cpu()
    if labels.is_cuda:
        labels = labels.cpu()

    # Convert to numpy arrays
    boxes = boxes.numpy()
    confs = confs.numpy()
    labels = labels.numpy()

    # Format bounding boxes into a JSON-compatible format
    detections = [
        {
            "label": label_mapping.get(int(labels[i]), "Unknown"),
            "confidence": float(confs[i]),
            "box": boxes[i].tolist()  # Convert to list for JSON serialization
        }
        for i in range(len(boxes))
    ]

    return {"detections": detections}
