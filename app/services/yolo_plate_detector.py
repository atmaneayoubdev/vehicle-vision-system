# app/services/yolo_plate_detector.py
import torch
from ultralytics import YOLO
import os
from app.core.logging import logger


class YOLOLicensePlateDetector:
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = torch.device(device)
        self.model_path = "app/models/best_licence_plate_detector.pt"  # Path to your YOLO model
        self.model = self._load_model()

    def _load_model(self) -> YOLO:
        """
        Load the YOLO model from the predefined path.
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model file not found at: {self.model_path}")

        abs_model_path = os.path.abspath(self.model_path)
        logger.info(
            f"INFO: Attempting to load YOLO model from {abs_model_path}")

        try:
            model_instance = YOLO(self.model_path)  # Load the model
            model_instance.to(self.device)
            logger.info(
                f"INFO: YOLO model successfully loaded from {abs_model_path}")
        except Exception as e:
            logger.error(f"Error loading the YOLO model: {e}")
            raise

        return model_instance

 # app/services/yolo_plate_detector.py

    def predict(self, img):
        """
        Run inference on the image and return the results.
        """
        if self.device.type == 'cuda':
            with torch.amp.autocast(device_type='cuda'):
                results = self.model(img)
        else:
            results = self.model(img)
        return results
