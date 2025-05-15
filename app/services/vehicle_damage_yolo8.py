import torch
from ultralytics import YOLO
import os
from app.core.logging import logger


class YOLOVehicleDamageDetector:
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = torch.device(device)
        self.model_path = "app/models/vehicle_damage_best.pt"  # Path to your YOLO model
        self.model = self._load_model()

    def _load_model(self) -> YOLO:
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model file not found at: {self.model_path}")

        abs_model_path = os.path.abspath(self.model_path)
        logger.info(
            f"INFO: Attempting to load vehicle_damage_best model from {abs_model_path}")

        try:
            model_instance = YOLO(self.model_path)
            model_instance.to(self.device)
            logger.info(
                f"INFO: vehicle_damage_best model successfully loaded from {abs_model_path}")
        except Exception as e:
            logger.error(f"Error loading the YOLO model: {e}")
            raise

        return model_instance

    def predict(self, image):
        # Always use YOLO's `predict` method for consistency
        return self.model.predict(image)

    @property
    def names(self):
        return self.model.names
