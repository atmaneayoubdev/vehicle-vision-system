# FILE: app/services/model_manager.py
import torch
from app.services.yolo_plate_detector import YOLOLicensePlateDetector
from app.services.vehicle_damage_yolo8 import YOLOVehicleDamageDetector


class ModelManager:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.vehicle_licence_plate_model = YOLOLicensePlateDetector(
            device=self.device)
        self.vehicle_damage_model = YOLOVehicleDamageDetector(
            device=self.device)

        # Initialize the Runway model
        # self.runway_model = RunwayModel(device=self.device)

    def get_vehicle_licence_model(self):
        return self.vehicle_licence_plate_model

    def get_vehicle_damage_model(self):
        return self.vehicle_damage_model

    # def get_runway_model(self):
    #     return self.runway_model


# Singleton instance of ModelManager
model_manager = ModelManager()
