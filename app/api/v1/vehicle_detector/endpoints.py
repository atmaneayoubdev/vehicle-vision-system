# app/api/v1/vehicle_damage_detection/endpoints.py

from fastapi import APIRouter, HTTPException, Request
from app.api.v1.license_plate_detector.schemas import LicensePlateDetectorRequest
from app.services.model_manager import model_manager
from app.utils.helper import decode_base64_to_image
from app.api.v1.license_plate_detector.utils import process_image_with_model
from app.core.logging import logger

router = APIRouter()

yolo_model = model_manager.get_vehicle_licence_model()


@router.post("/vehicle-detector")
async def vehicle_detector(req: LicensePlateDetectorRequest, request: Request):
    try:
        if "," in req.image:
            req.image = req.image.split(",")[1]

        image, _, _ = decode_base64_to_image(req.image)
        detections = process_image_with_model(image, yolo_model)

        # Filter only for vehicles
        vehicle_detections = [
            d for d in detections["detections"] if d["label"] == "Vehicle"]

        if not vehicle_detections:
            raise HTTPException(status_code=404, detail="Vehicle not found")

        return {"detections": vehicle_detections}

    except HTTPException as http_exception:
        logger.error(f"HTTPException: {http_exception.detail}", exc_info=True)
        raise http_exception

    except Exception as e:
        logger.error(
            f"Error occurred during vehicle detection: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
