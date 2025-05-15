# app/api/v1/license_plate_detector/endpoint.py
from fastapi import APIRouter, HTTPException, Request
from app.api.v1.license_plate_detector.schemas import LicensePlateDetectorRequest
from app.services.model_manager import model_manager
from app.utils.helper import decode_base64_to_image
from app.api.v1.license_plate_detector.utils import process_image_with_model
import time
from app.core.logging import logger

router = APIRouter()

# Get the YOLO model from the ModelManager
yolo_model = model_manager.get_vehicle_licence_model()


@router.post("/license-plate-detector")
async def license_plate_detector(req: LicensePlateDetectorRequest, request: Request):
    try:
        if "," in req.image:
            req.image = req.image.split(",")[1]

        image, _, _ = decode_base64_to_image(req.image)
        detections = process_image_with_model(image, yolo_model)

        # Filter only for license plate
        plate_detections = [
            d for d in detections["detections"] if d["label"] == "Vehicle Plate"]

        if not plate_detections:
            raise HTTPException(
                status_code=404, detail="License plate not found")

        return {"detections": plate_detections}

    except HTTPException as http_exception:
        logger.error(f"HTTPException: {http_exception.detail}", exc_info=True)
        raise http_exception

    except Exception as e:
        logger.error(
            f"Error occurred during license plate detection: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
