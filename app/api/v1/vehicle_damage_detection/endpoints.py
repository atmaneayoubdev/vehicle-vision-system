from fastapi import APIRouter, HTTPException, Request
from app.api.v1.vehicle_damage_detection.schemas import VehicleDamageDetectorRequest
from app.services.model_manager import model_manager
from app.utils.helper import decode_base64_to_image
from app.core.logging import logger
from app.api.v1.vehicle_damage_detection.utils import process_image_with_model


router = APIRouter()

# Get the YOLO model from the ModelManager for vehicle damage detection
vehicle_damage_model = model_manager.get_vehicle_damage_model()


@router.post("/vehicle-damage-detector")
async def vehicle_damage_detector(req: VehicleDamageDetectorRequest, request: Request):
    try:
        if "," in req.image:
            req.image = req.image.split(",")[1]

        # Decode the base64 image to an image format
        image, _, _ = decode_base64_to_image(req.image)

        # Use the utility function to process the image and get results from the model
        detections = process_image_with_model(image, vehicle_damage_model)

        # Filter detections for vehicle damage parts (e.g., "damaged door", "damaged bumper")
        damage_detections = [
            d for d in detections["detections"] if "damaged" in d["label"].lower()]

        if not damage_detections:
            raise HTTPException(
                status_code=404, detail="No vehicle damage detected"
            )

        return {"detections": damage_detections}

    except HTTPException as http_exception:
        logger.error(f"HTTPException: {http_exception.detail}", exc_info=True)
        raise http_exception

    except Exception as e:
        logger.error(
            f"Error occurred during vehicle damage detection: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
