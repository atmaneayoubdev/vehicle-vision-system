from contextlib import asynccontextmanager
import os
from fastapi import FastAPI, HTTPException, Request
from app.api.v1.license_plate_detector.endpoints import router as v1_license_plate_detector
from app.api.v1.vehicle_detector.endpoints import router as v1_vehicle_detector
from app.api.v1.vehicle_damage_detection.endpoints import router as v1_vehicle_damage_detector
from app.core.middleware import setup_middleware
import torch
from app.core.logging import configure_logger, logger
from app.services.model_manager import model_manager
# Import the lifespan context manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models at startup
    # This will call the method correctly
    model_manager
    yield

# Create FastAPI app instance and pass lifespan for startup/shutdown handling
app = FastAPI(lifespan=lifespan)

configure_logger()
# Set up middleware
setup_middleware(app)

# Include API routers
app.include_router(v1_license_plate_detector, prefix="/api/v1")
app.include_router(v1_vehicle_detector, prefix="/api/v1")
app.include_router(v1_vehicle_damage_detector, prefix="/api/v1")


@app.get("/health-check")
def home():
    return {"Health Check": "OK"}


@app.get("/check-gpu")
async def check_gpu(request: Request):
    user_id = getattr(request.state, 'user_id', None)
    logger.info(f"User ID from request state: {user_id}")

    if torch.cuda.is_available():
        gpu_info = {
            "device_count": torch.cuda.device_count(),
            "device_name": torch.cuda.get_device_name(0),
            "current_device": torch.cuda.current_device(),
        }
        return {"GPU Status": "Available", "Details": gpu_info}
    else:
        return {"GPU Status": "Not Available"}
