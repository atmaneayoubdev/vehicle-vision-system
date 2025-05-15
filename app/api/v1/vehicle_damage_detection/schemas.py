from pydantic import BaseModel, Field


class VehicleDamageDetectorRequest(BaseModel):
    image: str = Field(..., description="Base64 encoded image")
    origin: str = Field(..., description="The origin source of the API call")
