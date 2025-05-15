from pydantic import BaseModel, Field


class LicensePlateDetectorRequest(BaseModel):
    image: str = Field(..., description="Base64 encoded image")
    origin: str = Field(..., description="The origin source of the API call")
