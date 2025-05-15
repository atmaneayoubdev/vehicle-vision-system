# app/core/schemas.py
from pydantic import BaseModel, Field
from typing import Literal


class RequestStats(BaseModel):
    endpoint_name: str
    origin: Literal["internal", "external"]
    user_id: str
    # Time taken should be a positive integer
    time_taken: int = Field(..., ge=0)
    status_code: int = Field(..., ge=100, le=599)  # Valid HTTP status codes

    class Config:
        schema_extra = {
            "example": {
                "endpoint_name": "example_endpoint",
                "origin": "internal",
                "user_id": "user_id",
                "time_taken": 123,
                "status_code": 200
            }
        }
