from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field


class DoctorBase(BaseModel):
    name: str
    specialty: str
    location: str
    availability_slots: List[str] = Field(default_factory=list, description="List of ISO 8601 datetime strings")


class DoctorCreate(DoctorBase):
    pass


class DoctorResponse(DoctorBase):
    id: str
    created_at: datetime

    class Config:
        from_attributes = True
