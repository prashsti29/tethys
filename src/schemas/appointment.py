from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class AppointmentCreate(BaseModel):
    patient_id: str
    doctor_id: str
    start_time: datetime
    end_time: datetime
    status: str = Field(default="scheduled")


class AppointmentResponse(AppointmentCreate):
    id: str
    created_at: datetime

    class Config:
        from_attributes = True
