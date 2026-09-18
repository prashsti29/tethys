from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field


class SymptomExtraction(BaseModel):
    symptom_name: str = Field(description="Name or brief description of the symptom")
    duration: Optional[str] = Field(default=None, description="Duration or onset timing (e.g. '2 hours', '3 days')")
    severity: Optional[int] = Field(default=None, ge=1, le=10, description="Severity rating scale 1 (mild) to 10 (severe)")
    body_location: Optional[str] = Field(default=None, description="Anatomical location (e.g. 'chest', 'left arm')")


class IntakeExtractionOutput(BaseModel):
    chief_complaint: str = Field(description="Primary reason for seeking care")
    symptoms: List[SymptomExtraction] = Field(default_factory=list, description="Extracted individual symptoms")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Extraction confidence score from 0.0 to 1.0")
    follow_up_question: Optional[str] = Field(default=None, description="Adaptive follow-up question if clarification needed")
    red_flag_detected: bool = Field(default=False, description="True if potential emergency red flag is detected")
    red_flag_reason: Optional[str] = Field(default=None, description="Reason or rule triggered for red flag")


class CaseFileCreate(BaseModel):
    patient_id: Optional[str] = None
    chief_complaint: str
    extracted_symptoms: List[SymptomExtraction]
    confidence_score: float = 0.0
    red_flag_triggered: bool = False
    red_flag_reason: Optional[str] = None
    status: str = "intake"


class CaseFileResponse(CaseFileCreate):
    id: str
    created_at: datetime

    class Config:
        from_attributes = True
