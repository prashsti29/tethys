import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship

from src.db.session import Base


def generate_uuid() -> str:
    return str(uuid.uuid4())


class Patient(Base):
    __tablename__ = "patients"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    name = Column(String(100), nullable=False)
    phone = Column(String(20), nullable=True)
    date_of_birth = Column(String(10), nullable=True)
    medical_history = Column(JSONB, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)

    case_files = relationship("CaseFile", back_populates="patient")
    appointments = relationship("Appointment", back_populates="patient")


class Doctor(Base):
    __tablename__ = "doctors"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    name = Column(String(100), nullable=False)
    specialty = Column(String(50), nullable=False, index=True)
    location = Column(String(100), nullable=False)
    availability_slots = Column(JSONB, default=list)  # List of ISO datetime strings
    created_at = Column(DateTime, default=datetime.utcnow)

    appointments = relationship("Appointment", back_populates="doctor")


class Appointment(Base):
    __tablename__ = "appointments"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    patient_id = Column(String(36), ForeignKey("patients.id"), nullable=False)
    doctor_id = Column(String(36), ForeignKey("doctors.id"), nullable=False)
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=False)
    status = Column(String(20), default="scheduled")  # scheduled, completed, cancelled
    created_at = Column(DateTime, default=datetime.utcnow)

    patient = relationship("Patient", back_populates="appointments")
    doctor = relationship("Doctor", back_populates="appointments")


class CaseFile(Base):
    __tablename__ = "case_files"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    patient_id = Column(String(36), ForeignKey("patients.id"), nullable=True)
    chief_complaint = Column(Text, nullable=True)
    extracted_symptoms = Column(JSONB, default=list)
    confidence_score = Column(Float, default=0.0)
    red_flag_triggered = Column(Boolean, default=False)
    red_flag_reason = Column(Text, nullable=True)
    status = Column(String(30), default="intake")  # intake, escalated, scheduled
    created_at = Column(DateTime, default=datetime.utcnow)

    patient = relationship("Patient", back_populates="case_files")


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    event_type = Column(String(50), nullable=False, index=True)
    agent_name = Column(String(50), nullable=False)
    payload = Column(JSONB, default=dict)
    timestamp = Column(DateTime, default=datetime.utcnow)


class ProtocolDoc(Base):
    __tablename__ = "protocol_docs"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    title = Column(String(150), nullable=False)
    specialty = Column(String(50), nullable=False, index=True)
    content = Column(Text, nullable=False)
    embedding = Column(Vector(384), nullable=True)  # sentence-transformers all-MiniLM-L6-v2 dimension
    created_at = Column(DateTime, default=datetime.utcnow)
