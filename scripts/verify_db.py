#!/usr/bin/env python3
import logging
import sys
from pathlib import Path

# Add project root to python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.db.session import get_db_session
from src.models.entities import Doctor, Patient, ProtocolDoc
from src.schemas.doctor import DoctorResponse
from src.schemas.case_file import IntakeExtractionOutput, SymptomExtraction

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("verify_db")


def main():
    logger.info("=== Verifying Database & Pydantic Schemas ===")

    with get_db_session() as db:
        doctors = db.query(Doctor).all()
        logger.info(f"Retrieved {len(doctors)} doctors from Postgres:")
        for doc in doctors:
            doc_schema = DoctorResponse.model_validate(doc)
            logger.info(f"  - [{doc_schema.specialty}] {doc_schema.name} @ {doc_schema.location} ({len(doc_schema.availability_slots)} slots)")

        protocols = db.query(ProtocolDoc).all()
        logger.info(f"Retrieved {len(protocols)} protocol documents from Postgres:")
        for p in protocols:
            logger.info(f"  - [{p.specialty}] {p.title}")

        patients = db.query(Patient).all()
        logger.info(f"Retrieved {len(patients)} patients:")
        for pt in patients:
            logger.info(f"  - Patient: {pt.name}, History: {pt.medical_history}")

    # Test Pydantic intake validation schema
    test_extraction = IntakeExtractionOutput(
        chief_complaint="Severe left-sided chest pain radiating to left arm",
        symptoms=[
            SymptomExtraction(symptom_name="Chest pain", duration="30 minutes", severity=9, body_location="Left chest"),
            SymptomExtraction(symptom_name="Cold sweat", duration="15 minutes", severity=7)
        ],
        confidence_score=0.95,
        red_flag_detected=True,
        red_flag_reason="Red Flag Rule: Chest pain with radiation to arm",
    )
    logger.info("Validated Pydantic Intake Schema output:")
    logger.info(f"  - Red Flag: {test_extraction.red_flag_detected}")
    logger.info(f"  - Reason: {test_extraction.red_flag_reason}")
    logger.info(f"  - Symptoms: {[s.symptom_name for s in test_extraction.symptoms]}")

    logger.info("=== Database Verification Passed! ===")


if __name__ == "__main__":
    main()
