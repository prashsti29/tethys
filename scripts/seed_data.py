#!/usr/bin/env python3
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.db.session import get_db_session, init_db
from src.models.entities import Doctor, Patient, ProtocolDoc

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("seed_data")


def seed_database():
    logger.info("Starting database initialization...")
    init_db()

    with get_db_session() as db:
        # Check existing doctors
        existing_doctors = db.query(Doctor).count()
        if existing_doctors == 0:
            logger.info("Seeding Doctors data...")
            now = datetime.utcnow()
            tomorrow = now + timedelta(days=1)
            day_after = now + timedelta(days=2)

            doctors = [
                Doctor(
                    name="Dr. Sarah Jenkins",
                    specialty="Cardiology",
                    location="Heart & Vascular Center, Suite 301",
                    availability_slots=[
                        tomorrow.replace(hour=9, minute=0, second=0, microsecond=0).isoformat(),
                        tomorrow.replace(hour=11, minute=30, second=0, microsecond=0).isoformat(),
                        day_after.replace(hour=14, minute=0, second=0, microsecond=0).isoformat(),
                    ],
                ),
                Doctor(
                    name="Dr. Marcus Vance",
                    specialty="Neurology",
                    location="Neuroscience Clinic, Suite 405",
                    availability_slots=[
                        tomorrow.replace(hour=10, minute=0, second=0, microsecond=0).isoformat(),
                        day_after.replace(hour=9, minute=30, second=0, microsecond=0).isoformat(),
                        day_after.replace(hour=15, minute=0, second=0, microsecond=0).isoformat(),
                    ],
                ),
                Doctor(
                    name="Dr. Emily Chen",
                    specialty="Primary Care",
                    location="Community Health Hub, Room 102",
                    availability_slots=[
                        tomorrow.replace(hour=8, minute=30, second=0, microsecond=0).isoformat(),
                        tomorrow.replace(hour=13, minute=0, second=0, microsecond=0).isoformat(),
                        tomorrow.replace(hour=16, minute=0, second=0, microsecond=0).isoformat(),
                        day_after.replace(hour=11, minute=0, second=0, microsecond=0).isoformat(),
                    ],
                ),
                Doctor(
                    name="Dr. Aris Thorne",
                    specialty="Dermatology",
                    location="Skin & Wellness Pavilion, Suite 210",
                    availability_slots=[
                        tomorrow.replace(hour=14, minute=30, second=0, microsecond=0).isoformat(),
                        day_after.replace(hour=10, minute=30, second=0, microsecond=0).isoformat(),
                    ],
                ),
                Doctor(
                    name="Dr. Robert Miller",
                    specialty="Orthopedics",
                    location="Bone & Joint Institute, Suite 105",
                    availability_slots=[
                        tomorrow.replace(hour=15, minute=0, second=0, microsecond=0).isoformat(),
                        day_after.replace(hour=13, minute=30, second=0, microsecond=0).isoformat(),
                    ],
                ),
            ]
            db.add_all(doctors)
            logger.info(f"Seeded {len(doctors)} doctors.")

        # Check existing protocol docs
        existing_protocols = db.query(ProtocolDoc).count()
        if existing_protocols == 0:
            logger.info("Seeding Protocol Documents...")
            protocols = [
                ProtocolDoc(
                    title="Emergency Red Flag Rules for Chest Pain",
                    specialty="Emergency Medicine / Cardiology",
                    content=(
                        "CRITICAL EMERGENCY PROTOCOL: Chest pain accompanied by radiation to the left arm, "
                        "jaw pain, shortness of breath, diaphoresis (cold sweats), dizziness, or loss of consciousness "
                        "is a high-risk red flag indicating potential acute coronary syndrome or myocardial infarction. "
                        "IMMEDIATE ACTION: Stop intake process, instruct patient to call 911 / go to nearest Emergency Room immediately. "
                        "Do not schedule routine clinic appointments."
                    ),
                ),
                ProtocolDoc(
                    title="Sudden Severe Headache Protocol",
                    specialty="Emergency Medicine / Neurology",
                    content=(
                        "CRITICAL EMERGENCY PROTOCOL: Thunderclap headache (sudden onset reaching peak intensity within 1 minute), "
                        "headache accompanied by focal neurological deficits, neck stiffness, confusion, fever, or vision loss. "
                        "IMMEDIATE ACTION: Flag for urgent emergency evaluation. Direct patient to emergency services."
                    ),
                ),
                ProtocolDoc(
                    title="Primary Care Routine Triage Protocol",
                    specialty="Primary Care",
                    content=(
                        "Routine symptoms such as mild fatigue, seasonal allergies, cough without dyspnea, "
                        "or low-grade fever (< 100.4F) without red flags should be routed to Primary Care. "
                        "Recommend scheduling an appointment with a General Practitioner within 24 to 48 hours."
                    ),
                ),
                ProtocolDoc(
                    title="Dermatology Rash Evaluation Protocol",
                    specialty="Dermatology",
                    content=(
                        "Localized skin rashes, mild eczema flare-ups, acne, or non-spreading skin lesions. "
                        "RED FLAGS: Rapidly spreading rash with fever, mucosal involvement, or mucosal sloughing (Stevens-Johnson syndrome risk) "
                        "requires emergency evaluation."
                    ),
                ),
            ]
            db.add_all(protocols)
            logger.info(f"Seeded {len(protocols)} protocol documents.")

        # Check existing test patient
        existing_patients = db.query(Patient).count()
        if existing_patients == 0:
            logger.info("Seeding Test Patient...")
            patient = Patient(
                name="John Doe",
                phone="+15550192834",
                date_of_birth="1985-04-12",
                medical_history={"allergies": ["penicillin"], "pre-existing": ["hypertension"]},
            )
            db.add(patient)
            logger.info("Seeded test patient John Doe.")

    logger.info("=== Database Seeding Complete ===")


if __name__ == "__main__":
    seed_database()
