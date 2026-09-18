#!/usr/bin/env python3
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents.scheduling import SchedulingAgent
from src.db.session import get_db_session
from src.models.entities import Appointment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("test_scheduling")


def main():
    logger.info("=== Starting Phase 6 Scheduling Agent Evaluation ===")
    agent = SchedulingAgent()

    # ── Test 1: Query Available Slots for Cardiology ────────────────────────
    logger.info("\n--- Case 1: Query Available Cardiology Slots ---")
    slots = agent.find_available_slots("Cardiology")
    assert len(slots) > 0, "ERROR: No Cardiology slots returned!"
    logger.info(f"Found {len(slots)} Cardiology slots:")
    for s in slots:
        logger.info(f"  - {s.doctor_name} | {s.slot_time} | {s.location}")

    # ── Test 2: Book First Available Slot ───────────────────────────────────
    first_slot = slots[0]
    logger.info(f"\n--- Case 2: Book Slot for 'Alice Smith' with {first_slot.doctor_name} ---")
    result = agent.book_appointment(
        doctor_id=first_slot.doctor_id,
        slot_time=first_slot.slot_time,
        patient_name="Alice Smith",
    )
    assert result.success, f"ERROR: Booking failed — {result.message}"
    logger.info(f"Booking Result: {result.message}")
    logger.info(f"Appointment ID: {result.appointment_id}")

    # ── Test 3: Confirm it's in the Database ────────────────────────────────
    logger.info("\n--- Case 3: Verify Appointment Persisted in Postgres ---")
    with get_db_session() as db:
        appt = db.query(Appointment).filter(Appointment.id == result.appointment_id).first()
        assert appt is not None, "ERROR: Appointment not found in database!"
        assert appt.status == "scheduled", f"ERROR: Expected status 'scheduled', got '{appt.status}'"
        logger.info(f"  Appointment {appt.id} | Status: {appt.status} | Start: {appt.start_time}")

    # ── Test 4: Double-Booking Prevention ───────────────────────────────────
    logger.info("\n--- Case 4: Double-Booking Prevention Check ---")
    result2 = agent.book_appointment(
        doctor_id=first_slot.doctor_id,
        slot_time=first_slot.slot_time,
        patient_name="Bob Jones",
    )
    assert not result2.success, "ERROR: Double-booking was allowed!"
    logger.info(f"Double-book correctly rejected: {result2.message}")

    # ── Test 5: Query Primary Care Slots ────────────────────────────────────
    logger.info("\n--- Case 5: Query Available Primary Care Slots ---")
    pc_slots = agent.find_available_slots("Primary Care")
    assert len(pc_slots) > 0, "ERROR: No Primary Care slots returned!"
    logger.info(f"Found {len(pc_slots)} Primary Care slots:")
    for s in pc_slots:
        logger.info(f"  - {s.doctor_name} | {s.slot_time}")

    logger.info("\n=== Scheduling Agent Evaluation Completed Successfully! All Assertions Passed ===")


if __name__ == "__main__":
    main()
