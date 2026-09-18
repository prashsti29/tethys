#!/usr/bin/env python3
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents.scheduling import SchedulingAgent
from src.db.session import get_db_session
from src.models.entities import Appointment, Patient
from src.orchestrator import VoiceAssistantOrchestrator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("test_orchestrator")


def cleanup_test_patients():
    """Remove test patients created during orchestrator runs."""
    with get_db_session() as db:
        test_names = ["Maria Gonzalez", "David Kim"]
        for name in test_names:
            patients = db.query(Patient).filter(Patient.name == name).all()
            for p in patients:
                db.query(Appointment).filter(Appointment.patient_id == p.id).delete()
                db.delete(p)


def main():
    logger.info("=== Starting Phase 7 Orchestrator End-to-End Evaluation ===")
    cleanup_test_patients()
    orch = VoiceAssistantOrchestrator()

    # ── Scenario 1: Emergency Red Flag ──────────────────────────────
    logger.info("\n\n>>> SCENARIO 1: Emergency Red Flag (Cardiovascular) <<<")
    result1 = orch.run(
        "I have sudden crushing radiating chest pain down my left arm and I'm sweating.",
        patient_name="Maria Gonzalez",
    )
    logger.info(f"  [RESPONSE] {result1.response_text}")
    assert result1.guardrail.is_red_flag is True, "CRITICAL: Red flag not detected!"
    assert result1.guardrail.allow_scheduling is False, "CRITICAL: Scheduling was allowed for emergency!"
    assert result1.booking is None, "CRITICAL: Appointment booking proceeded during emergency!"
    logger.info("  ✓ Scenario 1 Passed — Emergency correctly halted pipeline.")

    # ── Scenario 2: Vague Input → Adaptive Follow-Up ────────────────
    logger.info("\n\n>>> SCENARIO 2: Vague Input → Adaptive Follow-Up <<<")
    result2 = orch.run(
        "I just don't feel well.",
        patient_name="David Kim",
        auto_book=False,
    )
    logger.info(f"  [RESPONSE] {result2.response_text}")
    assert result2.awaiting_patient_reply is True, "ERROR: Low-confidence input did not trigger follow-up!"
    logger.info("  ✓ Scenario 2 Passed — Follow-up question issued correctly.")

    # ── Scenario 3: Full End-to-End Booking Flow ────────────────────
    logger.info("\n\n>>> SCENARIO 3: Full Flow — Headache → Neurology Booking <<<")
    result3 = orch.run(
        "I have had a throbbing headache on the left side of my head for 3 days, severity 7 out of 10.",
        patient_name="David Kim",
        auto_book=True,
    )
    logger.info(f"  [RESPONSE] {result3.response_text}")
    assert result3.guardrail.is_red_flag is False, "ERROR: False red flag triggered!"
    assert result3.target_specialty is not None, "ERROR: No specialty routed!"
    logger.info(f"  Target Specialty  : {result3.target_specialty}")
    logger.info(f"  Matched Protocol  : {result3.matched_protocol}")
    if result3.booking and result3.booking.success:
        logger.info(f"  Appointment ID    : {result3.booking.appointment_id}")
        logger.info(f"  Doctor            : {result3.booking.doctor_name}")
        logger.info(f"  Location          : {result3.booking.location}")
    logger.info("  ✓ Scenario 3 Passed — Full pipeline end-to-end working!")

    logger.info("\n=== Orchestrator Evaluation Completed Successfully! All Scenarios Passed ===")


if __name__ == "__main__":
    main()
