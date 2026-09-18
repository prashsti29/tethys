#!/usr/bin/env python3
import logging
import sys
from pathlib import Path

# Add project root to python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.db.session import get_db_session
from src.guardrails.red_flags import RedFlagClassifier
from src.models.entities import AuditLog

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("test_guardrails")


def main():
    logger.info("=== Starting Phase 4 Guardrails Evaluation ===")
    classifier = RedFlagClassifier()

    test_cases = [
        {
            "name": "Case 1: Cardiovascular Emergency (Radiating Chest Pain)",
            "input": "I have sudden crushing radiating chest pain down my left arm and cold sweats.",
            "expect_red_flag": True,
            "expect_category": "CARDIOVASCULAR",
        },
        {
            "name": "Case 2: Neurological Stroke Emergency (Thunderclap Headache & Facial Droop)",
            "input": "I have a thunderclap headache and sudden facial droop.",
            "expect_red_flag": True,
            "expect_category": "NEUROLOGICAL",
        },
        {
            "name": "Case 3: Respiratory Distress (Gasping for air & Cyanosis)",
            "input": "I am gasping for air and my lips are turning blue.",
            "expect_red_flag": True,
            "expect_category": "RESPIRATORY",
        },
        {
            "name": "Case 4: Routine Care Input (Mild Cough)",
            "input": "I have had a mild cough and runny nose for 2 days.",
            "expect_red_flag": False,
            "expect_category": None,
        },
    ]

    for tc in test_cases:
        logger.info(f"\n--- Evaluating {tc['name']} ---")
        decision = classifier.evaluate(tc["input"])

        logger.info(f"Is Red Flag       : {decision.is_red_flag}")
        logger.info(f"Allow Scheduling  : {decision.allow_scheduling}")
        logger.info(f"Flag Human Review : {decision.flag_human_review}")

        if decision.is_red_flag:
            logger.info(f"Category Triggered: {decision.category}")
            logger.info(f"Rule Matched      : {decision.rule_triggered}")
            logger.info(f"ER Instruction    : {decision.emergency_instruction}")

            assert decision.allow_scheduling is False, "CRITICAL ERROR: Scheduling was allowed during red flag emergency!"
            assert decision.flag_human_review is True, "CRITICAL ERROR: Case was not flagged for human review!"
        else:
            assert decision.allow_scheduling is True, "ERROR: Routine case scheduling was blocked!"

    # Verify Audit Logs in Postgres
    logger.info("\n--- Verifying Audit Trail in Postgres Database ---")
    with get_db_session() as db:
        logs = db.query(AuditLog).filter(AuditLog.event_type == "RED_FLAG_TRIGGERED").all()
        logger.info(f"Persisted Red-Flag Audit Logs ({len(logs)} entries found):")
        for log in logs:
            logger.info(f"  - [{log.timestamp}] Category: {log.payload.get('category')} | Rule: {log.payload.get('rule_triggered')}")

    logger.info("\n=== Guardrails Evaluation Completed Successfully! All Gatekeeper Assertions Passed ===")


if __name__ == "__main__":
    main()
