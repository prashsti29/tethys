#!/usr/bin/env python3
import logging
import sys
from pathlib import Path

# Add project root to python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents.intake import IntakeAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("test_intake_agent")


def main():
    logger.info("=== Starting Phase 3 Intake Agent Evaluation ===")
    agent = IntakeAgent()

    test_cases = [
        {
            "name": "Case 1: Clear & Complete Symptom Description",
            "input": "I have had a dull headache for 3 days, severity 4 out of 10.",
            "expect_followup": False,
            "expect_red_flag": False,
        },
        {
            "name": "Case 2: Vague Symptom Description (Missing Severity)",
            "input": "My chest feels weird since this morning.",
            "expect_followup": True,
            "expect_red_flag": False,
        },
        {
            "name": "Case 3: Emergency Red Flag Input",
            "input": "I have sudden severe radiating chest pain down my left arm and cold sweats.",
            "expect_followup": False,
            "expect_red_flag": True,
        },
    ]

    for tc in test_cases:
        logger.info(f"\n--- Running {tc['name']} ---")
        result = agent.extract_case(tc["input"])

        logger.info(f"Chief Complaint   : {result.chief_complaint}")
        logger.info(f"Confidence Score  : {result.confidence_score}")
        logger.info(f"Red Flag Detected : {result.red_flag_detected}")
        if result.red_flag_reason:
            logger.info(f"Red Flag Reason   : {result.red_flag_reason}")
        if result.follow_up_question:
            logger.info(f"Adaptive FollowUp : {result.follow_up_question}")

        logger.info(f"Extracted Symptoms ({len(result.symptoms)}):")
        for s in result.symptoms:
            logger.info(f"  - Name: {s.symptom_name} | Duration: {s.duration} | Severity: {s.severity} | Location: {s.body_location}")

    # Fallback Test
    logger.info("\n--- Running Case 4: Direct Fallback Extractor Test ---")
    fallback_res = agent._fallback_extraction("Patient reports cough for 2 days and fever.")
    logger.info(f"Fallback Chief Complaint : {fallback_res.chief_complaint}")
    logger.info(f"Fallback Symptoms        : {[s.symptom_name for s in fallback_res.symptoms]}")
    logger.info(f"Fallback Confidence      : {fallback_res.confidence_score}")

    logger.info("\n=== Intake Agent Test Completed Successfully! ===")


if __name__ == "__main__":
    main()
